import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# graph for image input extraction
image_file = tf.placeholder(tf.string)
height = tf.placeholder(tf.int32)
width = tf.placeholder(tf.int32)
img_data = tf.read_file(image_file)
img = tf.image.decode_jpeg(img_data, channels=3)
img = tf.image.convert_image_dtype(img, dtype=tf.float32)
img = tf.image.central_crop(img, central_fraction=0.875)
img = tf.expand_dims(img, 0)
img = tf.image.resize_bilinear(img, [height, width], align_corners=False)
img = tf.subtract(img, 0.5)
img = tf.multiply(img, 2.0)

class ConvPhash:
    """ Convolutional Perceptual Hash
    """
    
    def __init__(self, module_url, ufile="/tmp/utranpose"):
        """ init method 
        args
        module_url -- url of hub module
        ufile      -- file to save the feature transform matrix from SVD
                      (optional)
        """
        self.module = hub.Module(module_url)
        self.h, self.w = hub.get_expected_image_size(self.module)
        self.ufile = ufile

        
    def process_image_files(self, img_dir, limit=None):
        """ process images from files in a directory
        args
        img_dir -- directory of jpeg images
        limit   -- maximum number of images to process from directory
                   (optional - defaults to no limit)
        return
        np.ndarray of images (size: n_images x h x w x 3)
        """

        os.chdir(img_dir)
        arr = []
        count = 0
        with tf.Session() as sess:
            for entry in os.scandir(img_dir):
                if entry.is_file() and entry.name.endswith('.jpg'):
                    arr.append(sess.run(img, feed_dict={image_file: entry.name,
                                                        height: self.h, width: self.w}))
                    count = count + 1
                if limit != None and count > limit :
                    break
        return np.concatenate(arr)


    def image_features(self, images):
        """ Get feature vectors for images from module
        args
        images -- ndarray of images obtained from process_image_files()
                  size: [no_images x h x w x 3] 
        return
        features -- ndarray (size no_images x feature_length)
        """
        outputs = self.module(images)
        features = outputs
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        features = sess.run(features)
        sess.close()
        return features


    def compute_transform(self, img_dir, limit=500):
        """ Perform SVD on covariance matrix of a list of
            image features from img_dir; save the tranpose
            of the U matrix, Utransp.
            
        args
        img_dir -- directory of images 
        limit   -- limit number of files (defaults to 500) 
        return 
        s       -- 1-D array of singular values
        """

        images = self.process_image_files(img_dir, limit=limit)
        features = self.image_features(images)
        cm = np.corrcoef(features, rowvar=False)
        u, s, v = np.linalg.svd(cm, full_matrices=False, compute_uv=True)
        u_transpose = np.transpose(u)
        np.savez_compressed(self.ufile, u=u_transpose)
        return s


    def load_ufile(self, ndims=256):
        """ load Utranspose from file
        args
        ndims -- no. of leading rows to use from matrix.
        """
        file = self.ufile + ".npz"
        uloaded = np.load(file)
        self.utranspose = uloaded['u']
        self.utranspose = self.utranspose[0:ndims, :]
        self.utranspose = self.utranspose.astype(np.float32)
        
    def image_phashes(self, images, ndims=256):
        """ Compute condensed image perceptual hashes from features.
        args
        images -- ndarray of preprocessed images (size: [no_images x h x w x 3])
        ndims  -- final no. dimensions for  hash. (optional)
        return -- ndarray of final hashes (size: feature_length x n_images)
        """
        # load file if not already loaded
        if not hasattr(self, 'utranspose') or not isinstance(self.utranspose, np.ndarray):
            self.load_ufile(ndims)
            
        # quantize feature vectors int bytes
        outputs  = self.module(images)
        features = tf.matmul(self.utranspose, outputs, transpose_a=False, transpose_b=True)
        qfeatures, fmin, fmax = tf.quantize(features, -50., 50., tf.quint8, mode='MIN_COMBINED')
        qfeatures = tf.bitcast(qfeatures, tf.uint8)
        

        # convert quantized features to a gray code
        m, n = qfeatures.shape
        one = tf.constant(1, dtype=tf.uint8)
        shifted_features = tf.bitwise.right_shift(qfeatures, one)
        gray_features = tf.bitwise.bitwise_xor(qfeatures, shifted_features)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        rfeatures, fmin, fmax = sess.run([gray_features, fmin, fmax])
        sess.close()
        return rfeatures, fmin, fmax

    def l1diff(self, x, y, axis=1):
        """ L1 distance metric
        args
        x -- ndarray of values (size: [height x width])
        y -- ndarray of values (same size as x)
        axis -- axis along which to sum elements
        return
        1-D ndarray of l1 diffs (axis dimension eliminated)
        
        """
        diff = np.sum(np.abs(x - y), axis=axis, keepdims=False)
        return diff


    def l2diff(self, x, y, axis=1):
        """ L2 distance metric
        args
        x -- ndarray of values (size: [height x width])
        y -- ndarray (same size)
        axis -- axis along which to sum elements
        return
        1-D ndarray of l2 diffs (axis dimension eliminated)
        """
        diff = np.sqrt(np.sum(np.square(x-y), axis=axis, keepdims=False))
        return diff


    def bitcount(self, x):
        """ count number of set bits in x argparse
        arg
        x -- int value
        return
        count of set bits
        """
        count = 0
        while (x):
            x &= x - 1
            count += 1
        return count

    def hamming_distance(self, x, y, axis=1):
        """ hamming distance metric
        args
        x -- 1-D ndarray of byte (or int) values
        y -- (same as x)
        axis -- axis along which to sum elements
        return
        1-D ndarray of hamming distances 
        """
        x_xor_y = np.bitwise_xor(x, y)
        apply_bitcount = np.vectorize(self.bitcount)
        counts = apply_bitcount(x_xor_y)
        diff = np.sum(counts, axis=axis, keepdims=False)
        return diff

    
if __name__ == "__main__":
    """ Utility to perform SVD Decomposition on covariance matrix
        of the feature vectors derived from a directory of images. 
        This provides the transform matrix to reduce the feature 
        vector dimension to a more compact representation.
    """
    import argparse
    import matplotlib.pyplot as plt
    from matplotlib import colors

    parser = argparse.ArgumentParser(description="Perform SVD Decomposition.")
    parser.add_argument("--dir", required=True, help="Directory of images")

    args = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.FATAL)
    
    module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"
    print("hub module: ", module_url)
    cp = ConvPhash(module_url)

    print("Compute svd on files in dir, ", args.dir)
    s = cp.compute_transform(args.dir)

    print("Normalized cumulative sum of singular values")
    cumsum_s = np.cumsum(s)
    cumsum_s /= np.amax(cumsum_s)

    plt.plot(cumsum_s)
    plt.title("Cumulative Sum of Singular Values")
    plt.xlabel("Singular Values")
    plt.ylabel("Magnitude")
    plt.show()
    print("Done.")

