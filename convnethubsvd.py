#! /usr/bin/env python3

import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

parser = argparse.ArgumentParser(description="Process arguments.")
parser.add_argument('--dir', required=True, help="Directory of Images")
args = parser.parse_args()

module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"

print("open module at ", module_url)
module = hub.Module(module_url)
height, width = hub.get_expected_image_size(module)
print("expect image size: {0}x{1}".format(height, width))

image_file = tf.placeholder(tf.string)
img_data = tf.read_file(image_file)
img = tf.image.decode_jpeg(img_data, channels=3)
img = tf.image.convert_image_dtype(img, dtype=tf.float32)
img = tf.image.central_crop(img, central_fraction=0.875)
img = tf.expand_dims(img, 0)
img = tf.image.resize_bilinear(img, [height, width], align_corners=False)
img = tf.subtract(img, 0.5)
img = tf.multiply(img, 2.0)


def process_image_files(img_dir, height, width):
    img_files = sorted(os.listdir(img_dir))
    os.chdir(img_dir)

    arr = []
    with tf.Session() as sess:
        for index in range(0, 500):
            arr.append(sess.run(img, feed_dict={image_file: img_files[index]}))

    return np.concatenate(arr)


def image_features(images):
    outputs = module(images)
    features = outputs
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    features = sess.run(features)
    sess.close()
    return features
    

print("process images in ", args.dir)
images = process_image_files(args.dir, height, width)
print("   found {0} images".format(len(images)))


print("Calc feature vectors")
features = image_features(images)
features = np.transpose(features)

m, n = features.shape
print("M: ", m)
print("N: ", n)


print("compute svd")
u, s, v =  np.linalg.svd(features, full_matrices=False, compute_uv=True)

max_sv = np.amax(s)
min_sv = np.amin(s)

print("max s.v. ", max_sv)
print("min s.v. ", min_sv)


print(u.shape)
print(s.shape)
print(v.shape)

cumsum_s = np.cumsum(s)
cumsum_s /= np.amax(cumsum_s)

plt.plot(cumsum_s)
plt.show()


print("Done.")

    


