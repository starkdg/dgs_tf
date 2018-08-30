#! /usr/bin/env python3
import convphash
import argparse
import numpy as np
from sklearn.neighbors import BallTree



def hamming_distance(x, y, axis=1):
    """ hamming distance metric
    args
    x -- 1-D ndarray of byte (or int) values
    y -- (same as x)
    axis -- axis along which to sum elements
    return
    1-D ndarray of hamming distances 
    """
    def bitcount(elem):
        """ count number of set bits in x argparse
        arg
        x -- int value
        return
        count of set bits
        """
        count = 0
        while (elem):
            elem &= elem - 1
            count += 1
            return count

    x_xor_y = np.bitwise_xor(x, y)
    apply_bitcount = np.vectorize(self.bitcount)
    counts = apply_bitcount(x_xor_y)
    diff = np.sum(counts, axis=axis, keepdims=False)
    return diff


parser = argparse.ArgumentParser(description="Index Image files with BallTree.")
parser.add_argument('--dir', required=True, help="Directory containing images to index")
args = parser.parse_args()

module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"

print("hub module: ", module_url)
cp = convphash.ConvPhash(module_url)


print("process images in ", args.dir)
images = cp.process_image_files(args.dir, limit=100)
phashes, p1_min, p1_max = cp.image_phashes(images, ndims=256)
print("phash size: ", phashes.shape)
print("range=({0},{1})".format(p1_min, p1_max))

points = np.transpose(phashes)

print("build tree")
tree = BallTree(points, leaf_size=10, metric=hamming_distance)

querypoint = points[3, :]
print("query point ", querypoint)
dist, ind =  tree.query(querypoint, k=2)
print("results:")
print(ind)
print(dist)


print("Done.")
