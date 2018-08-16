#! /usr/bin/env python3

import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

parser = argparse.ArgumentParser(description="Process arguments.")
parser.add_argument('--dir1', required=True, help="Directory containing original images")
parser.add_argument('--dir2', required=True, help="Directory containing modified images")
args = parser.parse_args()

module_url = "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/2"

print("open module at ", module_url)
inception_module = hub.Module(module_url)
height, width = hub.get_expected_image_size(inception_module)
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
        for file in img_files:
            arr.append(sess.run(img, feed_dict={image_file: file}))

    return np.concatenate(arr)


def image_features(images):
    outputs = inception_module(images)
    features = outputs
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    features = sess.run(features)
    sess.close()
    return features
    

def l1diff(x, y, axis=1):
    diff = np.sum(np.abs(x - y), axis=axis, keepdims=False)
    return diff


def l2diff(x, y, axis=1):
    diff = np.sqrt(np.sum(np.square(x-y), axis=axis, keepdims=False))
    return diff


print("process images in ", args.dir1)
images1 = process_image_files(args.dir1, height, width)
print("   found {0} images".format(len(images1)))

print("process images in ", args.dir2)
images2 = process_image_files(args.dir2, height, width)
print("   found {0} images".format(len(images2)))


print("Calc feature vectors")
features1 = image_features(images1)

print("Calc feature vectors")
features2 = image_features(images2)

batch_size, feature_length = features1.shape
print("feature vector length: ", feature_length)

print("Similarity measures.")
l2_inter_distances = l2diff(features1, features2)

inter_mean = np.mean(l2_inter_distances)
inter_median = np.median(l2_inter_distances)
inter_stddev = np.std(l2_inter_distances)
inter_max = np.max(l2_inter_distances)
inter_min = np.min(l2_inter_distances)

print("  mean ", inter_mean)
print("  median ", inter_median)
print("  std dev ", inter_stddev)
print("  max ", inter_max)
print("  min ", inter_min)


print("Dissimilarity measures.")
m, n  = features1.shape
l2_intras = []
for i in range(0,5):
    x = features1[i,:]
    for j in range(i+1, m):
        y = features1[j,:]
        l2 = l2diff(x,y, axis=0)
        l2_intras.append(l2)
        
l2_intra_distances = np.array(l2_intras)

intra_mean = np.mean(l2_intra_distances)
intra_median = np.median(l2_intra_distances)
intra_stddev = np.std(l2_intra_distances)
intra_max = np.max(l2_intra_distances)
intra_min = np.min(l2_intra_distances)

print("  mean ", intra_mean)
print("  median ", intra_median)
print("  std dev ", intra_stddev)
print("  max ", intra_max)
print("  min ", intra_min)

nbins = 20
plt.hist([l2_inter_distances, l2_intra_distances], bins=nbins,
         color=['cyan','lime'],
         density=True,
         histtype='barstacked')
plt.xlabel("l2 distance")
plt.ylabel("count")
plt.title("Distances Between Similar/Dissimilar Image Features")
plt.show()

print("Done.")

    


