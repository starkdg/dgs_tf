#! /usr/bin/env python3
import convphash
import argparse
import matplotlib.pyplot as plt
import numpy as np

""" Script to compare images in two directories, one being
    a slightly modified version of the others. The image files
    must have the same name so that similar images are compared.
"""


parser = argparse.ArgumentParser(description="Compare Image files.")
parser.add_argument('--dir1', required=True,
                    help="Directory containing original images")
parser.add_argument('--dir2', required=True,
                    help="Directory containing modified images")
args = parser.parse_args()

module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"

print("hub module: ", module_url)
cp = convphash.ConvPhash(module_url)


print("process images in ", args.dir1)
images1 = cp.process_image_files(args.dir1)
phash1, p1_min, p1_max = cp.image_phashes(images1, ndims=256)
print("phash size: ", phash1.shape)
print("range=({0},{1})".format(p1_min, p1_max))

print("process images in ", args.dir2)
images2 = cp.process_image_files(args.dir2)
phash2, p2_min, p2_max = cp.image_phashes(images2, ndims=256)
print("phash size: ", phash2.shape)
print("range=({0},{1})".format(p2_min, p2_max))

print("Similarity measures.")
hamming_inter_distances = cp.hamming_distance(phash1, phash2, axis=0)

inter_mean = np.mean(hamming_inter_distances)
inter_median = np.median(hamming_inter_distances)
inter_stddev = np.std(hamming_inter_distances)
inter_max = np.max(hamming_inter_distances)
inter_min = np.min(hamming_inter_distances)

print("  mean ", inter_mean)
print("  median ", inter_median)
print("  std dev ", inter_stddev)
print("  max ", inter_max)
print("  min ", inter_min)


print("Dissimilarity measures.")
m, n = phash1.shape
hamming_intras = []
for i in range(0, 5):
    x = phash1[:, i]
    for j in range(i+1, n):
        y = phash1[:, j]
        hdist = cp.hamming_distance(x, y, axis=0)
        hamming_intras.append(hdist)

hamming_intra_distances = np.array(hamming_intras)
intra_mean = np.mean(hamming_intra_distances)
intra_median = np.median(hamming_intra_distances)
intra_stddev = np.std(hamming_intra_distances)
intra_max = np.max(hamming_intra_distances)
intra_min = np.min(hamming_intra_distances)

print("  mean ", intra_mean)
print("  median ", intra_median)
print("  std dev ", intra_stddev)
print("  max ", intra_max)
print("  min ", intra_min)

nbins = 20
plt.hist([hamming_inter_distances, hamming_intra_distances], bins=nbins,
         color=['cyan', 'lime'],
         density=True,
         histtype='barstacked')
plt.xlabel("hamming distance")
plt.ylabel("counts")
plt.title("Distances Between Similar/Dissimilar Image Features")
plt.show()
print("Done.")
