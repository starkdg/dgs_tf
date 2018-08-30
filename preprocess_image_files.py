#! /usr/bin/env python3
import argparse
import os
import skimage
import numpy as np
import matplotlib
from skimage import io, filters, transform, util, draw

parser = argparse.ArgumentParser("Preprocess image files.")
parser.add_argument("--dir",
                    required=True,
                    help="directory of images to process")
args = parser.parse_args()
img_dir = args.dir

print("process images in ", img_dir)
os.chdir(path=img_dir)

orig_dir = "original"
blur_dir = "blurred"
compr_dir = "compressed"
cropped_dir = "cropped"
decimated_dir = "decimated"
noise_dir = "noise"
occluded_dir = "occluded"
rotated_dir = "rotated"

try:
    print("Create Directories")
    os.mkdir(orig_dir)
    os.mkdir(blur_dir)
    os.mkdir(compr_dir)
    os.mkdir(cropped_dir)
    os.mkdir(decimated_dir)
    os.mkdir(noise_dir)
    os.mkdir(occluded_dir)
    os.mkdir(rotated_dir)
except FileExistsError:
    print("Directories already exist.")


def blur_image_and_save(file, img):
    path = os.path.join(img_dir, blur_dir, file)
    blurred_img = filters.gaussian(img, sigma=1, multichannel=True)
    io.imsave(path, blurred_img, plugin="pil", quality=100)


def compress_image_and_save(file, img):
    path = os.path.join(img_dir, compr_dir, file)
    io.imsave(path, img, plugin="pil", quality=30)


def crop_image_and_save(file, img):
    path = os.path.join(img_dir, cropped_dir, file)
    h, w, d = img.shape
    w_margin = int(w*0.10)
    h_margin = int(h*0.10)
    cropped = util.crop(img, ((h_margin, h_margin), (w_margin, w_margin),(0,0)), copy=True)
    io.imsave(path, cropped, plugin="pil", quality=100)


def decimate_image_and_save(file, img):
    path = os.path.join(img_dir, decimated_dir, file)
    resized = transform.resize(img, [128, 128], mode='reflect',
                               anti_aliasing=True)
    io.imsave(path, resized, plugin="pil", quality=100)


def noise_image_and_save(file, img):
    path = os.path(img_dir, noise_dir, file)
    noisy = util.random_noise(img, mode='s&p')
    io.imsave(path, noisy, plugin="pil", quality=100)


def occlude_image_and_save(file, img):
    path = os.path.join(img_dir, occluded_dir, file)
    h, w, d = img.shape

    row_start = int(h*0.80)
    row_end = int(h*0.90)
    col_start = int(w*0.10)
    col_end = int(w*0.90)
    occluded_img = np.copy(img)
    occluded_img[row_start:row_end, col_start:col_end, :] = 255
    io.imsave(path, occluded_img, plugin="pil", quality=100)


def rotate_image_and_save(file, img):
    path = os.path.join(img_dir, rotated_dir, file)
    rotd = transform.rotate(img, 10., resize=False)
    io.imsave(path, rotd, plugin="pil", quality=100)


def save_original(file, img):
    path = os.path.join(img_dir, orig_dir, file)
    io.imsave(path, img, plugin="pil", quality=100)

    
count = 1
for entry in os.scandir(args.dir):
    if entry.is_file() and entry.name.endswith(".jpg"):
        print("({0}) : {1}".format(count, entry.name))
        img = io.imread(entry.name, plugin='pil')
        if  len(img.shape) == 3 and img.shape[2] >= 3:
            save_original(entry.name, img)
            blur_image_and_save(entry.name, img)
            compress_image_and_save(entry.name, img)
            crop_image_and_save(entry.name, img)
            decimate_image_and_save(entry.name, img)
            noise_image_and_save(entry.name, img)
            occlude_image_and_save(entry.name, img)
            rotate_image_and_save(entry.name, img)
        os.remove(entry.name)
        count = count + 1
print("Done.")
