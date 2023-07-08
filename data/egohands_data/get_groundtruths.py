import cv2
import os
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm


def get_boundaries_per_subdir(subdir_name, mat_file):
    jpeg_files = sorted([x for x in os.listdir(subdir_name) if '.jpg' in x])
    polygons = scipy.io.loadmat(mat_file)['polygons']
    for i, jpeg_file in enumerate(jpeg_files):
        name = jpeg_file[:-len('.jpg')]
        jpeg_file_path = os.path.join(subdir_name, jpeg_file)
        img = cv2.imread(jpeg_file_path)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for j in range(4):
            if polygons[0][i][j].shape[1] == 0:
                continue
            coordinates = [(x[0], x[1]) for x in polygons[0][i][j]]
            img_new = Image.new('L', (img.shape[1], img.shape[0]), 0)
            ImageDraw.Draw(img_new).polygon(coordinates, outline=1, fill=1)
            mask_new = np.array(img_new)
            mask = mask | mask_new
        os.makedirs(os.path.join('masks', os.path.basename(subdir_name)), exist_ok=True)
        np.save(os.path.join('masks', os.path.basename(subdir_name), f"{name}_mask.npy"), mask)

def get_masks_for_all(root_dir):
    sub_folders = [x for x in os.listdir(root_dir) if x != ".DS_Store"]
    for sub_folder in tqdm(sub_folders):
        sub_dir = os.path.join(root_dir, sub_folder)
        polygon = os.path.join(root_dir, sub_folder, "polygons.mat")
        get_boundaries_per_subdir(sub_dir, polygon)

if __name__ == "__main__":
    root_dir = '_LABELLED_SAMPLES'
    get_masks_for_all(root_dir)