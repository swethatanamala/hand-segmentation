import cv2
import numpy as np
import os
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class EgoHandsDataset(Dataset):
    def __init__(self, data_folder, mode="train", data_limit=None, transforms=None):
        self.transforms = transforms
        self.data_folder = data_folder
        self.all_images = sorted(glob(f"{data_folder}/images/*/*.jpg"))
        #self.all_masks = sorted(glob(f"{data_folder}/masks/*/*.npy"))
        self.all_masks = [os.path.join(data_folder, "masks", x.split('/')[-2:])[:-len('.jpg')] + "_mask.npy"
                          for x in self.all_images]
        self.train_val_dict = self.get_split()
        self.mode = mode
        if data_limit:
            self.train_val_dict[self.mode]["images"] = self.train_val_dict[self.mode]["images"][:data_limit]
            self.train_val_dict[self.mode]["masks"] = self.train_val_dict[self.mode]["masks"][:data_limit]
        assert len(self.train_val_dict[self.mode]["images"]) == len(self.train_val_dict[self.mode]["masks"]), \
            "images and masks length should match"
        self.images, self.masks = self.train_val_dict[self.mode]["images"], self.train_val_dict[self.mode]["masks"]
        self.transforms = transforms

    def get_split(self):
        names = list(set([os.path.basename(os.path.dirname(x)) for x in self.all_images]))
        train_len = int(len(names) * 0.7) + 1
        train_val_dict = {"train": {
                            "images": [filepath for filepath in self.all_images
                                       for name in names[:train_len] if name in filepath],
                            "masks": [filepath for filepath in self.all_masks
                                      for name in names[:train_len] if name in filepath]},
                         "val": {
                            "images": [filepath for filepath in self.all_images
                                        for name in names[train_len:] if name in filepath],
                            "masks": [filepath for filepath in self.all_masks
                                      for name in names[train_len:] if name in filepath]}
                        }
        return train_val_dict
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]
        img_name = os.path.basename(img_path)[:-len(".jpg")]
        mask_name = os.path.basename(mask_path)[:-len("_mask.npy")]
        assert img_name == mask_name, "image and mask should match"
        img = cv2.imread(img_path) / 255
        mask = np.load(mask_path)
        if self.transforms:
            transformed = self.transforms[self.mode]({"image": img, "target": mask})
            return transformed["image"], transformed["target"]
        else:
            return img, mask