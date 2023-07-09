import cv2
import numpy as np
import os
import torch
from . import transforms as tsfms
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class EgoHandsDataset(Dataset):
    def __init__(self, data_folder, mode="train", data_limit=None, transforms=None):
        self.transforms = transforms
        self.data_folder = data_folder
        self.all_images = sorted(glob(f"{data_folder}/images/*/*.jpg"))
        #self.all_masks = sorted(glob(f"{data_folder}/masks/*/*.npy"))
        self.all_masks = [os.path.join(data_folder, "masks", '/'.join(x.split('/')[-2:])[:-len('.jpg')] + "_mask.npy")
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
        names = sorted(list(set([os.path.basename(os.path.dirname(x)) for x in self.all_images])))
        train_len = int(len(names) * 0.65) + 1
        val_len = int(len(names) * 0.9)
        train_val_dict = {"train": {
                            "images": [filepath for filepath in self.all_images
                                       for name in names[:train_len] if name in filepath],
                            "masks": [filepath for filepath in self.all_masks
                                      for name in names[:train_len] if name in filepath]},
                         "val": {
                            "images": [filepath for filepath in self.all_images
                                        for name in names[train_len:val_len] if name in filepath],
                            "masks": [filepath for filepath in self.all_masks
                                      for name in names[train_len:val_len] if name in filepath]},
                         "test": {
                            "images": [filepath for filepath in self.all_images
                                        for name in names[val_len:] if name in filepath],
                            "masks": [filepath for filepath in self.all_masks
                                      for name in names[val_len:] if name in filepath]}
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
        
class ConcatHandsDataset(Dataset):
    def __init__(self, folders_list, mode='train', data_limit=None, transforms=None,
                 root_cache_path='/cache/datanas1/swetha/'):
        self.folders_list = folders_list
        self.transforms = transforms
        self.component_datasets = [
            EgoHandsDataset(os.path.join(root_cache_path, folder), mode, data_limit)
            for folder in tqdm(folders_list, desc='Creating dataset')
        ]
        self.mode = mode
        self.concat_dataset = torch.utils.data.ConcatDataset(
            self.component_datasets)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        img, mask = self.concat_dataset[idx]
        if self.transforms is not None:
            transformed = self.transforms[self.mode]({"image": img, "target": mask})
            return transformed["image"], transformed["target"]

        return img, mask

def get_dataloaders(args, folders):
    transforms = {
        "train": tsfms.Compose([
            #tsfms.RandomBrightnessJitter(1),
            #tsfms.RandomSaturationJitter(1),
            #tsfms.RandomContrastJitter(1),
            #tsfms.RandomIntensityJitter(0.9, 0.9, 0.9),
            #tsfms.RandomNoise(0.2),
            #tsfms.RandomSizedCrop(512, frac_range=[0.08, 1]),
            tsfms.RandomRotate(30),
            tsfms.RandomHorizontalFlip(),
            tsfms.Resize((512, 512)),
            tsfms.Clip(),
            tsfms.ToTensor(),
        ]),
        "val": tsfms.Compose([
            tsfms.Resize((512, 512)),
            tsfms.Clip(),
            tsfms.ToTensor()
        ])
    }
    train_limit = None
    val_limit = None
    if args.data_limit:
        train_limit = int(args.data_limit * 0.7)
        val_limit = int(args.data_limit * 0.3)
    train_dataset = ConcatHandsDataset(folders, data_limit=train_limit, transforms=transforms)
    val_dataset = ConcatHandsDataset(folders, mode='val', data_limit=val_limit, transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Create your train data loader
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)  # Create your validation data loader
    
    return {"train": train_loader,
            "val": val_loader}
