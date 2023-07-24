import cv2
import random
import numpy as np
import os
import torch
from . import transforms as tsfms
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry


class EgoHandsDataset(Dataset):
    def __init__(self, data_folder, mode="train", data_limit=None, transforms=None, model_type=None):
        self.data_folder = data_folder
        self.all_images = sorted(glob(f"{data_folder}/images/*/*.jpg"))
        random.seed(42)
        random.shuffle(self.all_images)
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
        self.model_type = model_type
        self.bbox_coords = {}
        if self.model_type:
            for mask_path in self.all_masks:
                mask = np.load(mask_path)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                if len(contours) >= 1:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    self.bbox_coords[os.path.basename(mask_path)[:-len('_mask.npy')]] = np.array([x, y, x + w, y + h])
                else:
                    self.bbox_coords[os.path.basename(mask_path)[:-len('_mask.npy')]] = None


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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tsfms.Resize((512, 512))(img)
        print("before cv2", img.shape)
        mask = np.load(mask_path)
        mask = tsfms.Resize((512, 512))(mask)

        if self.model_type:
            print("img", img.shape())
            transform = ResizeLongestSide(1024)
            original_img_size = img.shape[:2]
            img = transform.apply_image(img)
            img_torch = torch.as_tensor(img)
            print("img_torch", img_torch.size())
            transformed_img = img_torch.permute(2, 0, 1).contiguous()
            print("transformed_img", transformed_img.size())
            bbox_coord = self.bbox_coords[os.path.basename(mask_path)[:-len('_mask.npy')]]

            if bbox_coord is None:
                box_torch = 100 #None
            else:
                box = transform.apply_boxes(bbox_coord, original_img_size)
                box_torch = torch.as_tensor(box, dtype=torch.float)
                box_torch = box_torch[None, :]
            return {'image': transformed_img,
                    'mask': mask,
                    'input_size': tuple(transformed_img.shape[-2:]),
                    'original_image_size': original_img_size,
                    'bbox_coord': box_torch,
                    '_id': os.path.basename(mask_path)[:-len('_mask.npy')]}
        
        

class NoHandCookingDataset(Dataset):
    def __init__(self, data_folder, mode="train", data_limit=None, transforms=None, model_type=None):
        self.data_folder = data_folder
        self.all_images = sorted(glob(f"{data_folder}/*/none_with_out_hands/*.jpg"))
        random.seed(42)
        random.shuffle(self.all_images)
        self.train_val_dict = self.get_split()
        self.mode = mode
        if data_limit:
            self.train_val_dict[self.mode]["images"] = self.train_val_dict[self.mode]["images"][:data_limit]
        self.images = self.train_val_dict[self.mode]["images"]
        self.transforms = transforms
        
        
    def get_split(self):
        names = sorted(list(set([os.path.basename(os.path.dirname(os.path.dirname(x))) 
                                 for x in self.all_images])))
        train_len = int(len(names) * 0.65) + 1
        val_len = int(len(names) * 0.9)
        train_val_dict = {"train": 
                            {"images": [filepath for filepath in self.all_images
                                       for name in names[:train_len] if name in filepath]},
                         "val":
                            {"images": [filepath for filepath in self.all_images
                                       for name in names[train_len:val_len] if name in filepath]},
                         "test":
                            {"images": [filepath for filepath in self.all_images
                                       for name in names[val_len:] if name in filepath]}
                        }
        return train_val_dict
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        img = cv2.imread(img_path) / 255
        mask = np.zeros((img.shape[0], img.shape[1]))
       
        if self.transforms:
            transformed = self.transforms[self.mode]({"image": img, "target": mask})
            return transformed["image"], transformed["target"]
        else:
            return img, mask
    

class ConcatHandsDataset(Dataset):
    def __init__(self, folders_list, mode='train', data_limit=None, transforms=None, model_type=None, 
                 root_cache_path='/cache/datanas1/swetha/'):
        self.folders_list = folders_list
        self.transforms = transforms
        self.component_datasets = [
            EgoHandsDataset(os.path.join(root_cache_path, folder), mode, data_limit, transforms=transforms, model_type='sam')
            for folder in tqdm(folders_list, desc='Creating dataset')
        ]
        #self.component_datasets.extend([NoHandCookingDataset("/cache/datanas1/swetha/youcook2/manual",
        #                                                     mode, data_limit)])
        self.mode = mode
        self.model_type = model_type
        self.concat_dataset = torch.utils.data.ConcatDataset(
            self.component_datasets)
        checkpoint_path = "/home/users/swetha/projects/personal/sam-finetuning/checkpoints/sam_vit_h_4b8939.pth"
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        if self.model_type:
            #self.concat_dataset[idx]['image'] = self.sam.preprocess(self.concat_dataset[idx]['image'])
            #print("concatdataset", self.concat_dataset[idx]["image"].size(), self.model_type, self.concat_dataset[idx]['image'].dtype)
            return self.concat_dataset[idx]
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
            #tsfms.RandomRotate(30),
            #tsfms.RandomHorizontalFlip(),
            tsfms.Resize((512, 512)),
            tsfms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": tsfms.Compose([
            tsfms.Resize((512, 512)),
            tsfms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    train_limit = None
    val_limit = None
    transforms = None
    if args.data_limit:
        train_limit = int(args.data_limit * 0.7)
        val_limit = int(args.data_limit * 0.3)
    train_dataset = ConcatHandsDataset(folders, data_limit=train_limit, transforms=transforms, model_type='sam')
    val_dataset = ConcatHandsDataset(folders, mode='val', data_limit=val_limit, transforms=transforms, model_type='sam')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # Create your train data loader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)  # Create your validation data loader
    
    return {"train": train_loader,
            "val": val_loader}
