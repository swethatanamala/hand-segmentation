import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import get_model
from src import transforms as tsfms
from scipy.ndimage import zoom
from skimage import segmentation


class TestDataset(Dataset):
    def __init__(self, data_folder, transforms=None):
        self.transforms = transforms
        self.data_folder = data_folder
        self.all_images = sorted(glob(f"{data_folder}/*.jpg"))
        self.transforms = transforms
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        img_path = self.all_images[index]
        img = cv2.imread(img_path) / 255
        if self.transforms:
            return self.transforms(img), os.path.basename(img_path)[:-len('.png')]
        else:
            return img, os.path.basename(img_path)[:-len('.png')]

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def HandPredictions(folder_path, args):
    model = get_model(args)
    root_mask = folder_path + "_maskv4"
    os.makedirs(root_mask, exist_ok=True)
    transforms = tsfms.Compose([
                    tsfms.Resize((512, 512)),
                    tsfms.Clip(),
                    tsfms.ToTensor()
                ])
    testdataset = TestDataset(folder_path, transforms=transforms)
    testdataloader = DataLoader(testdataset, batch_size=2, shuffle=False)
    model.eval()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  
    model.to(device)
    img = cv2.imread(testdataset.all_images[0])
    w, h = img.shape[1], img.shape[0]
    for images, names in tqdm(testdataloader):
        print(names)
        images = images.to(device)
        print(images.size())
        with torch.no_grad():
            output = model(images)
        print(output.size())
        predictions = torch.argmax(output, dim=1)
        predictions = predictions.unsqueeze(1).float()
        upsampled_mask = F.interpolate(predictions, size=(h, w), mode='nearest').squeeze(1).cpu().numpy()
        for i in range(upsampled_mask.shape[0]):
            np.save(os.path.join(root_mask, names[i] + "_mask.npy"), upsampled_mask[i])

        


folder_path = "/home/users/swetha/projects/personal/hand-segmentation/data/test/One-Pot_Chicken_Fajita_Pasta"
testdataset = TestDataset(folder_path)
args = {"resume": "checkpoints/resnet50_aug_added_intensity_noise_all_folders_add_negative_cooking/checkpoint_best_save.pth"}
args = AttributeDict(args)
model = get_model(args)
HandPredictions(folder_path, args)


