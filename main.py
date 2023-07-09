import argparse
import datetime
import torch
import torch.nn as nn
import os
import torch.optim as optim
from src.dataset import EgoHandsDataset
from src.utils import save_checkpoint, get_model
import src.transforms as tsfms
from src.train import run_epoch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

date = str(datetime.datetime.now().date())
parser = argparse.ArgumentParser(description='HandSegmentation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint to be resumed')
parser.add_argument('--exp_name', type=str, default='exp_' + date,
                    help='Experiment prefix for checkpoint')
parser.add_argument('--data_limit', type=int, default=None,
                    help='Number if data to be limited')

args = parser.parse_args()

def main(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device, args, writer):
    data_loaders = {"train": train_loader,
                   "val": val_loader}
    best_dice = float('-Inf')
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        model.to(device)
        train_dice = run_epoch(model, data_loaders, "train", epoch, num_epochs, criterion, optimizer, scheduler, device, writer)
        val_dice = run_epoch(model, data_loaders, "val", epoch, num_epochs, criterion, optimizer, scheduler, device, writer)
        os.makedirs(f"checkpoints/{args.exp_name}", exist_ok=True)
        model_save_path = f"checkpoints/{args.exp_name}/checkpoint_overfit.pth"
        best_save_path = f"checkpoints/{args.exp_name}/checkpoint_best_save.pth"
        save_checkpoint(epoch, model, optimizer, model_save_path)
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            save_checkpoint(epoch, model, optimizer, best_save_path)
        print(f"Best val score is {str(best_dice)} at epoch {str(best_epoch)}")


transforms = {
    "train": tsfms.Compose([
        #tsfms.RandomBrightnessJitter(1),
        #tsfms.RandomSaturationJitter(1),
        #tsfms.RandomContrastJitter(1),
        tsfms.RandomIntensityJitter(0.9, 0.9, 0.9),
        tsfms.RandomNoise(0.2),
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

model = get_model(args)
train_limit = None
val_limit = None
if args.data_limit:
    train_limit = int(args.data_limit * 0.7)
    val_limit = int(args.data_limit * 0.3)
writer = SummaryWriter(f"logs/{args.exp_name}")
writer.add_text("Experiment Name", args.exp_name)
data_folder = "/cache/datanas1/swetha/egohands_data"
train_dataset = EgoHandsDataset(data_folder, data_limit=train_limit, transforms=transforms)
val_dataset = EgoHandsDataset(data_folder, mode='val', data_limit=val_limit, transforms=transforms)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Create your train data loader
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)  # Create your validation data loader
num_epochs = 100  # Specify the number of training epochs
criterion = nn.CrossEntropyLoss()  # Define your loss function
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # Define your optimizer
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose your device
main(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device, args, writer)
writer.close()