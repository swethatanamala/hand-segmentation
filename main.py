import argparse
import datetime
import torch
import torch.nn as nn
import os
import torch.optim as optim
from src.dataset import get_dataloaders
from src.utils import save_checkpoint, get_model
from src.train import run_epoch
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
parser.add_argument('--model_name', type=str, default=None,
                    help='Number if data to be limited')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Batch size to be determined')

args = parser.parse_args()

def main(model, data_loaders, num_epochs, criterion, args,  optimizer, scheduler=None, device=None, writer=None):
    best_dice = float('-Inf')
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        torch.cuda.empty_cache()
        train_dice = run_epoch(model, data_loaders, "train", epoch, criterion, optimizer, scheduler=scheduler, device = device, writer=writer)
        val_dice = run_epoch(model, data_loaders, "val", epoch, criterion, optimizer, scheduler=scheduler, device = device, writer=writer)
        os.makedirs(f"checkpoints/{args.exp_name}", exist_ok=True)
        model_save_path = f"checkpoints/{args.exp_name}/checkpoint_overfit.pth"
        best_save_path = f"checkpoints/{args.exp_name}/checkpoint_best_save.pth"
        save_checkpoint(epoch, model, optimizer, model_save_path)
        
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            save_checkpoint(epoch, model, optimizer, best_save_path)
        print(f"Best val score is {str(best_dice)} at epoch {str(best_epoch)}")
        


model = get_model(args)
writer = SummaryWriter(f"logs/{args.exp_name}")
writer.add_text("Experiment Name", args.exp_name)
folders  = ["gtea", "gtea_gaze_plus", "egohands_data", "hands_over_face"]
dataloaders = get_dataloaders(args, folders)
num_epochs = 100  # Specify the number of training epochs
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # Choose your device
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.mask_decoder.parameters(), lr=0.001, momentum=0.9)  # Define your optimizer
#scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

main(model, dataloaders, num_epochs, criterion, args, optimizer, scheduler=None, device=device, writer=writer)
writer.close()
