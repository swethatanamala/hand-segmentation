import torch
import torch.nn as nn
import torch.optim as optim
import os
from .metrics import dice_score
from tqdm import tqdm
from tensorboardX import SummaryWriter


def run_epoch(model, data_loaders, mode, epoch, num_epochs, criterion, optimizer, scheduler, device, writer):
    running_loss = 0
    total_dice = 0
    num_samples = 0
    if mode == "train":
        model.train()
    else:
        model.eval()
    
    for images, targets in tqdm(data_loaders[mode]):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(mode == 'train'):
            outputs = model(images)
            loss = criterion(outputs, targets)
            if mode == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            dice = dice_score(outputs, targets)
            total_dice += dice * len(images)
            num_samples += len(images)
    if mode == 'train':
        scheduler.step()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            print('learning rate at ', epoch + 1, lr)
            writer.add_scalar('lr', lr, epoch + 1)
    epoch_loss = running_loss / len(data_loaders['train'])
    writer.add_scalar(f"Loss/{mode}", epoch_loss, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], {mode} Loss: {epoch_loss:.4f}")
    average_dice = total_dice/num_samples
    writer.add_scalar(f"Dice/{mode}", average_dice, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], {mode} Dice Score: {average_dice:.4f}")
    return average_dice
