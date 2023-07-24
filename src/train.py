import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from .metrics import dice_score
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn.functional import threshold, normalize
from statistics import mean


def run_epoch(model, data_loaders, mode, epoch, criterion, optimizer, scheduler=None, device=None, writer=None):
    if mode == "train":
        model.train()
    else:
        model.eval()
    # for param in model.image_encoder.parameters():
    #     param.requires_grad = False
    # model.image_encoder.eval()
    # for param in model.prompt_encoder.parameters():
    #     param.requires_grad = False
    # model.prompt_encoder.eval()
    # if mode != 'train':
    #     for param in model.mask_decoder.parameters():
    #         param.requires_grad = False
    #     model.mask_decoder.eval()
    # else:
    #     model.mask_decoder.train()
    epoch_losses = []
    total_dice = 0
    num_samples = 0
    for input in tqdm(data_loaders[mode]):
        images = input["image"].to(device)
        images = model.preprocess(images)
        targets = input["mask"].to(device)
        print("images", images.size())
        print("targets", targets.size())
        box_torch = input['bbox_coord'].to(device)
        optimizer.zero_grad()
        if box_torch.dtype == torch.int64:
            box_torch = None
        else:
            box_torch = box_torch.to(device)

         
        with torch.no_grad():
            image_embedding = model.image_encoder(images)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None
                    )
        print("sparse", sparse_embeddings.size())
        print("dense", dense_embeddings.size())
        with torch.set_grad_enabled(mode == 'train'):
            low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

            print("low_res_masks", low_res_masks.size())
            print("input_size", input["input_size"])
            print("original_image_size", input['original_image_size'])
            upscaled_masks = model.postprocess_masks(low_res_masks, input['input_size'], input['original_image_size']).to(device)
            background = torch.where(upscaled_masks <= model.mask_threshold, upscaled_masks, 0)
            foreground = torch.where(upscaled_masks > model.mask_threshold, upscaled_masks, 0)
            preds = torch.cat([-1 * background, foreground], dim = 1)
            targets = targets.long()

            print("preds", preds.size())
            print("targets", targets.size())
            loss = criterion(preds, targets)
            dice = dice_score(preds, targets)
            total_dice += dice * len(images)
            num_samples += len(images)
            if mode == 'train':
                loss.backward()
                optimizer.step()
        epoch_losses.append(loss.item())
    average_dice = total_dice/num_samples
    # if mode == 'train':
    #     if scheduler is not None:
    #         scheduler.step()
    # if epoch % 10 == 0:
    #     current_lr = optimizer.param_groups[0]['lr']
    #     optimizer.param_groups[0]['lr'] = current_lr / 2
    current_lr = optimizer.param_groups[0]['lr']
    epoch_loss = mean(epoch_losses)
    writer.add_scalar(f"Loss/{mode}", epoch_loss, epoch)
    if mode == "train":
        writer.add_scalar('lr', current_lr, epoch)
    writer.add_scalar(f"Dice/{mode}", average_dice, epoch)
    print(f'{mode} Epoch: {epoch} - learning rage: {current_lr}')
    print(f'Mean loss: {epoch_loss}')
    print(f'Mean dice score: {average_dice}')

    return average_dice