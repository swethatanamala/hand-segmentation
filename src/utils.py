import torch
import segmentation_models_pytorch as smp

def save_checkpoint(epoch, model, optimizer, filename):
    checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
    torch.save(checkpoint, filename)

def get_model(args):
    model = smp.Unet(
                    encoder_name='resnet50',  # Choose the encoder backbone, e.g., 'resnet18', 'resnet34', 'resnet50'
                    encoder_weights='imagenet',  # Use ImageNet pretraining weights
                    in_channels=3,  # Number of input channels (e.g., 3 for RGB images)
                    classes=2  # Number of output classes (e.g., 2 for binary segmentation)
                )
    if args.resume:
        checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model
        