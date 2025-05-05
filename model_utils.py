import torch
import os
import torch.nn as nn
from torchvision import models

def get_model(model_name, unfreeze_layers=None, num_classes=29, use_pretrained=True):
    if unfreeze_layers is not None and unfreeze_layers not in ['C0', 'C1', 'C2']:
        raise ValueError("Invalid layer to unfreeze. Choose from 'C0', 'C1', or 'C2'.")

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.resnet50(weights=weights)
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze and replace the final layer
        if unfreeze_layers is not None:
            num_filters = model.fc.in_features
            model.fc = nn.Linear(num_filters, num_classes)

        if unfreeze_layers == 'C1':
            for param in model.layer4.parameters():
                param.requires_grad = True

        elif unfreeze_layers == 'C2':
            for param in model.layer4.parameters():
                param.requires_grad = True

            for param in model.layer3.parameters():
                param.requires_grad = True

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze and replace the final layer
        if unfreeze_layers is not None:
            num_filters = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_filters, num_classes)

        if unfreeze_layers == 'C1':
            for param in model.features[8].parameters():
                param.requires_grad = True

        elif unfreeze_layers == 'C2':
            for param in model.features[8].parameters():
                param.requires_grad = True

            for param in model.features[7].parameters():
                param.requires_grad = True

    else:
        raise ValueError("Invalid model name. Choose from 'resnet50' or 'efficientnet_b0'.")
    return model

def load_model_checkpoint(checkpoint_path: str,
                          model: nn.Module,
                          optimizer = None,
                          scheduler = None,
                          device = None
                         ):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path} ...")
    try:
        # Load checkpoint onto the specified device directly if possible
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            # Ensure model is on the correct device after loading state dict
            if device:
                 model.to(device)
            print(f"-> Model state loaded successfully.")
        else:
            print("Warning: 'model_state_dict' not found in checkpoint.")

        # Load optimizer state (optional)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer states to the correct device (important if loading CPU->GPU or vice-versa)
            if device:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            print("-> Optimizer state loaded successfully.")
        elif optimizer is not None:
            print("Warning: Optimizer provided but 'optimizer_state_dict' not found in checkpoint.")

        # Load scheduler state (optional)
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("-> Scheduler state loaded successfully.")
        elif scheduler is not None:
            print("Warning: Scheduler provided but 'scheduler_state_dict' not found in checkpoint.")

        # Load epoch and best accuracy
        # Checkpoint saves the epoch *completed*, so we start from the next one
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"-> Resuming from epoch {start_epoch} (epoch {checkpoint.get('epoch', -1)} completed).")
        print(f"-> Best validation accuracy recorded: {best_val_acc:.4f}")

        print("Checkpoint loaded successfully.")
        return model, optimizer, scheduler, start_epoch, best_val_acc

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Depending on desired behavior, you might re-raise, or return defaults
        # Returning defaults here to allow the caller to potentially start fresh
        # Ensure model is still on the correct device even if load failed partially
        if device:
            model.to(device)
        return model, optimizer, scheduler, 0, 0.0
