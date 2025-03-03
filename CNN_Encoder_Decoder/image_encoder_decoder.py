"""
Image Encoder-Decoder Module for AI-Generated Image Detection.

This module implements a convolutional neural network (CNN) model with a 
UNet-style decoder for AI-generated image classification and heatmap visualization 
using Grad-CAM. It includes functions for training, evaluation, hyperparameter 
tuning, and Grad-CAM visualization.

Main Features:
---------------
- **CustomDataset**: A PyTorch dataset class for loading images and labels from a CSV file.
- **CNNModel**: A convolutional neural network with Grad-CAM support and a UNet-style decoder.
- **UNetDecoder**: A decoder that progressively upsamples feature maps and incorporates skip connections.
- **FocalLoss**: A custom loss function to handle class imbalance.
- **Grad-CAM Visualization**: Functions to compute and overlay Grad-CAM heatmaps.
- **Hyperparameter Tuning**: Uses Ray Tune for automatic model tuning.
- **Early Stopping & Checkpointing**: Implements training checkpoints and early stopping.
- **Evaluation Metrics**: Computes accuracy, precision, recall, F1-score, IoU, and Dice score.

Dependencies:
--------------
This module requires the following libraries:
- `torch`, `torchvision`
- `numpy`, `pandas`
- `opencv-python` (`cv2`)
- `ray[tune]`
- `scikit-learn` (`sklearn.metrics`)
- `PIL` (Pillow)
- `matplotlib`

For usage examples and setup instructions, refer to the **README** file.

Author:
--------
Pierce Ohlmeyer-Dawson
"""

# Standard Library
import os
import sys
import random

# Third-Party Libraries
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# PyTorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR

# Torchvision
from torchvision.transforms import transforms

# Ray Tune (Hyperparameter Optimization)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

# Scikit-Learn (Evaluation Metrics)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#If DEBUG_MODE is TRUE, it will skip hyperparameter tuning
DEBUG_MODE = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((32, 32)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((32, 32)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class CustomDataset(Dataset):
    """Custom dataset for loading images and labels from a CSV file.

    This dataset reads image paths and their corresponding labels from a CSV file,
    loads the images from disk, and applies optional transformations.

    Args:
        annotations_file (str): Path to the CSV file containing image paths and labels.
        img_dir (str): Directory where images are stored.
        image_transform (callable, optional): A function/transform to apply to the images.

    Attributes:
        img_labels (pd.DataFrame): DataFrame containing image filenames and labels.
        img_dir (str): Path to the directory containing images.
        transform (callable, optional): Transform function for preprocessing images.
    """

    def __init__(self, annotations_file, img_dir, image_transform=None):
        """Initialize the dataset by loading labels from a CSV file."""
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = image_transform

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """Retrieve an image and its corresponding label by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (PIL.Image.Image): The raw image (if no transform is applied).
                - image (torch.Tensor): The transformed image (if a transform is provided).
                - label (float): The corresponding label for the image.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = float(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label



class FocalLoss(nn.Module):
    """Implementation of the Focal Loss function.

    Focal Loss is designed to address class imbalance by down-weighting easy examples 
    and focusing more on hard examples. It modifies the standard binary cross-entropy loss 
    by adding a modulating factor `(1 - p_t)^gamma` to emphasize misclassified examples.

    Args:
        alpha (float, optional): Scaling factor for positive class (default: 0.25).
        gamma (float, optional): Modulating factor for hard examples (default: 2.0).
        reduction (str, optional): Specifies the reduction method. 
            Options: "mean" (default), "sum", or "none".
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """Initialize the Focal Loss function with given parameters."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute the focal loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (N, *).
            targets (torch.Tensor): Ground truth labels of shape (N, *), same shape as `inputs`.

        Returns:
            torch.Tensor: The computed focal loss. If `reduction` is:
                - `"mean"`: Returns the mean loss.
                - `"sum"`: Returns the summed loss.
                - `"none"`: Returns the unaggregated loss tensor.
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # p_t is the predicted probability
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_gradcam(model, image_tensor):
    """Compute Grad-CAM activations for a given image tensor.

    Grad-CAM (Gradient-weighted Class Activation Mapping) generates a heatmap
    highlighting the most important regions of an image for a given model prediction.

    Args:
        model (torch.nn.Module): The neural network model.
        image_tensor (torch.Tensor): The input image tensor of shape (C, H, W)
            (or (N, C, H, W) if batched), where:
            - C = number of channels (e.g., 3 for RGB)
            - H = image height
            - W = image width

    Returns:
        torch.Tensor: The activation map after the second convolutional layer,
        used for Grad-CAM visualization. Shape: (N, C', H', W'), where C' is the 
        number of output channels in `model.conv2`.
    """
    model.eval()

    x = model.conv1(image_tensor)
    x = model.silu1(x)
    x = model.pool1(x)

    x = model.conv2(x)
    x = model.bn2(x)
    x = model.silu2(x)
    x = model.pool2(x)

    activations = x

    return activations


class UNetDecoder(nn.Module):
    """UNet-style decoder with bilinear upsampling and spatial attention.

    This decoder is part of a U-Net-like architecture, used to progressively 
    upsample feature maps while incorporating skip connections from the encoder.
    It applies bilinear upsampling, convolutional layers, and activation functions 
    to generate a refined output.

    Args:
        num_filters (int): The base number of convolutional filters.

    Attributes:
        upsample1 (nn.Upsample): First bilinear upsampling layer (scale factor 2).
        upsample2 (nn.Upsample): Second bilinear upsampling layer (scale factor 2).
        conv_skip (nn.Conv2d): 1x1 convolution to process the skip connection.
        conv1 (nn.Conv2d): Convolutional layer to combine encoder and skip features.
        conv2 (nn.Conv2d): Final convolutional layer producing a single-channel output.
    """

    def __init__(self, num_filters):
        """Initialize the UNetDecoder with specified filter size."""
        super().__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_skip = nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=1)
        self.conv1 = nn.Conv2d(num_filters * 4, num_filters * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters * 2, 1, kernel_size=3, padding=1)

    def forward(self, x, skip_connection):
        """Perform forward pass through the decoder.

        Args:
            x (torch.Tensor): The input feature map from the encoder.
            skip_connection (torch.Tensor): The corresponding skip connection feature map.

        Returns:
            torch.Tensor: The upsampled and processed feature map with shape 
            (N, 1, H_out, W_out), where H_out and W_out are the final spatial dimensions.
        """
        if x.shape[2:] != skip_connection.shape[2:]:
            skip_connection = F.interpolate(
                skip_connection,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=True
            )
        x = torch.cat([x, skip_connection], dim=1)
        x = F.silu(self.conv1(x))
        x = self.upsample2(x)
        x = torch.sigmoid(self.conv2(x))
        return x


class CNNModel(nn.Module):
    """CNN model with Grad-CAM support, UNet-style decoder, and spatial attention.

    This convolutional neural network is designed for classification tasks and 
    includes Grad-CAM support for visualizing important regions. It uses a 
    U-Net-style decoder to generate pixel-wise heatmaps.

    Args:
        num_filters (int, optional): Number of filters in the first convolutional layer. 
            The number of filters doubles in deeper layers. Default is 32.
        dropout (float, optional): Dropout probability for the classifier. Default is 0.5.

    Attributes:
        gradients (torch.Tensor or None): Stores gradients for Grad-CAM.
        conv1 (nn.Conv2d): First convolutional layer.
        silu1 (nn.SiLU): Activation function for `conv1`.
        pool1 (nn.MaxPool2d): Max pooling layer for downsampling.
        conv2 (nn.Conv2d): Second convolutional layer with increased filters.
        bn2 (nn.BatchNorm2d): Batch normalization layer after `conv2`.
        silu2 (nn.SiLU): Activation function for `conv2`.
        pool2 (nn.MaxPool2d): Second max pooling layer.
        global_pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        classifier (nn.Sequential): Fully connected classifier.
        decoder (UNetDecoder): U-Net-style decoder for generating heatmaps.
    """

    def __init__(self, num_filters=32, dropout=0.5):
        """Initialize the CNN model with convolutional layers, Grad-CAM support, 
        a classification head, and a UNet-style decoder."""
        super().__init__()
        self.gradients = None

        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.silu1 = nn.SiLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling

        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        self.silu2 = nn.SiLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Further downsampling

        last_conv_layer = self.conv2
        last_conv_layer.register_full_backward_hook(self._save_gradient_hook)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 2, 1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
        )

        self.decoder = UNetDecoder(num_filters)

    def _save_gradient_hook(self, _, __, grad_output):
        """Hook to store gradients for Grad-CAM.

        This method captures gradients during backpropagation for Grad-CAM visualization.

        Args:
            _ (Unused): Placeholder argument for PyTorch's hook interface.
            __ (Unused): Placeholder argument for PyTorch's hook interface.
            grad_output (tuple of torch.Tensor): Gradients from the backward pass.
        """
        if self.gradients is None:
            self.gradients = grad_output[0].detach().clone()
        else:
            self.gradients += grad_output[0].detach().clone()

    def get_gradients(self):
        """Retrieve and reset stored gradients for Grad-CAM.

        Returns:
            torch.Tensor or None: The stored gradients if available, otherwise None.
        """
        gradients = self.gradients.clone() if self.gradients is not None else None
        self.gradients = None
        return gradients

    def forward(self, x, train_decoder=True, train_encoder=True):
        """Perform a forward pass through the CNN model.

        This method processes the input through convolutional layers, extracts 
        global features for classification, and optionally generates a Grad-CAM 
        heatmap using a UNet-style decoder.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W).
            train_decoder (bool, optional): Whether to compute the heatmap using the decoder. 
                Default is True.
            train_encoder (bool, optional): Whether to compute gradients for the encoder. 
                Default is True.

        Returns:
            tuple:
                - class_output (torch.Tensor): The classification output of shape (N, 1).
                - heatmap (torch.Tensor): The Grad-CAM heatmap of shape (N, 1, H_out, W_out).
                  If `train_decoder` is False, a tensor of zeros is returned instead.
        """
        with torch.no_grad() if not train_encoder else torch.enable_grad():
            x = self.conv1(x)
            x = self.silu1(x)
            x = self.pool1(x)

            skip_connection = x

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.silu2(x)
            x = self.pool2(x)

        global_features = self.global_pool(x).flatten(start_dim=1)
        class_output = self.classifier(global_features)

        # Ensure skip connection has the correct number of channels
        if skip_connection.shape[1] != x.shape[1]:
            if skip_connection.shape[1] > x.shape[1]:
                skip_connection = skip_connection[:, :x.shape[1], :, :].contiguous()
            else:
                pad_shape = (
                    skip_connection.shape[0],
                    x.shape[1] - skip_connection.shape[1],
                    *skip_connection.shape[2:]
                )
                padding = torch.zeros(pad_shape, device=skip_connection.device)
                skip_connection = torch.cat([skip_connection, padding], dim=1)

        # Generate heatmap using decoder or return zeros if disabled
        if not train_decoder:
            heatmap = torch.zeros_like(x[:, :1, :, :])
        else:
            heatmap = self.decoder(x, skip_connection)

        return class_output, heatmap


def total_variation_loss(heatmap):
    """Compute the total variation (TV) loss for image smoothness.

    Total Variation (TV) loss is used to encourage spatial smoothness in images 
    by penalizing local variations. It helps in reducing noise while preserving edges.

    Args:
        heatmap (torch.Tensor): The input tensor of shape (N, C, H, W), where:
            - N: Batch size
            - C: Number of channels
            - H: Height of the feature map
            - W: Width of the feature map

    Returns:
        torch.Tensor: A scalar tensor representing the total variation loss.
    """
    return (
        torch.mean(torch.abs(heatmap[:, :, :, :-1] - heatmap[:, :, :, 1:])) +
        torch.mean(torch.abs(heatmap[:, :, :-1, :] - heatmap[:, :, 1:, :]))
    )

def adjust_trainable_layers(model, epoch, total_epochs):
    """Adjust which parts of the model are trainable based on the training stage.

    This function controls which layers of the model are updated during training. 
    It follows a staged training approach:
    
    - In the first half of training, the **encoder** is trained while the **decoder** is frozen.
    - In the second half of training, the **decoder** is trained while the **encoder** is frozen.

    Args:
        model (torch.nn.Module): The model whose trainable layers are adjusted.
        epoch (int): The current epoch number.
        total_epochs (int): The total number of training epochs.
    """
    if epoch < total_epochs // 2:
        print("Training Encoder (Decoder Frozen)")

        for param in [
            model.conv1, model.silu1, model.pool1,
            model.conv2, model.bn2, model.silu2, model.pool2
        ]:
            param.requires_grad_(True)

        for param in model.decoder.parameters():
            param.requires_grad_(False)

    else:
        print("Training Decoder (Encoder Frozen)")

        for param in [
            model.conv1, model.silu1, model.pool1,
            model.conv2, model.bn2, model.silu2, model.pool2
        ]:
            param.requires_grad_(False)

        for param in model.decoder.parameters():
            param.requires_grad_(True)


def compute_losses(class_output, heatmap, labels, loss_params):
    """Compute classification, heatmap, and total variation losses.

    This function calculates the combined loss for a model that outputs both 
    classification predictions and pixel-wise heatmaps. It includes:
    
    - **Classification Loss:** Computes the primary classification loss using 
      the criterion provided in `loss_params`.
    - **Heatmap Loss:** Applies focal loss, IoU loss, and total variation loss 
      to encourage accurate and smooth heatmaps.
    - **Total Loss:** The final loss is a weighted sum of classification and heatmap losses.

    Args:
        class_output (torch.Tensor): The classification output tensor of shape (N, 1).
        heatmap (torch.Tensor): The predicted heatmap tensor of shape (N, 1, H, W).
        labels (torch.Tensor): Ground truth labels of shape (N,).
        loss_params (dict): Dictionary containing:
            - `"criterion"` (callable): Loss function for classification.
    
    Returns:
        torch.Tensor: The total computed loss, which is a weighted sum of the 
        classification loss and heatmap loss.
    """
    class_loss = loss_params["criterion"](class_output, labels)
    pixel_labels = labels.view(-1, 1, 1, 1).expand_as(heatmap)
    heatmap_loss = focal_iou_tv_loss(heatmap, pixel_labels)
    return class_loss + 0.5 * heatmap_loss


def train_model(model_instance, train_loader, val_loader, config):
    """Train the CNN model with staged training and early stopping.

    This function trains a convolutional neural network using a staged training approach:
    
    - The **encoder** is trained in the first half of the epochs, while the **decoder** is frozen.
    - The **decoder** is trained in the second half of the epochs, while the **encoder** is frozen.
    
    It also implements **early stopping** to prevent overfitting. The model is evaluated 
    using both classification accuracy and heatmap metrics (IoU and Dice score).

    Args:
        model_instance (nn.Module): The CNN model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        config (dict): Configuration dictionary containing:
            - `"num_epochs"` (int): Total number of training epochs.
            - `"tv_loss_weight"` (float, optional): Weight for total variation loss. Defaults to 1.0.

    Returns:
        tuple:
            - best_model_state_dict (dict): The state dictionary of the best-performing model.
            - best_val_acc (float): The highest validation accuracy achieved during training.
    """
    optimizer, train_scheduler, criterion, focal_criterion, scaler = (
        setup_training_tools(model_instance, config)
    )

    loss_params = {
        "criterion": criterion,
        "focal_criterion": focal_criterion,
        "tv_loss_weight": config.get("tv_loss_weight", 1.0)
    }

    state = {
        "best_val_acc": 0,
        "best_model": None,
        "patience": 5,
        "no_improve_epochs": 0
    }
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(config["num_epochs"]):
        config["epoch"] = epoch
        model_instance.train()
        adjust_trainable_layers(model_instance, epoch, config["num_epochs"])

        train_metrics = train_one_epoch(
            model_instance, train_loader,
            {"optimizer": optimizer, "scheduler": train_scheduler, "scaler": scaler},
            loss_params,
            config
        )

        val_acc, heatmap_metrics = evaluate_model(model_instance, val_loader, evaluate_heatmap=True)
        if heatmap_metrics is not None:
            iou, dice = heatmap_metrics
        else:
            iou, dice = None, None

        train_scheduler.step()

        print(f"Epoch [{epoch + 1}/{config['num_epochs']}] - "
              f"Train Loss: {train_metrics['loss']:.4f} "
              f"- Train Acc: {train_metrics['accuracy']:.2f}% - "
              f"Val Acc: {val_acc:.2f}% - IoU: {iou if iou is not None else 'N/A'} "
              f"- Dice: {dice if dice is not None else 'N/A'}")

        if check_early_stopping(
            state,
            model_instance,
            val_acc,
            (iou + dice) / 2 if iou is not None and dice is not None else iou or dice,
            epoch
        ):
            break

    return state["best_model"], state["best_val_acc"]


def check_early_stopping(state, model, val_acc, heatmap_acc, epoch):
    """Check for early stopping and save the best model checkpoint.

    This function monitors the validation accuracy (`val_acc`) and heatmap accuracy 
    (`heatmap_acc`) to determine whether training should stop early. If an improvement 
    is detected, it updates the best model state and resets the no-improvement counter.
    If no improvement is observed for a number of consecutive epochs equal to the patience 
    threshold, early stopping is triggered.

    Args:
        state (dict): A dictionary containing:
            - `"best_val_acc"` (float): The highest recorded validation accuracy.
            - `"best_model"` (dict): The state dictionary of the best model.
            - `"patience"` (int): The number of epochs allowed without improvement.
            - `"no_improve_epochs"` (int): Counter for epochs without improvement.
        model (torch.nn.Module): The model being trained.
        val_acc (float): The current validation accuracy.
        heatmap_acc (float): The current heatmap accuracy (IoU/Dice score).
        epoch (int): The current training epoch.

    Returns:
        bool: `True` if early stopping is triggered, otherwise `False`.
    """
    if val_acc > state["best_val_acc"] or heatmap_acc > state["best_val_acc"]:
        state.update({"best_val_acc": val_acc, "best_model": model.state_dict()})
        state["no_improve_epochs"] = 0
        torch.save(state["best_model"], f"checkpoints/best_model_epoch_{epoch + 1}.pth")
        return False

    state["no_improve_epochs"] += 1

    if state["no_improve_epochs"] == state["patience"]:
        print("Early stopping triggered.")
        return True

    return False


def focal_iou_tv_loss(preds, targets, alpha=0.25, gamma=2.0, tv_weight=0.01):
    """Compute the combined Focal, IoU, and Total Variation (TV) loss.

    This loss function integrates three components:
    
    - **Focal Loss:** Mitigates class imbalance by focusing more on hard-to-classify examples.
    - **IoU Loss:** Measures the intersection-over-union (IoU) between predicted and target masks.
    - **Total Variation (TV) Loss:** Encourages spatial smoothness in predictions.

    Args:
        preds (torch.Tensor): Predicted logits of shape (N, C, H, W).
        targets (torch.Tensor): Ground truth binary masks of shape (N, C, H, W).
        alpha (float, optional): Focal loss weighting factor for positive samples. Default is 0.25.
        gamma (float, optional): Focal loss focusing parameter. Default is 2.0.
        tv_weight (float, optional): Weight assigned to the total variation loss. Default is 0.01.

    Returns:
        torch.Tensor: The computed loss value (scalar).
    """
    bce = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = 1 - (intersection + 1e-6) / (union + 1e-6)

    tv = torch.mean(torch.abs(preds[:, :, :, :-1] - preds[:, :, :, 1:])) + \
         torch.mean(torch.abs(preds[:, :, :-1, :] - preds[:, :, 1:, :]))

    return focal.mean() + iou + tv_weight * tv


def setup_training_tools(model, config):
    """Set up the optimizer, learning rate scheduler, loss functions, and gradient scaler.

    This function initializes the training tools required for model optimization, including:
    
    - **Optimizer:** Adam optimizer with weight decay.
    - **Learning Rate Scheduler:** StepLR scheduler that decays the learning rate every 5 epochs.
    - **Loss Functions:** BCEWithLogitsLoss for classification and Focal IoU TV loss for heatmaps.
    - **Gradient Scaler:** Mixed-precision training scaler for efficiency.

    Args:
        model (torch.nn.Module): The neural network model being trained.
        config (dict): Dictionary containing training hyperparameters, including:
            - `"learning_rate"` (float): The initial learning rate.
            - `"weight_decay"` (float): Weight decay for regularization.

    Returns:
        tuple: A tuple containing:
            - optimizer (torch.optim.Optimizer): The Adam optimizer.
            - train_scheduler (torch.optim.lr_scheduler.StepLR): The learning rate scheduler.
            - criterion (torch.nn.Module): The binary cross-entropy loss function.
            - focal_criterion (callable): The focal IoU TV loss function.
            - scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
    """
    optimizer = optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    train_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()
    focal_criterion = focal_iou_tv_loss

    scaler = torch.amp.GradScaler(enabled=True)
    return optimizer, train_scheduler, criterion, focal_criterion, scaler


def train_one_epoch(model, dataloader, optim_params, loss_params, config):
    """Train the model for one epoch and return training metrics.

    This function performs one full pass over the training dataset, optimizing 
    the model using mixed-precision training. It calculates the total loss 
    (classification + heatmap loss) and tracks training accuracy.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optim_params (dict): Dictionary containing:
            - `"optimizer"` (torch.optim.Optimizer): The optimizer.
            - `"scaler"` (torch.cuda.amp.GradScaler): Scaler for mixed precision training.
        loss_params (dict): Dictionary containing:
            - `"criterion"` (callable): Classification loss function.
            - `"focal_criterion"` (callable): Heatmap loss function.
        config (dict): Dictionary containing training configurations, including:
            - `"train_decoder"` (bool, optional): Whether to train the decoder. Defaults to `True`.
            - `"train_encoder"` (bool, optional): Whether to train the encoder. Defaults to `True`.

    Returns:
        dict: A dictionary containing:
            - `"loss"` (float): The average training loss for the epoch.
            - `"accuracy"` (float): The training accuracy percentage.
    """
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    optimizer, scaler = optim_params["optimizer"], optim_params["scaler"]

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
            class_output, heatmap = model(
                images,
                train_decoder=config.get("train_decoder", True),
                train_encoder=config.get("train_encoder", True)
            )
            loss = compute_losses(class_output, heatmap, labels, loss_params)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        if not torch.isnan(loss) and not torch.isinf(loss):
            running_loss += loss.item()

        correct_train += float((torch.sigmoid(class_output).round() == labels).sum().item())
        total_train += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = (correct_train / total_train) * 100

    return {"loss": avg_loss, "accuracy": accuracy}


def intersection_over_union(preds, labels):
    """Compute the Intersection over Union (IoU) score.

    IoU measures the overlap between the predicted and ground truth masks. It is 
    defined as the ratio of the intersection to the union of the two sets.

    Args:
        preds (torch.Tensor): The predicted binary mask of shape (N, C, H, W).
        labels (torch.Tensor): The ground truth binary mask of shape (N, C, H, W).

    Returns:
        torch.Tensor: The IoU score as a scalar tensor, computed as:
            IoU = (intersection + 1e-6) / (union + 1e-6),
        where a small epsilon (1e-6) is added to prevent division by zero.
    """
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def dice_score(preds, labels):
    """Compute the Dice coefficient score.

    The Dice coefficient is a measure of overlap between two sets, commonly 
    used for evaluating image segmentation models. It is defined as:

        Dice = (2 * |A ∩ B|) / (|A| + |B|)

    where A is the predicted mask and B is the ground truth mask.

    Args:
        preds (torch.Tensor): The predicted binary mask of shape (N, C, H, W).
        labels (torch.Tensor): The ground truth binary mask of shape (N, C, H, W).

    Returns:
        torch.Tensor: The Dice coefficient score as a scalar tensor, computed as:
            Dice = (2 * intersection + 1e-6) / (preds.sum() + labels.sum() + 1e-6),
        where a small epsilon (1e-6) is added to prevent division by zero.
    """
    intersection = (preds * labels).sum()
    return (2 * intersection + 1e-6) / (preds.sum() + labels.sum() + 1e-6)


def evaluate_model(model, dataloader, evaluate_heatmap=False):
    """Evaluate the model for classification accuracy and optional heatmap metrics.

    This function computes the **classification accuracy** for the given dataset. 
    If `evaluate_heatmap` is enabled, it also calculates the **Intersection over Union (IoU)** 
    and **Dice score** for the predicted heatmaps.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        evaluate_heatmap (bool, optional): If `True`, computes IoU and Dice score for heatmaps. 
            Defaults to `False`.

    Returns:
        tuple:
            - image_accuracy (float): The classification accuracy in percentage.
            - heatmap_metrics (tuple or None): A tuple `(iou, dice)` if `evaluate_heatmap=True`, 
              otherwise `None`.
    """
    model.eval()
    correct, total = 0, 0
    all_pixel_preds, all_pixel_labels = ([], []) if evaluate_heatmap else (None, None)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            class_output, heatmap = model(images, train_decoder=evaluate_heatmap)

            correct += (torch.sigmoid(class_output).round() == labels).sum().item()
            total += labels.size(0)

            if evaluate_heatmap:
                pixel_labels = labels.view(-1, 1, 1, 1).expand_as(heatmap)
                all_pixel_preds.extend(torch.sigmoid(heatmap).round().cpu().numpy().flatten())
                all_pixel_labels.extend(pixel_labels.cpu().numpy().flatten())

    image_accuracy = 100 * correct / total

    if evaluate_heatmap:
        iou = intersection_over_union(np.array(all_pixel_preds), np.array(all_pixel_labels))
        dice = dice_score(np.array(all_pixel_preds), np.array(all_pixel_labels))

        print(f"Image Acc: {image_accuracy:.2f}% | IoU: {iou:.4f} | Dice: {dice:.4f}")
        return image_accuracy, (iou, dice)

    return image_accuracy, None


def train_hyperparam_tuning(config, train_loader=None, val_loader=None):
    """Train the model using hyperparameter tuning and report validation accuracy.

    This function trains a CNN model with given hyperparameters, optimizing for 
    classification accuracy. It uses mixed-precision training, evaluates the model 
    on a validation set, and tracks the best-performing model.

    Args:
        config (dict): Dictionary containing hyperparameters, including:
            - `"num_filters"` (int): Number of filters in the first convolutional layer.
            - `"dropout"` (float): Dropout rate for the classifier.
            - `"learning_rate"` (float): Learning rate for the optimizer.
            - `"weight_decay"` (float): Weight decay (L2 regularization) for the optimizer.
            - `"num_epochs"` (int): Total number of training epochs.
        train_loader (torch.utils.data.DataLoader, optional): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset.

    Returns:
        None: The function does not return a value but reports validation accuracy 
        using `session.report()`.
    """
    model_instance = CNNModel(num_filters=config["num_filters"],
                              dropout=config["dropout"]).to(DEVICE)

    if config["optimizer"] == "adamw":
        optimizer = optim.AdamW(model_instance.parameters(), 
                                lr=config["learning_rate"], 
                                weight_decay=config["weight_decay"])
    elif config["optimizer"] == "adam":
        optimizer = optim.Adam(model_instance.parameters(), 
                            lr=config["learning_rate"], 
                            weight_decay=config["weight_decay"])
    elif config["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(model_instance.parameters(), 
                                lr=config["learning_rate"], 
                                weight_decay=config["weight_decay"])
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(model_instance.parameters(), 
                            lr=config["learning_rate"], 
                            momentum=0.9, 
                            weight_decay=config["weight_decay"])
    elif config["optimizer"] == "nadam":
        optimizer = optim.NAdam(model_instance.parameters(), 
                                lr=config["learning_rate"], 
                                weight_decay=config["weight_decay"])
    elif config["optimizer"] == "adagrad":
        optimizer = optim.Adagrad(model_instance.parameters(), 
                                lr=config["learning_rate"], 
                                weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    criterion = nn.BCEWithLogitsLoss()
    focal_criterion = focal_iou_tv_loss
    scaler = torch.amp.GradScaler(enabled=True)

    best_val_acc = 0.0

    for epoch in range(config["num_epochs"]):
        model_instance.train()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                class_output, heatmap = model_instance(images, train_decoder=False)
                loss = criterion(class_output, labels) + 0.5 * focal_criterion(
                    heatmap, labels.view(-1, 1, 1, 1).expand_as(heatmap))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model_instance.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                class_output, _ = model_instance(images, train_decoder=False)
                correct += (torch.sigmoid(class_output).round() == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_dir = session.get_checkpoint()
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(model_instance.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        session.report({"val_acc": val_acc, "train_loss": loss.item(), "epoch": epoch})


def evaluate_model_with_metrics(model, dataloader, evaluate_heatmap=False):
    """Evaluate the model on a dataset, computing classification and heatmap metrics.

    This function evaluates a trained model on a given dataset, computing:
    
    - **Classification Metrics:** Accuracy, precision, recall, and F1-score.
    - **Heatmap Metrics (Optional):** IoU, Dice score, pixel-wise accuracy, precision, recall, and F1-score.

    If `evaluate_heatmap=True`, the function also evaluates pixel-wise predictions 
    for segmentation tasks.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        evaluate_heatmap (bool, optional): If `True`, computes segmentation metrics. 
            Defaults to `False`.

    Returns:
        dict: A dictionary containing classification metrics:
            - `"accuracy"` (float): Image classification accuracy.
            - `"precision"` (float): Precision score.
            - `"recall"` (float): Recall score.
            - `"f1"` (float): F1 score.

        If `evaluate_heatmap=True`, returns an additional dictionary with pixel-wise segmentation metrics:
            - `"accuracy"` (float): Pixel-wise accuracy.
            - `"precision"` (float): Pixel-wise precision.
            - `"recall"` (float): Pixel-wise recall.
            - `"f1"` (float): Pixel-wise F1 score.
            - `"iou"` (float): Intersection over Union (IoU) score.
            - `"dice"` (float): Dice coefficient.
    """
    model.eval()
    all_preds, all_labels = [], []
    all_pixel_preds = all_pixel_labels = None if not evaluate_heatmap else ([], [])

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            class_output, heatmap = model(images, train_decoder=evaluate_heatmap)

            all_preds.extend(torch.sigmoid(class_output).round().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if evaluate_heatmap:
                all_pixel_preds.extend(torch.sigmoid(heatmap).round().cpu().numpy().flatten())
                all_pixel_labels.extend(labels.expand_as(heatmap).cpu().numpy().flatten())

    image_metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=1),
        "recall": recall_score(all_labels, all_preds, zero_division=1),
        "f1": f1_score(all_labels, all_preds, zero_division=1)
    }

    print(f"Test Accuracy: {image_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Precision: {image_metrics['precision']:.4f}")
    print(f"Test Recall: {image_metrics['recall']:.4f}")
    print(f"Test F1 Score: {image_metrics['f1']:.4f}")

    if evaluate_heatmap:
        pixel_metrics = {
            "accuracy": accuracy_score(all_pixel_labels, all_pixel_preds),
            "precision": precision_score(all_pixel_labels, all_pixel_preds, zero_division=1),
            "recall": recall_score(all_pixel_labels, all_pixel_preds, zero_division=1),
            "f1": f1_score(all_pixel_labels, all_pixel_preds, zero_division=1),
            "iou": intersection_over_union(np.array(all_pixel_preds), np.array(all_pixel_labels)),
            "dice": dice_score(np.array(all_pixel_preds), np.array(all_pixel_labels))
        }

        print(f"IoU Score: {pixel_metrics['iou']:.4f}")
        print(f"Dice Score: {pixel_metrics['dice']:.4f}")
        print(f"Pixel-Wise Accuracy: {pixel_metrics['accuracy'] * 100:.2f}%")
        print(f"Pixel-Wise Precision: {pixel_metrics['precision']:.4f}")
        print(f"Pixel-Wise Recall: {pixel_metrics['recall']:.4f}")
        print(f"Pixel-Wise F1 Score: {pixel_metrics['f1']:.4f}")

        return image_metrics, pixel_metrics

    return image_metrics


def visualize_with_gradcam(model, img_path, img_transform, colormap="jet", alpha_intensity=0.6):
    """Overlay a Grad-CAM heatmap on the original image and display classification results.

    This function computes the Grad-CAM heatmap for an input image, overlays it onto the 
    original image, and displays both the original image and the Grad-CAM visualization 
    with classification results.

    Args:
        model (torch.nn.Module): The trained CNN model for classification.
        img_path (str): Path to the input image.
        img_transform (callable): Transformation function to preprocess the image.
        colormap (str, optional): Colormap for the Grad-CAM heatmap overlay. Default is "jet".
        alpha_intensity (float, optional): Weighting factor for blending the heatmap with 
            the original image. Default is 0.6.

    Returns:
        None: Displays the original image and the Grad-CAM heatmap overlay using Matplotlib.
    """
    model.eval()
    original_img = Image.open(img_path).convert("RGB")
    input_tensor = img_transform(original_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        class_output, _ = model(input_tensor, train_decoder=False)
        probability = torch.sigmoid(class_output).item()

    predicted_class = "Real" if probability >= 0.5 else "Fake"
    confidence = max(probability, 1 - probability)

    gradcam_heatmap = compute_gradcam(model, input_tensor)

    if isinstance(gradcam_heatmap, torch.Tensor):
        gradcam_heatmap = gradcam_heatmap.squeeze().detach().cpu().numpy()

    if gradcam_heatmap.ndim > 2:
        gradcam_heatmap = np.mean(gradcam_heatmap, axis=0)

    gradcam_heatmap = cv2.resize(gradcam_heatmap, (original_img.width, original_img.height))

    gradcam_heatmap = (
        gradcam_heatmap - gradcam_heatmap.min()) / (
        gradcam_heatmap.max() - gradcam_heatmap.min() + 1e-7
    )

    original_img = np.array(original_img, dtype=np.float64) / 255.0

    heatmap_colored = plt.get_cmap(colormap)(gradcam_heatmap)[..., :3]

    if heatmap_colored.shape[:2] != original_img.shape[:2]:
        heatmap_colored = cv2.resize(
            heatmap_colored,
            (original_img.shape[1], original_img.shape[0])
            )

    overlay = (1 - alpha_intensity) * original_img + alpha_intensity * heatmap_colored

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(original_img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(overlay)
    ax[1].set_title(f"Grad-CAM Heatmap\nPrediction: {predicted_class} ({confidence:.2%})")
    ax[1].axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    LABELS_CSV = "/home/pie_crusher/CNN_AI_REAL/labels.csv"
    IMAGE_DIRECTORY = "/home/pie_crusher/CNN_AI_REAL/image_directory"

    # If `--skip-main` is provided, load the model and perform inference
    if "--skip-main" in sys.argv:
        CHECKPOINT_PATH = "/home/pie_crusher/CNN_AI_REAL/best_cnn_real_fake.pth"
        IMAGE_DIR = "/home/pie_crusher/CNN_AI_REAL/image_directory"

        if not os.path.exists(CHECKPOINT_PATH):
            print(f"Error: Checkpoint file {CHECKPOINT_PATH} not found.")
            sys.exit(1)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        best_config = checkpoint["config"]
        cnn_model = CNNModel(
            num_filters=best_config["num_filters"], dropout=best_config["dropout"]
        ).to(DEVICE)
        cnn_model.load_state_dict(checkpoint["model_state_dict"])
        cnn_model.eval()

        print(f"Loaded trained model with validation accuracy: {checkpoint['val_acc']:.2f}%")

        image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg"))]
        if not image_files:
            print(f"Error: No images found in {IMAGE_DIR}")
            sys.exit(1)

        TEST_IMAGE_PATH = os.path.join(IMAGE_DIR, random.choice(image_files))
        print(f"Selected Random Image: {TEST_IMAGE_PATH}")

        visualize_with_gradcam(
            cnn_model, TEST_IMAGE_PATH, transform, colormap="magma", alpha_intensity=0.9
        )

        sys.exit(0)


    dataset = CustomDataset(LABELS_CSV, IMAGE_DIRECTORY, image_transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )


    BATCH_SIZE = 64
    num_workers = min(4, os.cpu_count() - 1)

    train_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=2,
         drop_last=True
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=2,
         drop_last=True
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=2,
         drop_last=True
    )


    search_space = {
        "learning_rate": tune.loguniform(5e-4, 3e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-4),
        "num_filters": tune.choice([96, 128, 160, 192]),
        "dropout": tune.uniform(0.15, 0.3),
        "num_epochs": tune.choice([10, 15, 20, 25]),
        "batch_size": tune.choice([64]),
        "optimizer": tune.choice(["adamw"]),
        }

    if DEBUG_MODE:
        best_config = {
            "num_filters": 32,
            "dropout": 0.3,
            "learning_rate": 1e-3,
            "num_epochs": 5,
            "weight_decay": 1e-4,
            "batch_size": 16,
        }
        print("Debug mode ON: Skipping hyperparameter tuning. Using fixed config:", best_config)

    else:
        scheduler = ASHAScheduler(
            metric="val_acc",
            mode="max",
            max_t=25,
            grace_period=5,
            reduction_factor=3
            )
        tuner = tune.run(
            tune.with_parameters(
                train_hyperparam_tuning,
                train_loader=train_data_loader,
                val_loader=val_data_loader
            ),
            config=search_space,
            num_samples=50,
            scheduler=scheduler,
            resources_per_trial={"cpu": 4, "gpu": 0.25},
            max_concurrent_trials=0
            )

        best_trial = tuner.get_best_trial("val_acc", "max", "last")
        best_config = best_trial.config  # Extract best hyperparameters
        print("Hyperparameter tuning complete. Best config:", best_config)

        best_checkpoint = best_trial.checkpoint
        if best_checkpoint:
            best_model_path = os.path.join(best_checkpoint.to_directory(), "best_model.pth")
            best_model = CNNModel(
                num_filters=best_config["num_filters"],
                dropout=best_config["dropout"]
            ).to(DEVICE)
            best_model.load_state_dict(torch.load(best_model_path))

            print(f"Loaded best model from checkpoint: {best_model_path}")

        else:
            print("Warning: No checkpoint found for best trial. Training model again.")

            best_model_state_dict, _ = train_model(
                CNNModel(
                    num_filters=best_config["num_filters"],
                    dropout=best_config["dropout"]
                ).to(DEVICE),
                train_loader=train_data_loader,
                val_loader=val_data_loader,
                config=best_config
                )

            best_model = CNNModel(
                num_filters=best_config["num_filters"],
                dropout=best_config["dropout"]
            ).to(DEVICE)
            best_model.load_state_dict(best_model_state_dict)

        best_model.eval()

        test_metrics = evaluate_model_with_metrics(best_model, test_data_loader)


    best_model, best_val_acc = train_model(
        CNNModel(
            num_filters=best_config["num_filters"], dropout=best_config["dropout"]
        ).to(DEVICE),
        train_loader=train_data_loader,
        val_loader=val_data_loader,
        config=best_config
    )


    torch.save({
        "model_state_dict": best_model,
        "val_acc": best_val_acc,
        "config": best_config,
        "num_epochs": best_config["num_epochs"],
    }, "best_cnn_real_fake.pth")

    print("Best model saved successfully!")


    checkpoint = torch.load("best_cnn_real_fake.pth")
    cnn_model = CNNModel(
        num_filters=best_config["num_filters"], dropout=best_config["dropout"]
    ).to(DEVICE)
    cnn_model.load_state_dict(checkpoint["model_state_dict"])
    cnn_model.to(DEVICE)

    best_val_acc = checkpoint["val_acc"]
    hyperparams = checkpoint["config"]
    num_epochs = checkpoint["num_epochs"]

    print(f"\nLoaded model with best validation accuracy: {best_val_acc:.2f}%")
    print(f"Hyperparameters used: {hyperparams}")
    print(f"Trained for {num_epochs} epochs")


    test_metrics = evaluate_model_with_metrics(cnn_model, test_data_loader)
    test_accuracy, test_precision, test_recall, test_f1 = test_metrics


    random_idx = random.randint(0, len(test_dataset) - 1)
    image_path = os.path.join(IMAGE_DIRECTORY, dataset.img_labels.iloc[random_idx, 0])
    print(f"Visualizing heatmap for: {image_path}")

    visualize_with_gradcam(cnn_model, image_path, inference_transform)
