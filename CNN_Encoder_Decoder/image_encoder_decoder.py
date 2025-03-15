"""
Image Encoder-Decoder Module for AI-Generated Image Detection.

This module implements a convolutional neural network (CNN) with a UNet-style decoder 
to classify AI-generated images and visualize activations using Grad-CAM. It includes 
functions for model training, evaluation, hyperparameter tuning, and result visualization.

Main Features:
--------------
- **CustomDataset**: PyTorch dataset class for loading images and labels from a CSV file.
- **CNNModel**: A convolutional neural network with Grad-CAM support and a UNet-style decoder.
- **UNetDecoder**: Decoder that progressively upsamples feature maps and incorporates skip connections.
- **FocalLoss**: Custom loss function designed to handle class imbalance.
- **Grad-CAM Visualization**: Computes and overlays Grad-CAM heatmaps for model interpretability.
- **Hyperparameter Tuning**: Uses Ray Tune for automatic model tuning.
- **Early Stopping & Checkpointing**: Implements training checkpoints and early stopping mechanisms.
- **Evaluation Metrics**: Computes accuracy, precision, recall, F1-score, IoU, and Dice score.
- **Principal Component Analysis (PCA)**: Visualizes feature distributions using PCA.

Dependencies:
-------------
This module requires the following Python libraries:
- **Deep Learning**: `torch`, `torchvision`
- **Data Processing**: `numpy`, `pandas`
- **Image Processing**: `opencv-python` (`cv2`), `PIL` (Pillow)
- **Visualization**: `matplotlib`, `seaborn` (optional)
- **Hyperparameter Tuning**: `ray[tune]`
- **Evaluation Metrics**: `scikit-learn` (`sklearn.metrics`)
- **Dimensionality Reduction**: `sklearn.decomposition.PCA`

For usage examples, refer to the **README** file.

Author:
-------
Pierce Ohlmeyer-Dawson and Abhinav Madabhushi
"""

# Standard Library
import os
import sys
import random
import json

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
from ray.train import Checkpoint

# Scikit-Learn (Evaluation Metrics)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

#If DEBUG_MODE is TRUE, it will skip hyperparameter tuning
DEBUG_MODE = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Augment with random flips
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Small geometric distortions
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # Adjust color properties
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((32, 32)),  
    transforms.Normalize([0.5], [0.5])  # Center pixel values around 0
])

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((32, 32)),
    transforms.Normalize([0.5], [0.5])
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
        pt = torch.exp(-bce_loss)
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
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        self.silu2 = nn.SiLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

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

def evaluate_val_loss(model, dataloader, criterion):
    """Evaluate validation loss over the dataset.

    Runs the model in evaluation mode without updating gradients and computes 
    the average loss across the validation set.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (callable): Loss function used for evaluation.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            class_output, _ = model(images, train_decoder=False)
            loss = criterion(class_output, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


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
            - dict: The state dictionary of the best-performing model.
            - float: The highest validation accuracy achieved during training.
            - dict: Training history containing loss, accuracy, and segmentation metrics.
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
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "iou": [],
        "dice": []
    }

    

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
        
        val_loss = evaluate_val_loss(model_instance, val_loader, criterion)
        train_scheduler.step()
        
        history["train_loss"].append(train_metrics['loss'])
        history["train_acc"].append(train_metrics['accuracy'])
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["iou"].append(iou if iou is not None else np.nan)
        history["dice"].append(dice if dice is not None else np.nan)


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
        if val_acc > state["best_val_acc"]:
            state["best_val_acc"] = val_acc
            state["best_model"] = model_instance.state_dict()

    return state["best_model"], state["best_val_acc"], history


def check_early_stopping(state, model, val_acc, heatmap_acc, epoch):
    """Check if training should stop early and save the best model checkpoint.

    This function monitors validation accuracy (`val_acc`) and heatmap accuracy (`heatmap_acc`).
    If either metric improves, it updates the best model state and resets the no-improvement counter.
    If no improvement occurs for a consecutive number of epochs equal to the patience threshold, 
    early stopping is triggered.

    Args:
        state (dict): A dictionary tracking training progress:
            - `"best_val_acc"` (float): Highest recorded validation accuracy.
            - `"best_model"` (dict): Model checkpoint of the best-performing model.
            - `"patience"` (int): Maximum epochs allowed without improvement.
            - `"no_improve_epochs"` (int): Consecutive epochs without improvement.
        model (torch.nn.Module): The model being trained.
        val_acc (float): Current validation accuracy.
        heatmap_acc (float): Current heatmap accuracy (IoU/Dice score).
        epoch (int): Current training epoch.

    Returns:
        bool: `True` if early stopping is triggered (patience exceeded), otherwise `False`.
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

    - **Focal Loss:** Reduces the impact of class imbalance by down-weighting easy examples 
        and focusing more on hard-to-classify cases.
    - **IoU Loss:** Measures the overlap loss between predicted and target masks, 
        penalizing low intersection-over-union (IoU).
    - **Total Variation (TV) Loss:** Promotes spatial smoothness by penalizing local pixel variations.

    Args:
        preds (torch.Tensor): Model output logits of shape (N, C, H, W), where C is the number of classes.
        targets (torch.Tensor): Ground truth binary masks of shape (N, C, H, W).
        alpha (float, optional): Focal loss weighting factor for positive samples. Default is 0.25.
        gamma (float, optional): Focal loss focusing parameter. Default is 2.0.
        tv_weight (float, optional): Weighting factor for total variation loss. Default is 0.01.

    Returns:
        torch.Tensor: A scalar tensor representing the combined loss value.
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
    """Initialize the optimizer, scheduler, loss functions, and gradient scaler.

    This function sets up essential components for training, including:

    - **Optimizer:** Configurable optimizer (`adamw`, `adam`, `rmsprop`, `sgd`, `nadam`, `adagrad`).
    - **Learning Rate Scheduler:** StepLR with a decay factor.
    - **Loss Functions:** BCEWithLogitsLoss for classification and Focal IoU TV loss for heatmaps.
    - **Gradient Scaler:** Enables mixed-precision training for efficiency.

    Args:
        model (torch.nn.Module): The neural network model being trained.
        config (dict): Training configuration parameters:
            - `"learning_rate"` (float): Initial learning rate.
            - `"weight_decay"` (float): Regularization weight decay.
            - `"optimizer"` (str): Name of the optimizer to use.
            - `"momentum"` (float, optional): Momentum factor (for SGD only).

    Returns:
        tuple:
            - torch.optim.Optimizer: Configured optimizer.
            - torch.optim.lr_scheduler.StepLR: Learning rate scheduler.
            - torch.nn.Module: Binary cross-entropy loss function.
            - callable: Focal IoU TV loss function.
            - torch.cuda.amp.GradScaler: Mixed-precision training scaler.

    Raises:
        ValueError: If the specified optimizer is not recognized.
    """
    optimizer_name = config["optimizer"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    momentum = config["momentum"]

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "nadam":
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"Using optimizer: {optimizer_name} | LR: {learning_rate:.6f} | Weight Decay: {weight_decay:.6f}")

    train_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()
    focal_criterion = focal_iou_tv_loss

    scaler = torch.amp.GradScaler(enabled=True)
    
    return optimizer, train_scheduler, criterion, focal_criterion, scaler


def train_one_epoch(model, dataloader, optim_params, loss_params, config):
    """Train the model for one epoch and return training metrics.

    This function performs a full pass over the training dataset, optimizing 
    the model using mixed-precision training. It calculates the total loss 
    (classification + heatmap loss) and tracks training accuracy.

    Mixed precision is enabled via `torch.amp.autocast` to improve computational efficiency.

    Args:
        model (torch.nn.Module): The neural network model being trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optim_params (dict): Dictionary containing optimizer and gradient scaler:
            - `"optimizer"` (torch.optim.Optimizer): The optimizer.
            - `"scaler"` (torch.cuda.amp.GradScaler): Scaler for mixed-precision training.
        loss_params (dict): Dictionary containing loss functions:
            - `"criterion"` (callable): Classification loss function.
            - `"focal_criterion"` (callable): Heatmap loss function.
        config (dict): Training configuration options:
            - `"train_decoder"` (bool, optional): Whether to train the decoder. Defaults to `True`.
            - `"train_encoder"` (bool, optional): Whether to train the encoder. Defaults to `True`.

    Returns:
        dict: Training metrics for the epoch:
            - `"loss"` (float): The average training loss.
            - `"accuracy"` (float): Training accuracy as a percentage.
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
    """Compute the Intersection over Union (IoU) score for binary masks.

    IoU measures the overlap between predicted and ground truth masks, used 
    in image segmentation tasks. It is defined as the ratio of the intersection to 
    the union of the two sets:

        IoU = (intersection + ε) / (union + ε)

    where ε = 1e-6 is a small constant added to prevent division by zero.

    Args:
        preds (torch.Tensor): The predicted binary mask of shape (N, C, H, W).
                            If `preds` contains logits, thresholding should be applied.
        labels (torch.Tensor): The ground truth binary mask of shape (N, C, H, W).

    Returns:
        torch.Tensor: Scalar IoU score representing the overlap between `preds` and `labels`.
    """
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def dice_score(preds, labels):
    """Compute the Dice coefficient score for binary segmentation masks.

    The Dice coefficient measures the similarity between predicted and ground truth masks, 
    commonly used for evaluating segmentation models. It is defined as:

        Dice = (2 * |A ∩ B|) / (|A| + |B|)

    where A is the predicted mask and B is the ground truth mask. A small constant (ε = 1e-6) 
    is added to prevent division by zero.

    Args:
        preds (torch.Tensor): The predicted binary mask of shape (N, C, H, W).
                            If `preds` contains logits, thresholding should be applied.
        labels (torch.Tensor): The ground truth binary mask of shape (N, C, H, W).

    Returns:
        torch.Tensor: A scalar tensor representing the Dice coefficient score, ranging from 0 to 1.
                    A higher score indicates better segmentation performance.
    """
    intersection = (preds * labels).sum()
    return (2 * intersection + 1e-6) / (preds.sum() + labels.sum() + 1e-6)


def evaluate_model(model, dataloader, evaluate_heatmap=False):
    """Evaluate the model for classification accuracy and optional heatmap metrics.

    This function computes the **classification accuracy** of the model on the provided dataset.
    If `evaluate_heatmap` is enabled, it also calculates **Intersection over Union (IoU)** 
    and **Dice score** for the predicted heatmaps, which measure segmentation quality.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        evaluate_heatmap (bool, optional): Whether to compute segmentation metrics (IoU and Dice). 
            Defaults to `False`.

    Returns:
        tuple:
            - float: The classification accuracy as a percentage.
            - tuple[float, float] or None: If `evaluate_heatmap=True`, returns `(iou, dice)`, 
              where both IoU and Dice scores are floats. Otherwise, returns `None`.
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
    """Train a CNN model with hyperparameter tuning and checkpointing.

    This function trains a convolutional neural network while optimizing hyperparameters. 
    It supports:
    
    - **Multiple optimizers**, selectable via `config["optimizer"]`.
    - **Mixed-precision training** using `torch.amp.autocast` and gradient scaling.
    - **Checkpointing** to resume training and store the best model state.
    - **Validation accuracy tracking** to monitor performance.

    Args:
        config (dict): Training configuration, including:
            - `"num_filters"` (int): Number of filters in the CNN model.
            - `"dropout"` (float): Dropout rate for regularization.
            - `"optimizer"` (str): Name of the optimizer (`adamw`, `adam`, `rmsprop`, etc.).
            - `"learning_rate"` (float): Learning rate for the optimizer.
            - `"weight_decay"` (float): L2 regularization factor.
            - `"momentum"` (float, optional): Momentum (for SGD-based optimizers).
            - `"num_epochs"` (int): Total training epochs.
        train_loader (torch.utils.data.DataLoader, optional): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset.

    Returns:
        None: Training results and validation accuracy are logged during execution.
        Checkpoints are saved automatically to resume training if needed.
    """

    model_instance = CNNModel(num_filters=config["num_filters"],
                              dropout=config["dropout"]).to(DEVICE)

    optimizer_name = config["optimizer"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    momentum = config["momentum"]

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model_instance.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "nadam":
        optimizer = optim.NAdam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"Using optimizer: {optimizer_name} | LR: {learning_rate:.6f} | Weight Decay: {weight_decay:.6f}")

    criterion = nn.BCEWithLogitsLoss()
    focal_criterion = focal_iou_tv_loss
    scaler = torch.amp.GradScaler(enabled=True)

    checkpoint = session.get_checkpoint()
    best_val_acc = 0.0
    start_epoch = 0

    checkpoint_dir_name = "./checkpoints"
    os.makedirs(checkpoint_dir_name, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir_name, "checkpoint.pth")

    if checkpoint:
        checkpoint_data = checkpoint.to_dict()
        model_instance.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        best_val_acc = checkpoint_data.get("best_val_acc", 0.0)
        start_epoch = checkpoint_data.get("epoch", 0) + 1
        print(f"Resuming training from epoch {start_epoch}, best val_acc: {best_val_acc:.2f}")

    for epoch in range(start_epoch, config["num_epochs"]):
        model_instance.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                class_output, heatmap = model_instance(images, train_decoder=False)
                loss = criterion(class_output, labels) + 0.5 * focal_criterion(
                    heatmap, labels.view(-1, 1, 1, 1).expand_as(heatmap)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model_instance.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                class_output, _ = model_instance(images, train_decoder=False)
                predictions = torch.sigmoid(class_output).round()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Validation Accuracy: {val_acc:.2f}%, Training Loss: {avg_train_loss:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "epoch": epoch,
                "model_state_dict": model_instance.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }, checkpoint_path)

            checkpoint = Checkpoint.from_directory(checkpoint_dir_name)

            session.report(
                {"val_acc": val_acc, "train_loss": avg_train_loss, "epoch": epoch},
                checkpoint=checkpoint
            )
        else:
            session.report({"val_acc": val_acc, "train_loss": avg_train_loss, "epoch": epoch})

def plot_pca_features(features, labels, n_components=3):
    """Plot PCA-transformed CNN features in 2D or 3D.

    This function applies Principal Component Analysis (PCA) to reduce the 
    dimensionality of CNN-extracted features and visualizes the transformed 
    data. It supports both **2D (n_components=2)** and **3D (n_components=3)** plots.

    Args:
        features (np.ndarray or torch.Tensor): The extracted CNN features, 
            with shape (N, D), where N is the number of samples, and D is the 
            feature dimensionality.
        labels (np.ndarray or torch.Tensor): Binary labels of shape (N,), 
            where 0 represents "FAKE" and 1 represents "REAL".
        n_components (int, optional): Number of principal components to retain. 
            Must be either `2` or `3`. Defaults to `3`.

    Raises:
        ValueError: If `n_components` is not 2 or 3.

    Returns:
        None: Displays a 2D or 3D scatter plot of the PCA-transformed features.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance (first {n_components} PCs): {explained_variance}")
    print(f"Total variance explained: {explained_variance.sum() * 100:.2f}%")

    if n_components == 2:
        plt.scatter(pca_result[labels==0, 0], pca_result[labels==0, 1], label='FAKE', alpha=0.6)
        plt.scatter(pca_result[labels==1, 0], pca_result[labels==1, 1], label='REAL', alpha=0.6)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('2D PCA of CNN Features')
        plt.legend()
        plt.grid()
        plt.show()
    elif n_components == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_result[labels==0, 0], pca_result[labels==0, 1], pca_result[labels==0, 2], label='FAKE', alpha=0.6)
        ax.scatter(pca_result[labels==1, 0], pca_result[labels==1, 1], pca_result[labels==1, 2], label='REAL', alpha=0.6)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D PCA of CNN Features')
        ax.legend()
        plt.show()
        
def plot_explained_variance(features, max_components=15):
    """Plot the explained variance of principal components.

    This function applies Principal Component Analysis (PCA) to the given features
    and visualizes both the **individual explained variance** and the **cumulative 
    explained variance** across the first `max_components` principal components.

    Args:
        features (np.ndarray or torch.Tensor): The extracted CNN features with 
            shape (N, D), where N is the number of samples and D is the feature dimensionality.
        max_components (int, optional): The number of principal components to consider.
            Defaults to `15`.

    Returns:
        None: Displays a bar chart and a cumulative variance step plot.
    """
    pca = PCA(n_components=max_components)
    pca.fit(features)
    
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8,5))
    plt.bar(range(1, max_components+1), explained_variance, alpha=0.6, label='Individual Explained Variance')
    plt.step(range(1, max_components+1), np.cumsum(explained_variance), where='mid', color='red', label='Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.xticks(range(1, max_components+1))
    plt.legend()
    plt.grid()
    plt.show()
    

def plot_confusion_matrix(true_labels, predicted_labels, classes=['FAKE', 'REAL']):
    """Plot a confusion matrix for classification results.

    This function generates a confusion matrix visualization to assess the 
    performance of a binary classification model. It displays the number of 
    correctly and incorrectly classified samples.

    Args:
        true_labels (array-like): Ground truth labels of shape (N,).
        predicted_labels (array-like): Model-predicted labels of shape (N,).
        classes (list, optional): List of class names corresponding to label values.
            Defaults to `['FAKE', 'REAL']`.

    Returns:
        None: Displays the confusion matrix plot.
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.grid(False)
    plt.show()

def plot_training_history(train_loss, val_loss, train_acc, val_acc):
    """Plot the training history of loss and accuracy.

    This function visualizes the training process by plotting:
    
    - **Training vs. Validation Loss** over epochs.
    - **Training vs. Validation Accuracy** over epochs.

    Args:
        train_loss (list or np.ndarray): Training loss values per epoch.
        val_loss (list or np.ndarray): Validation loss values per epoch.
        train_acc (list or np.ndarray): Training accuracy values per epoch.
        val_acc (list or np.ndarray): Validation accuracy values per epoch.

    Returns:
        None: Displays the training history plots.
    """
    epochs = range(1, len(train_loss)+1)

    plt.figure(figsize=(14,5))

    # Loss plot
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid()

    # Accuracy plot
    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def evaluate_model_with_metrics(model, dataloader, evaluate_heatmap=False):
    """Evaluate a model on classification and optional segmentation metrics.

    This function assesses a trained model on a dataset, computing:
    
    - **Classification Metrics:** Accuracy, precision, recall, and F1-score.
    - **Segmentation Metrics (Optional):** IoU, Dice score, and pixel-wise accuracy, precision, recall, and F1-score.

    If `evaluate_heatmap=True`, the function evaluates pixel-wise predictions for segmentation tasks.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        evaluate_heatmap (bool, optional): If `True`, computes segmentation metrics. 
            Defaults to `False`.

    Returns:
        dict: A dictionary containing classification metrics:
            - `"accuracy"` (float): Image classification accuracy.
            - `"precision"` (float): Classification precision score.
            - `"recall"` (float): Classification recall score.
            - `"f1"` (float): Classification F1-score.

        If `evaluate_heatmap=True`, returns an additional dictionary with pixel-wise segmentation metrics:
            - `"accuracy"` (float): Pixel-wise segmentation accuracy.
            - `"precision"` (float): Pixel-wise precision.
            - `"recall"` (float): Pixel-wise recall.
            - `"f1"` (float): Pixel-wise F1-score.
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


def visualize_with_gradcam(model, img_path, img_transform, colormap="jet_r", alpha_intensity=0.6):
    """Overlay Grad-CAM and decoder heatmaps on an image and display classification results.

    This function applies Grad-CAM to visualize class-specific activations and overlays 
    both the **Grad-CAM heatmap** and the **decoder heatmap** on the original image. 
    It enhances the decoder heatmap contrast using CLAHE and ensures proper normalization 
    for clear visual overlays.

    Enhancements:
        - Uses CLAHE to improve decoder heatmap contrast.
        - Ensures correct normalization of both heatmaps.
        - Overlays heatmaps with adjustable transparency.

    Args:
        model (torch.nn.Module): The trained CNN model for classification.
        img_path (str): Path to the input image.
        img_transform (callable): Transformation function for preprocessing the image.
        colormap (str, optional): Colormap for visualizing heatmaps. Default is `"jet_r"`.
        alpha_intensity (float, optional): Transparency factor for blending heatmaps with 
            the original image (range: 0-1). Default is `0.6`.

    Returns:
        None: Displays three images:
            - The original image.
            - The Grad-CAM overlay with classification results.
            - The enhanced decoder heatmap overlay.
    """
    model.eval()
    original_img = Image.open(img_path).convert("RGB")
    input_tensor = img_transform(original_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        class_output, decoder_heatmap = model(input_tensor, train_decoder=True)
        probability = torch.sigmoid(class_output).item()

    predicted_class = "Real" if probability >= 0.5 else "Fake"
    confidence = max(probability, 1 - probability)

    gradcam_heatmap = compute_gradcam(model, input_tensor)

    def process_heatmap(heatmap, img_shape, enhance_contrast=False):
        """Process a heatmap for visualization.

        This function converts a model-generated heatmap into a format suitable for display. 
        It ensures the heatmap is correctly resized, normalized, and optionally enhances its 
        contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            heatmap (torch.Tensor or np.ndarray): The input heatmap, which can be a PyTorch tensor 
                or a NumPy array. If a tensor, it is converted to a NumPy array.
            img_shape (tuple): Target image shape as (height, width) for resizing.
            enhance_contrast (bool, optional): Whether to apply contrast enhancement using CLAHE.
                Defaults to `False`.

        Returns:
            np.ndarray: A processed heatmap in RGB format, ready for visualization.
        """
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.squeeze().detach().cpu().numpy()
        if heatmap.ndim > 2:
            heatmap = np.mean(heatmap, axis=0)

        heatmap = cv2.resize(heatmap, img_shape)

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)

        if enhance_contrast:
            heatmap = np.uint8(255 * heatmap)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            heatmap = clahe.apply(heatmap)
            heatmap = heatmap / 255.0

        return plt.get_cmap(colormap)(heatmap)[..., :3]

    original_img = np.array(original_img, dtype=np.float64) / 255.0

    gradcam_overlay = (1 - alpha_intensity) * original_img + alpha_intensity * process_heatmap(
        gradcam_heatmap, (original_img.shape[1], original_img.shape[0])
    )

    decoder_overlay = (1 - alpha_intensity) * original_img + alpha_intensity * process_heatmap(
        decoder_heatmap.squeeze().cpu().numpy(), (original_img.shape[1], original_img.shape[0]), enhance_contrast=True
    )

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(original_img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(gradcam_overlay)
    ax[1].set_title(f"Grad-CAM Heatmap\nPrediction: {predicted_class} ({confidence:.2%})")
    ax[1].axis("off")

    ax[2].imshow(decoder_overlay)
    ax[2].set_title("Enhanced Decoder Heatmap (CLAHE Applied)")
    ax[2].axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    LABELS_CSV = "/home/pie_crusher/CNN_AI_REAL/labels.csv"
    IMAGE_DIRECTORY = "/home/pie_crusher/CNN_AI_REAL/image_directory"

    # Skip training if "--skip-main" argument is provided and load a pre-trained model
    if "--skip-main" in sys.argv:
        try:
            # Retrieve checkpoint and parameters file paths from command-line arguments
            checkpoint_index = sys.argv.index("--checkpoint-path") + 1
            params_index = sys.argv.index("--params-path") + 1

            CHECKPOINT_PATH = sys.argv[checkpoint_index]
            PARAMS_PATH = sys.argv[params_index]
        except (ValueError, IndexError):
            print("Error: Missing arguments.")
            sys.exit(1)

        # Ensure the checkpoint and parameter files exist
        if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(PARAMS_PATH):
            print("Checkpoint or params file missing.")
            sys.exit(1)

        # Load hyperparameters from the provided JSON file
        with open(PARAMS_PATH, "r") as f:
            best_config = json.load(f)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model with the best hyperparameters
        cnn_model = CNNModel(num_filters=best_config["num_filters"], dropout=best_config["dropout"]).to(DEVICE)

        # Load model checkpoint with training history
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        cnn_model.load_state_dict(checkpoint["model_state_dict"])
        cnn_model.eval()

        print(f"Loaded model from: {CHECKPOINT_PATH}")
        print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")

        # Plot training history if available
        if "history" in checkpoint:
            history = checkpoint["history"]
            print("Plotting Training History...")
            plot_training_history(
                history["train_loss"], 
                history["val_loss"], 
                history["train_acc"], 
                history["val_acc"]
            )
        else:
            print("No training history found in checkpoint. Skipping history plot.")

        # Load test dataset from checkpoint if available, otherwise create a new dataset
        if "test_dataset" in checkpoint:
            test_dataset = checkpoint["test_dataset"]
            print("Loaded test dataset from checkpoint.")
        else:
            print("No test dataset found in checkpoint. Using default dataset.")
            test_dataset = CustomDataset(LABELS_CSV, IMAGE_DIRECTORY, image_transform=transform)

        # Set up test DataLoader
        BATCH_SIZE = 64
        num_workers = min(4, os.cpu_count() - 1)

        test_data_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2, drop_last=False
        )

        print("Test DataLoader rebuilt successfully.")

        # Evaluate the model on the test set
        test_metrics = evaluate_model_with_metrics(cnn_model, test_data_loader)

        # Extract CNN features for PCA and confusion matrix analysis
        all_features, all_labels, all_preds = [], [], []
        with torch.no_grad():
            for images, labels in test_data_loader:
                images = images.to(DEVICE)
                feats, _ = cnn_model(images, train_decoder=False)

                # Extract global features from the model
                global_feats = cnn_model.global_pool(
                    cnn_model.conv2(
                        cnn_model.pool1(cnn_model.silu1(cnn_model.conv1(images)))
                    )
                ).view(images.size(0), -1).cpu().numpy()

                # Obtain classification predictions
                preds = torch.sigmoid(feats).round().cpu().numpy()

                all_features.append(global_feats)
                all_labels.append(labels.numpy())
                all_preds.append(preds)

        # Convert lists to numpy arrays
        all_features = np.concatenate(all_features)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)

        # Visualize PCA projections and explained variance
        plot_pca_features(all_features, all_labels, n_components=2)
        plot_pca_features(all_features, all_labels, n_components=3)
        plot_explained_variance(all_features, max_components=15)

        # Generate confusion matrix
        plot_confusion_matrix(all_labels, all_preds)

        # Select a random image from the dataset and visualize Grad-CAM
        TEST_IMAGE_PATH = os.path.join(IMAGE_DIRECTORY, random.choice(os.listdir(IMAGE_DIRECTORY)))
        print(f"Visualizing Grad-CAM for image: {TEST_IMAGE_PATH}")
        visualize_with_gradcam(cnn_model, TEST_IMAGE_PATH, inference_transform, colormap="jet_r", alpha_intensity=0.9)

        sys.exit(0)

    # Initialize the dataset and apply transformations
    dataset = CustomDataset(LABELS_CSV, IMAGE_DIRECTORY, image_transform=transform)

    # Split dataset into training (80%), validation (10%), and test (10%) sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Define batch size and set number of workers for parallel data loading
    BATCH_SIZE = 64
    num_workers = min(4, os.cpu_count() - 1)  # Ensure efficient parallel processing

    # Create data loaders for training, validation, and testing
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=(num_workers > 0), prefetch_factor=2, drop_last=True)

    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=(num_workers > 0), prefetch_factor=2, drop_last=True)

    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=(num_workers > 0), prefetch_factor=2, drop_last=True)

    # Define the hyperparameter search space for tuning
    search_space = {
        "optimizer": tune.choice(["adamw", "adam", "rmsprop", "sgd", "nadam", "adagrad"]),

        "learning_rate": tune.sample_from(
            lambda config: float(np.random.uniform(5e-4, 3e-3)) if config["optimizer"] in ["adamw", "adam", "nadam"]
            else float(np.random.uniform(1e-4, 2e-3)) if config["optimizer"] == "rmsprop"
            else float(np.random.uniform(5e-4, 1e-2)) if config["optimizer"] == "sgd"
            else float(np.random.uniform(1e-3, 5e-3))
        ),

        "weight_decay": tune.loguniform(1e-6, 1e-3),  

        "num_filters": tune.choice([64, 96, 128, 160, 192]),  

        "dropout": tune.uniform(0.05, 0.5),  

        "num_epochs": tune.choice([10, 15, 20, 25, 30, 50]),

        "momentum": tune.sample_from(
            lambda config: float(np.random.uniform(0.6, 0.98)) if config["optimizer"] == "sgd" else 0.0
        )
    }

    # If debug mode is enabled, skip hyperparameter tuning and use a predefined config
    if DEBUG_MODE:
        best_config = {
            "num_filters": 32,
            "dropout": 0.3,
            "learning_rate": 1e-3,
            "num_epochs": 5,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "optimizer": "adamw",
        }
        print("Debug mode ON: Skipping hyperparameter tuning. Using fixed config:", best_config)

    else:
        # Define ASHA scheduler for efficient hyperparameter search
        scheduler = ASHAScheduler(
            metric="val_acc",
            mode="max",
            max_t=50,
            grace_period=5,
            reduction_factor=3
        )

        # Run hyperparameter tuning using Ray Tune
        tuner = tune.run(
            tune.with_parameters(
                train_hyperparam_tuning,
                train_loader=train_data_loader,
                val_loader=val_data_loader
            ),
            config=search_space,
            num_samples=1000,
            scheduler=scheduler,
            resources_per_trial={"cpu": 4, "gpu": 0.1},
            storage_path=os.path.abspath("./ray_results"),
            name="train_hyperparam_tuning",
            checkpoint_score_attr="val_acc",
            resume="Auto"
        )

        # Retrieve the best trial configuration from tuning
        best_trial = tuner.get_best_trial("val_acc", "max", "last")
        best_config = best_trial.config
        print("Hyperparameter tuning complete. Best config:", best_config)

        # Load the best model checkpoint if available
        best_checkpoint = tuner.get_best_checkpoint(best_trial, metric="val_acc", mode="max")

        # Initialize model with best hyperparameters
        best_model = CNNModel(
            num_filters=best_config["num_filters"],
            dropout=best_config["dropout"]
        ).to(DEVICE)

        # Load model state from the best checkpoint
        if best_checkpoint:
            best_model_path = os.path.join(best_checkpoint.to_directory(), "checkpoint.pth")

            if os.path.exists(best_model_path):
                print(f"Loading best model from {best_model_path}")
                checkpoint = torch.load(best_model_path)
                best_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                best_checkpoint = None

        # If no valid checkpoint is found, retrain the model
        if best_checkpoint is None:
            print("No valid checkpoint found. Training model again.")
            best_model, best_val_acc, history = train_model(
                CNNModel(num_filters=best_config["num_filters"], dropout=best_config["dropout"]).to(DEVICE),
                train_loader=train_data_loader,
                val_loader=val_data_loader,
                config=best_config
            )
            best_model_state_dict = best_model.state_dict()
            best_model.load_state_dict(best_model_state_dict)

        # Evaluate the best model on the test set
        best_model.eval()
        test_metrics = evaluate_model_with_metrics(best_model, test_data_loader)

    # Train the final model using the best configuration found
    best_model, best_val_acc, history = train_model(
        CNNModel(num_filters=best_config["num_filters"], dropout=best_config["dropout"]).to(DEVICE),
        train_loader=train_data_loader,
        val_loader=val_data_loader,
        config=best_config
    )

    # Save the best-trained model along with metadata
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "val_acc": best_val_acc,
        "config": best_config,
        "num_epochs": best_config["num_epochs"],
        "history": history,
        "test_dataset": test_dataset
    }, "best_cnn_real_fake.pth")

    print("Best model saved successfully!")

    # Reload the saved model for further evaluation
    checkpoint = torch.load("best_cnn_real_fake.pth")
    cnn_model = CNNModel(
        num_filters=best_config["num_filters"], dropout=best_config["dropout"]
    ).to(DEVICE)
    cnn_model.load_state_dict(checkpoint["model_state_dict"])
    cnn_model.to(DEVICE)

    # Display summary of the best model
    best_val_acc = checkpoint["val_acc"]
    hyperparams = checkpoint["config"]
    num_epochs = checkpoint["num_epochs"]

    print(f"\nLoaded model with best validation accuracy: {best_val_acc:.2f}%")
    print(f"Hyperparameters used: {hyperparams}")
    print(f"Trained for {num_epochs} epochs")

    # Evaluate the final model on the test dataset
    test_metrics = evaluate_model_with_metrics(cnn_model, test_data_loader)
    test_accuracy, test_precision, test_recall, test_f1 = test_metrics

    # Select a random image from the test dataset for Grad-CAM visualization
    random_idx = random.randint(0, len(test_dataset) - 1)
    image_path = os.path.join(IMAGE_DIRECTORY, dataset.img_labels.iloc[random_idx, 0])
    print(f"Visualizing heatmap for: {image_path}")

    # Generate and display Grad-CAM heatmap for the selected image
    visualize_with_gradcam(cnn_model, image_path, inference_transform)
