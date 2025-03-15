# AI-Generated Image Detection with CNN & Grad-CAM

## Overview
This project implements a **Convolutional Neural Network (CNN) with a UNet-style decoder** for detecting AI-generated images. It supports **Grad-CAM for visualization**, **hyperparameter tuning with Ray Tune**, and various **evaluation metrics**.

## Features
- **AI Image Classification**: Predicts whether an image is AI-generated or real.
- **Grad-CAM Visualization**: Highlights important image regions for model decisions.
- **UNet-Style Decoder**: Generates pixel-wise heatmaps.
- **Hyperparameter Optimization**: Uses Ray Tune for automatic tuning.
- **Evaluation Metrics**: Computes accuracy, precision, recall, F1-score, IoU, and Dice score.
- **Graphical Analysis**: Generates plots for performance evaluation, including loss curves, accuracy trends, PCA projections, and confusion matrices.

## Installation
Install required dependencies:
```bash
pip install torch torchvision numpy pandas opencv-python matplotlib scikit-learn ray[tune] pillow
```
For **GPU acceleration**, install CUDA-compatible PyTorch (https://pytorch.org/get-started/locally/).

Note: Model was trained and ran on a RTX 3080 Laptop GPU

## Usage
### Train the Model
Run the following command to train the model:
```bash
python image_encoder_decoder.py
```

### Evaluate a Pre-Trained Model
To evaluate a saved model, use:
```bash
python image_encoder_decoder.py --skip-main --checkpoint-path <checkpoint.pth> --params-path <params.json>
```
For Replication:
<checkpoint.pth> -> PATH to current_model_check.pth in the new_checkpoint file
<config.json> -> PATH to params.json in the new_checkpoint file

## Hyperparameter Tuning
Modify the **search space** in `image_encoder_decoder.py`:
```python
search_space = {
    "optimizer": tune.choice(["adamw", "adam", "sgd"]),
    "learning_rate": tune.loguniform(1e-4, 1e-2),
    "num_filters": tune.choice([64, 128, 192]),
    "dropout": tune.uniform(0.1, 0.5),
    "num_epochs": tune.choice([10, 20, 30]),
}


## Model Evaluation
The script reports:
- **Test Accuracy**
- **Precision, Recall, and F1-score**
- **IoU and Dice Coefficient**
- **Confusion Matrix**
- **PCA Feature Visualizations**
- **Training and Validation Loss Curves**
- **Training and Validation Accuracy Trends**

### Graphical Analysis
#### **1. Training vs. Validation Loss**
![Loss Curve](results/loss_curve.png)

#### **2. Training vs. Validation Accuracy**
![Accuracy Trend](results/accuracy_trend.png)

#### **3. PCA Projection of CNN Features**
- **2D PCA Projection:**
  ![PCA 2D](results/pca_2d.png)
- **3D PCA Projection:**
  ![PCA 3D](results/pca_3d.png)

#### **4. Confusion Matrix**
![Confusion Matrix](results/confusion_matrix.png)


## Contributors
- **Pierce Ohlmeyer-Dawson**
- **Abhinav Madabhushi**


