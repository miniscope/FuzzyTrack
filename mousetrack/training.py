"""Training functions for CNN model."""
import os
from typing import List, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
import numpy as np

from .models import MouseCNN, MouseHeatmapCNN
from .dataset import FrameDataset
from .config import IMG_SIZE, HEATMAP_SIZE
from . import logger


class HeatmapLoss(nn.Module):
    """Combined MSE + peak coordinate loss for heatmap regression."""
    def __init__(self, mse_weight=1.0, coord_weight=10.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.coord_weight = coord_weight
        self.mse = nn.MSELoss()

    def forward(self, pred_heatmap, target_heatmap):
        """
        Args:
            pred_heatmap: (B, 1, H, W) predicted heatmap
            target_heatmap: (B, 1, H, W) target heatmap
        """
        # 1. Standard MSE loss on heatmaps
        mse_loss = self.mse(pred_heatmap, target_heatmap)

        # 2. Peak coordinate loss (forces correct peak location)
        B, C, H, W = target_heatmap.shape
        target_flat = target_heatmap.view(B, -1)
        target_max_indices = torch.argmax(target_flat, dim=1)
        target_y = (target_max_indices // W).float() / (H - 1)
        target_x = (target_max_indices % W).float() / (W - 1)

        # Extract peak from prediction
        pred_flat = pred_heatmap.view(B, -1)
        pred_max_indices = torch.argmax(pred_flat, dim=1)
        pred_y = (pred_max_indices // W).float() / (H - 1)
        pred_x = (pred_max_indices % W).float() / (W - 1)

        # L2 distance between peaks
        coord_loss = torch.mean((pred_x - target_x)**2 + (pred_y - target_y)**2)

        return self.mse_weight * mse_loss + self.coord_weight * coord_loss


def train_cnn(
    video_paths: Union[str, List[str]],
    annotations_paths: Union[str, List[str]],
    output_path: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    val_split: float,
    learning_rate: float,
    logdir: str,
    use_heatmap: bool,
    backbone: str = 'resnet18',
    heatmap_sigma: float = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_frames: bool = True,
):
    """
    Train CNN model for coordinate prediction.

    Args:
        video_paths: Path(s) to input video file(s)
        annotations_paths: Path(s) to annotations CSV file(s) - must match video_paths
        output_path: Path to save trained model
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        val_split: Validation split ratio
        learning_rate: Learning rate
        logdir: TensorBoard log directory
        use_heatmap: If True, use heatmap regression instead of direct coordinates
        heatmap_sigma: Gaussian sigma for heatmap target generation
    """
    # Normalize to lists
    if isinstance(video_paths, str):
        video_paths = [video_paths]
    if isinstance(annotations_paths, str):
        annotations_paths = [annotations_paths]

    if len(video_paths) != len(annotations_paths):
        raise ValueError(f"Number of videos ({len(video_paths)}) must match annotations ({len(annotations_paths)})")

    # Load datasets from all video/annotation pairs
    datasets = []
    total_samples = 0
    for video_path, annotations_path in zip(video_paths, annotations_paths):
        ds = FrameDataset(video_path, annotations_path, use_heatmap=use_heatmap, heatmap_sigma=heatmap_sigma, cache_frames=cache_frames)
        datasets.append(ds)
        total_samples += len(ds)
        logger.info(f"Loaded {len(ds)} samples from {video_path}")

    # Combine all datasets
    if len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        full_dataset = ConcatDataset(datasets)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Set is_train flag for augmentation
    def set_is_train(dataset, is_train):
        """Recursively set is_train on underlying datasets."""
        if hasattr(dataset, 'dataset'):  # Subset
            set_is_train(dataset.dataset, is_train)
        elif hasattr(dataset, 'datasets'):  # ConcatDataset
            for ds in dataset.datasets:
                ds.is_train = is_train
        else:  # Base dataset
            dataset.is_train = is_train

    set_is_train(train_dataset, True)
    set_is_train(val_dataset, False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Backbone: {backbone}")
    if use_heatmap and heatmap_sigma is not None:
        logger.info(f"Heatmap sigma: {heatmap_sigma}")

    # Model and optimizer
    if use_heatmap:
        model = MouseHeatmapCNN(backbone=backbone)
        # Use hybrid loss: MSE + peak coordinate loss
        criterion = HeatmapLoss(mse_weight=1.0, coord_weight=10.0)
    else:
        model = MouseCNN(backbone=backbone)
        criterion = nn.MSELoss()

    model = model.to(device)
    # Use AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler: reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    # Get video dimensions from first video
    cap = cv2.VideoCapture(video_paths[0])
    ret, sample_frame = cap.read()
    if ret:
        video_height, video_width = sample_frame.shape[:2]
    else:
        video_width, video_height = IMG_SIZE[0], IMG_SIZE[1]
    cap.release()
    if video_width != video_height:
        logger.warning(
            f"Warning: training video is non-square ({video_width}x{video_height}). "
            f"Frames are resized to {IMG_SIZE[0]}x{IMG_SIZE[1]}, which stretches aspect ratio."
        )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=logdir)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    logger.info(f"Training on {train_size} samples, validating on {val_size} samples")
    logger.info(f"Video dimensions: {video_width}x{video_height}")
    
    pbar = tqdm(range(max_epochs), desc="Training CNN")
    
    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, targets)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                pred = model(imgs)
                loss = criterion(pred, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        # Step learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log to TensorBoard
        writer.add_scalar("Loss/train_coord", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_coord", avg_val_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        
        # Early stopping
        improvement = best_val_loss - avg_val_loss
        if improvement > 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            status = "★ BEST"
        else:
            patience_counter += 1
            status = f"({patience_counter}/{patience})"
        
        pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'status': status
        })
        
        if patience_counter >= patience:
            pbar.set_postfix({'status': f'Early stopping at epoch {epoch+1}'})
            break
    
    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model (val_loss: {best_val_loss:.4f})")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), output_path)
    writer.close()
    logger.info(f"Model saved to {output_path}")
    logger.info(f"TensorBoard logs saved to {logdir}")
