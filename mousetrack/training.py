"""Training functions for CNN model."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
import numpy as np

from .models import MouseCNN, MouseHeatmapCNN
from .dataset import FrameDataset
from .config import IMG_SIZE, HEATMAP_SIZE
from . import logger


def train_cnn(
    video_path: str,
    annotations_path: str,
    output_path: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    val_split: float,
    learning_rate: float,
    logdir: str,
    use_heatmap: bool,
):
    """
    Train CNN model for coordinate prediction.
    
    Args:
        video_path: Path to input video file
        annotations_path: Path to annotations CSV file
        output_path: Path to save trained model
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        val_split: Validation split ratio
        learning_rate: Learning rate
        logdir: TensorBoard log directory
        use_heatmap: If True, use heatmap regression instead of direct coordinates
    """
    # Load dataset
    full_dataset = FrameDataset(video_path, annotations_path, use_heatmap=use_heatmap)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model and optimizer
    if use_heatmap:
        model = MouseHeatmapCNN()
        criterion = nn.MSELoss()  # MSE loss on heatmaps
    else:
        model = MouseCNN()
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Get video dimensions
    cap = cv2.VideoCapture(video_path)
    ret, sample_frame = cap.read()
    if ret:
        video_height, video_width = sample_frame.shape[:2]
    else:
        video_width, video_height = IMG_SIZE[0], IMG_SIZE[1]
    cap.release()
    
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
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                pred = model(imgs)
                loss = criterion(pred, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train_coord", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_coord", avg_val_loss, epoch)
        
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
    
    torch.save(model.state_dict(), output_path)
    writer.close()
    logger.info(f"Model saved to {output_path}")
    logger.info(f"TensorBoard logs saved to {logdir}")
