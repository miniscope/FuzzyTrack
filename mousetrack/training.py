"""Training functions for CNN model."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
import numpy as np

from .models import MouseCNN, CoordinateLSTM
from .dataset import FrameDataset, CoordinateSequenceDataset
from .config import IMG_SIZE
from . import logger


def train_cnn(
    video_path: str,
    annotations_path: str,
    output_path: str = "mouse_cnn.pth",
    batch_size: int = 16,
    max_epochs: int = 100,
    patience: int = 10,
    val_split: float = 0.2,
    learning_rate: float = 5e-4,
    logdir: str = "runs/mouse_tracker",
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
    """
    # Load dataset
    full_dataset = FrameDataset(video_path, annotations_path)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model and optimizer
    model = MouseCNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
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
        for imgs, targets_coords in train_loader:
            optimizer.zero_grad()
            pred_coords = model(imgs)
            loss = criterion(pred_coords, targets_coords)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets_coords in val_loader:
                pred_coords = model(imgs)
                loss = criterion(pred_coords, targets_coords)
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


def train_lstm(
    cnn_model_path: str,
    video_path: str,
    annotations_path: str,
    output_path: str = "models/mouse_lstm.pth",
    sequence_length: int = 10,
    batch_size: int = 32,
    max_epochs: int = 2000,
    patience: int = 100,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
):
    """
    Train LSTM model for coordinate refinement.
    
    Args:
        cnn_model_path: Path to trained CNN model
        video_path: Path to input video file
        annotations_path: Path to annotations CSV file
        output_path: Path to save trained model
        sequence_length: Length of input sequences
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        learning_rate: Learning rate
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CNN model
    cnn = MouseCNN()
    cnn.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn.to(device)
    cnn.eval()
    
    # Generate CNN predictions
    dataset = FrameDataset(video_path, annotations_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    cnn_predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for imgs, _, target_coords in tqdm(loader, desc="Generating CNN predictions", leave=False):
            imgs = imgs.to(device)
            pred_coords = cnn(imgs)
            cnn_predictions.append(pred_coords[0].cpu().numpy())
            ground_truth.append(target_coords[0].cpu().numpy())
    
    cnn_predictions = np.array(cnn_predictions)
    ground_truth = np.array(ground_truth)
    
    # Create sequence dataset
    seq_dataset = CoordinateSequenceDataset(
        cnn_predictions, 
        ground_truth, 
        sequence_length=sequence_length
    )
    
    # Split dataset
    train_size = int(len(seq_dataset) * 0.8)
    val_size = len(seq_dataset) - train_size
    train_dataset, val_dataset = random_split(seq_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = CoordinateLSTM(input_size=4, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    logger.info(f"Training on {train_size} sequences, validating on {val_size} sequences")
    logger.info(f"Sequence length: {sequence_length}")
    
    pbar = tqdm(range(max_epochs), desc="Training LSTM")
    
    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0
        for input_seqs, target_seqs in train_loader:
            input_seqs = input_seqs.to(device)
            target_seqs = target_seqs.to(device)
            optimizer.zero_grad()
            refined = model(input_seqs)
            loss = criterion(refined, target_seqs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_seqs, target_seqs in val_loader:
                input_seqs = input_seqs.to(device)
                target_seqs = target_seqs.to(device)
                refined = model(input_seqs)
                loss = criterion(refined, target_seqs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Early stopping
        improvement = best_val_loss - avg_val_loss
        if improvement > 1e-6:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
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
    logger.info(f"Model saved to {output_path}")
