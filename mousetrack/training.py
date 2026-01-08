"""Training functions for CNN model."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
import numpy as np

from .models import MouseCNN, MouseHeatmapCNN, CoordinateLSTM
from .dataset import FrameDataset, CoordinateSequenceDataset
from .config import IMG_SIZE, HEATMAP_SIZE
from . import logger
import pandas as pd


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
    
    # Load annotations
    annotations_df = pd.read_csv(annotations_path)
    annotated_frames = set(annotations_df['frame_idx'].values)
    frame_to_gt = {row['frame_idx']: np.array([row['x'], row['y']], dtype=np.float32) 
                   for _, row in annotations_df.iterrows()}
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Only generate sequences near annotated frames (within window)
    # This avoids inaccurate interpolation over large gaps
    max_gap_from_annotation = sequence_length * 2  # Allow sequences up to 2x sequence_length away from annotations
    
    # Find frame ranges that are close to annotations
    annotated_list = sorted(annotated_frames)
    valid_frame_ranges = []
    for ann_frame in annotated_list:
        start = max(0, ann_frame - max_gap_from_annotation)
        end = min(total_frames, ann_frame + max_gap_from_annotation + 1)
        valid_frame_ranges.append((start, end))
    
    # Merge overlapping ranges
    valid_frame_ranges.sort()
    merged_ranges = []
    for start, end in valid_frame_ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))
    
    # Generate CNN predictions only for valid frame ranges
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Error reading video")
        return
    
    prev_gray = cv2.resize(prev_frame, IMG_SIZE)
    prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
    
    cnn_predictions = []
    ground_truth = []
    frame_indices = []
    
    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc="Generating CNN predictions", leave=False):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if frame is in valid range
            in_valid_range = any(start <= frame_idx < end for start, end in merged_ranges)
            if not in_valid_range:
                prev_gray = cv2.cvtColor(cv2.resize(frame, IMG_SIZE), cv2.COLOR_BGR2GRAY)
                continue
            
            # Preprocess
            curr_small = cv2.resize(frame, IMG_SIZE)
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(curr_gray, prev_gray)
            input_img = diff.astype(np.float32) / 255.0
            input_tensor = torch.tensor(input_img).unsqueeze(0).unsqueeze(0).to(device)
            
            # CNN prediction
            pred_coords = cnn(input_tensor)
            pred_norm = pred_coords[0].cpu().numpy()
            cnn_predictions.append(pred_norm)
            frame_indices.append(frame_idx)
            
            # Ground truth: use annotation if available, otherwise interpolate between nearest annotations
            if frame_idx in frame_to_gt:
                gt_pixel = frame_to_gt[frame_idx]
                gt_norm = np.array([gt_pixel[0] / width, gt_pixel[1] / height], dtype=np.float32)
            else:
                # Linear interpolation between surrounding annotations
                left_idx = max([f for f in annotated_list if f < frame_idx], default=None)
                right_idx = min([f for f in annotated_list if f > frame_idx], default=None)
                if left_idx is not None and right_idx is not None:
                    # Interpolate between left and right annotations
                    alpha = (frame_idx - left_idx) / (right_idx - left_idx)
                    left_gt = frame_to_gt[left_idx]
                    right_gt = frame_to_gt[right_idx]
                    gt_pixel = left_gt * (1 - alpha) + right_gt * alpha
                elif left_idx is not None:
                    gt_pixel = frame_to_gt[left_idx]
                else:
                    gt_pixel = frame_to_gt[right_idx]
                gt_norm = np.array([gt_pixel[0] / width, gt_pixel[1] / height], dtype=np.float32)
            
            ground_truth.append(gt_norm)
            prev_gray = curr_gray
    
    cap.release()
    
    cnn_predictions = np.array(cnn_predictions)
    ground_truth = np.array(ground_truth)
    
    logger.info(f"Generated {len(cnn_predictions)} predictions from {len(merged_ranges)} valid ranges near annotations")
    
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
