"""Dataset classes for training."""
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
from .config import LABEL_MAP, IMG_SIZE


class FrameDataset(Dataset):
    """Dataset for CNN training: loads frames and returns motion difference images."""
    
    def __init__(self, video_path, csv_path):
        self.video_path = video_path
        self.df = pd.read_csv(csv_path)
        self.cap = cv2.VideoCapture(video_path)  # Keep open for speed
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_idx = row['frame_idx']
        label_str = row['label']
        target_x = row['x']
        target_y = row['y']
        
        # 1. Load Current Frame and Previous Frame (for motion)
        # Handle edge case: frame_idx 0 has no previous frame
        if frame_idx == 0:
            # For first frame, use current frame as both prev and curr (no motion = all zeros diff)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame_prev = self.cap.read()
            # Use same frame for curr (no previous frame exists)
            frame_curr = frame_prev.copy() if ret1 else None
            ret2 = ret1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret1, frame_prev = self.cap.read()
            ret2, frame_curr = self.cap.read()
        
        if not ret1 or not ret2:
            # Fallback if read fails (edge cases)
            image = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)
        else:
            # Resize
            f_prev = cv2.resize(frame_prev, IMG_SIZE)
            f_curr = cv2.resize(frame_curr, IMG_SIZE)
            
            # Convert to Gray
            g_prev = cv2.cvtColor(f_prev, cv2.COLOR_BGR2GRAY)
            g_curr = cv2.cvtColor(f_curr, cv2.COLOR_BGR2GRAY)
            
            # Difference Image (Motion only)
            diff = cv2.absdiff(g_curr, g_prev)
            
            # Normalize 0-1
            image = diff.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)  # (1, H, W)

        # 2. Prepare Targets
        # Classification Target (Which Zone?) - kept for compatibility
        label_idx = LABEL_MAP[label_str]
        
        # Regression Target (Where is it?)
        # We predict normalized (0-1) X and Y coordinates on the screen
        if ret2 and frame_curr is not None:
            # frame_curr.shape[:2] returns (height, width)
            h_orig, w_orig = frame_curr.shape[:2]
        else:
            # Fallback: use IMG_SIZE if frame read failed
            w_orig, h_orig = IMG_SIZE[0], IMG_SIZE[1]
        
        # Normalize coordinates: x divided by width, y divided by height
        norm_x = target_x / w_orig
        norm_y = target_y / h_orig
        coords = np.array([norm_x, norm_y], dtype=np.float32)

        return torch.tensor(image), torch.tensor(label_idx), torch.tensor(coords)


class CoordinateSequenceDataset(Dataset):
    """Dataset for LSTM training: sequences of coordinates."""
    
    def __init__(self, input_coordinates, target_coordinates, sequence_length=10, stride=1):
        """
        Args:
            input_coordinates: CNN predictions (noisy input) - List or array of (x, y) coordinates (normalized 0-1)
            target_coordinates: Ground truth annotations (target) - List or array of (x, y) coordinates (normalized 0-1)
            sequence_length: Length of input sequences
            stride: Step size between sequences
        """
        self.input_coords = np.array(input_coordinates, dtype=np.float32)
        self.target_coords = np.array(target_coordinates, dtype=np.float32)
        self.sequence_length = sequence_length
        self.stride = stride
        
        assert len(self.input_coords) == len(self.target_coords), "Input and target must have same length"
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.input_coords) - sequence_length + 1, stride):
            input_seq = self.input_coords[i:i + sequence_length]
            target_seq = self.target_coords[i:i + sequence_length]
            self.sequences.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Input: CNN predictions (noisy) with velocity features
        # Target: Ground truth annotations (clean)
        input_seq, target_seq = self.sequences[idx]
        
        # Compute velocity features for input sequence
        velocities = np.zeros_like(input_seq)
        if len(input_seq) > 1:
            velocities[1:] = input_seq[1:] - input_seq[:-1]
        
        # Combine coordinates and velocity: [x, y, vx, vy]
        input_features = np.concatenate([input_seq, velocities], axis=-1)  # (T, 4)
        
        input_features = torch.tensor(input_features, dtype=torch.float32)
        target_seq = torch.tensor(target_seq, dtype=torch.float32)
        
        return input_features, target_seq
