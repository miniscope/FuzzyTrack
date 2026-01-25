"""Dataset classes for training."""
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
from .config import IMG_SIZE, HEATMAP_SIZE, HEATMAP_SIGMA


class FrameDataset(Dataset):
    """Dataset for CNN training: loads frames and returns motion difference images."""

    def __init__(self, video_path, csv_path, use_heatmap=True, heatmap_sigma=None):
        self.video_path = video_path
        self.df = pd.read_csv(csv_path)
        self.cap = None  # Opened lazily per worker (cv2.VideoCapture is not fork-safe)
        self.use_heatmap = use_heatmap
        self.heatmap_sigma = heatmap_sigma if heatmap_sigma is not None else HEATMAP_SIGMA

    def _get_cap(self):
        """Get VideoCapture, opening it if needed (fork-safe)."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
        return self.cap
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_idx = row['frame_idx']
        target_x = row['x']
        target_y = row['y']
        
        # 1. Load Current Frame and Previous Frame (for motion)
        cap = self._get_cap()
        # Handle edge case: frame_idx 0 has no previous frame
        if frame_idx == 0:
            # For first frame, use current frame as both prev and curr (no motion = all zeros diff)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame_prev = cap.read()
            # Use same frame for curr (no previous frame exists)
            frame_curr = frame_prev.copy() if ret1 else None
            ret2 = ret1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret1, frame_prev = cap.read()
            ret2, frame_curr = cap.read()
        
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

        if self.use_heatmap:
            # Generate heatmap target
            heatmap = self._generate_heatmap(norm_x, norm_y, HEATMAP_SIZE[0], HEATMAP_SIZE[1], self.heatmap_sigma)
            # Add channel dimension: (H, W) -> (1, H, W) to match model output
            heatmap = np.expand_dims(heatmap, axis=0)
            return torch.tensor(image), torch.tensor(heatmap, dtype=torch.float32)
        else:
            return torch.tensor(image), torch.tensor(coords)
    
    def _generate_heatmap(self, x, y, h, w, sigma):
        """Generate Gaussian heatmap from normalized coordinates."""
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Convert normalized coordinates to heatmap coordinates
        hm_x = int(x * w)
        hm_y = int(y * h)
        
        # Clamp to valid range
        hm_x = np.clip(hm_x, 0, w - 1)
        hm_y = np.clip(hm_y, 0, h - 1)
        
        # Generate Gaussian
        y_coords, x_coords = np.ogrid[:h, :w]
        heatmap = np.exp(-((x_coords - hm_x)**2 + (y_coords - hm_y)**2) / (2 * sigma**2))
        
        return heatmap
