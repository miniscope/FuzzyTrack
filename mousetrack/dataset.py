"""Dataset classes for training."""
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
from .config import IMG_SIZE, HEATMAP_SIZE, HEATMAP_SIGMA


class FrameDataset(Dataset):
    """Dataset for CNN training: loads frames and returns motion difference images."""

    def __init__(self, video_path, csv_path, use_heatmap=True, heatmap_sigma=None, cache_frames=True):
        self.video_path = video_path
        self.df = pd.read_csv(csv_path)
        self.cap = None  # Opened lazily per worker (cv2.VideoCapture is not fork-safe)
        self.use_heatmap = use_heatmap
        self.heatmap_sigma = heatmap_sigma if heatmap_sigma is not None else HEATMAP_SIGMA
        self.cache_frames = cache_frames
        self.frame_cache = {}
        self.video_size = None  # (width, height)

        if cache_frames:
            self._preload_frames()

    def _preload_frames(self):
        """Preload all required frames into memory."""
        # Get unique frame indices needed (current and previous for each sample)
        frame_indices = set()
        for _, row in self.df.iterrows():
            frame_idx = row['frame_idx']
            frame_indices.add(frame_idx)
            if frame_idx > 0:
                frame_indices.add(frame_idx - 1)

        # Load frames
        cap = cv2.VideoCapture(self.video_path)
        for frame_idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                if self.video_size is None:
                    self.video_size = (frame.shape[1], frame.shape[0])  # (w, h)
                # Resize and convert to grayscale immediately to save memory
                resized = cv2.resize(frame, IMG_SIZE)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                self.frame_cache[frame_idx] = gray
        cap.release()

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

        if self.cache_frames and frame_idx in self.frame_cache:
            # Use cached frames (already resized and grayscale)
            g_curr = self.frame_cache[frame_idx]
            g_prev = self.frame_cache.get(frame_idx - 1, g_curr) if frame_idx > 0 else g_curr

            # Difference Image (Motion only)
            diff = cv2.absdiff(g_curr, g_prev)
            image = diff.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)  # (1, H, W)

            w_orig, h_orig = self.video_size if self.video_size else IMG_SIZE
        else:
            # Fallback to video reading
            cap = self._get_cap()
            if frame_idx == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret1, frame_prev = cap.read()
                frame_curr = frame_prev.copy() if ret1 else None
                ret2 = ret1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                ret1, frame_prev = cap.read()
                ret2, frame_curr = cap.read()

            if not ret1 or not ret2:
                image = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)
                w_orig, h_orig = IMG_SIZE
            else:
                f_prev = cv2.resize(frame_prev, IMG_SIZE)
                f_curr = cv2.resize(frame_curr, IMG_SIZE)
                g_prev = cv2.cvtColor(f_prev, cv2.COLOR_BGR2GRAY)
                g_curr = cv2.cvtColor(f_curr, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(g_curr, g_prev)
                image = diff.astype(np.float32) / 255.0
                image = np.expand_dims(image, axis=0)
                h_orig, w_orig = frame_curr.shape[:2]

        # Normalize coordinates
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
