"""Training functions for CNN model."""

import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import logger
from .checkpoints import save_checkpoint
from .config import IMG_SIZE
from .dataset import FrameDataset
from .models import MouseHeatmapCNN
from .video import get_video_info


class HeatmapLoss(nn.Module):
    """Combined MSE + peak coordinate loss for heatmap training."""

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
        coord_loss = torch.mean((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)

        return self.mse_weight * mse_loss + self.coord_weight * coord_loss


def _extract_heatmap_coords(
    heatmap: torch.Tensor,
    peak_blend_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract normalized x/y coordinates from a batch of heatmaps."""
    batch_size, _, heatmap_h, heatmap_w = heatmap.shape
    heatmap = torch.clamp(heatmap, min=0.0)
    flat = heatmap.view(batch_size, -1)
    max_indices = torch.argmax(flat, dim=1)
    peak_y = (max_indices // heatmap_w).float()
    peak_x = (max_indices % heatmap_w).float()

    total = flat.sum(dim=1)
    avg_x = torch.full_like(peak_x, (heatmap_w - 1) / 2.0)
    avg_y = torch.full_like(peak_y, (heatmap_h - 1) / 2.0)

    valid = total > 0
    if valid.any():
        y_coords = torch.arange(heatmap_h, device=heatmap.device, dtype=heatmap.dtype).view(
            1, heatmap_h, 1
        )
        x_coords = torch.arange(heatmap_w, device=heatmap.device, dtype=heatmap.dtype).view(
            1, 1, heatmap_w
        )
        avg_x_valid = (heatmap[valid, 0] * x_coords).sum(dim=(1, 2)) / total[valid]
        avg_y_valid = (heatmap[valid, 0] * y_coords).sum(dim=(1, 2)) / total[valid]
        avg_x[valid] = avg_x_valid
        avg_y[valid] = avg_y_valid

    alpha = float(max(0.0, min(1.0, peak_blend_alpha)))
    coord_x = (1.0 - alpha) * avg_x + alpha * peak_x
    coord_y = (1.0 - alpha) * avg_y + alpha * peak_y
    norm_x = (
        torch.clamp(coord_x / (heatmap_w - 1), 0.0, 1.0) if heatmap_w > 1 else coord_x.fill_(0.5)
    )
    norm_y = (
        torch.clamp(coord_y / (heatmap_h - 1), 0.0, 1.0) if heatmap_h > 1 else coord_y.fill_(0.5)
    )
    return norm_x, norm_y


def _pixel_error(
    pred_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    video_width: int,
    video_height: int,
    peak_blend_alpha: float,
) -> torch.Tensor:
    """Return mean Euclidean coordinate error in original video pixels."""
    batch_size, _, heatmap_h, heatmap_w = target_heatmap.shape

    target_flat = target_heatmap.view(batch_size, -1)
    target_max_indices = torch.argmax(target_flat, dim=1)
    target_y = (target_max_indices // heatmap_w).float() / (heatmap_h - 1)
    target_x = (target_max_indices % heatmap_w).float() / (heatmap_w - 1)

    pred_x, pred_y = _extract_heatmap_coords(pred_heatmap, peak_blend_alpha)

    dx_px = (pred_x - target_x) * (video_width - 1)
    dy_px = (pred_y - target_y) * (video_height - 1)
    return torch.sqrt(dx_px**2 + dy_px**2).mean()


def _split_train_val_datasets(
    datasets: list[FrameDataset],
    val_split: float,
    seed: int = 42,
):
    """Split into train/val datasets with a deterministic frame-wise split."""
    full_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    dataset_size = len(full_dataset)
    if dataset_size < 2:
        raise ValueError("Need at least 2 annotated samples to create train/validation splits")

    val_size = int(dataset_size * val_split)
    val_size = max(1, val_size)
    val_size = min(val_size, dataset_size - 1)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    split_summary = f"frame split ({train_size} train / {val_size} val)"
    return train_dataset, val_dataset, train_size, val_size, split_summary


def train_cnn(
    video_paths: str | list[str],
    annotations_paths: str | list[str],
    output_path: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    val_split: float,
    learning_rate: float,
    logdir: str,
    backbone: str = "resnet18",
    input_mode: str = "grayscale_diff",
    heatmap_sigma: float = None,
    peak_blend_alpha: float = 0.25,
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
        heatmap_sigma: Gaussian sigma for heatmap target generation
    """
    # Normalize to lists
    if isinstance(video_paths, str):
        video_paths = [video_paths]
    if isinstance(annotations_paths, str):
        annotations_paths = [annotations_paths]

    if len(video_paths) != len(annotations_paths):
        raise ValueError(
            f"Number of videos ({len(video_paths)}) must match annotations ({len(annotations_paths)})"
        )

    # Load datasets from all video/annotation pairs
    datasets = []
    for video_path, annotations_path in zip(video_paths, annotations_paths, strict=True):
        ds = FrameDataset(
            video_path,
            annotations_path,
            heatmap_sigma=heatmap_sigma,
            input_mode=input_mode,
            cache_frames=cache_frames,
        )
        datasets.append(ds)
        logger.info(f"Loaded {len(ds)} samples from {video_path}")

    train_dataset, val_dataset, train_size, val_size, split_summary = _split_train_val_datasets(
        datasets,
        val_split,
    )

    # Set is_train flag for augmentation
    def set_is_train(dataset, is_train):
        """Recursively set is_train on underlying datasets."""
        if hasattr(dataset, "dataset"):  # Subset
            set_is_train(dataset.dataset, is_train)
        elif hasattr(dataset, "datasets"):  # ConcatDataset
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
    logger.info(f"Input mode: {input_mode}")
    if heatmap_sigma is not None:
        logger.info(f"Heatmap sigma: {heatmap_sigma}")

    # Model and optimizer
    model = MouseHeatmapCNN(backbone=backbone, input_mode=input_mode)
    criterion = HeatmapLoss(mse_weight=1.0, coord_weight=10.0)

    model = model.to(device)
    # Use AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler: reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    # Get video dimensions from first video
    video_info = get_video_info(video_paths[0], require_square=True)
    video_width = video_info.width
    video_height = video_info.height

    # TensorBoard
    writer = SummaryWriter(log_dir=logdir)

    # Training loop
    best_val_loss = float("inf")
    best_val_track_pixel_error = float("inf")
    best_val_argmax_pixel_error = float("inf")
    patience_counter = 0
    best_model_state = None

    logger.info(f"Training on {train_size} samples, validating on {val_size} samples")
    logger.info(f"Validation strategy: {split_summary}")
    logger.info(f"Video dimensions: {video_width}x{video_height}")

    pbar = tqdm(range(max_epochs), desc="Training CNN")

    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
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
        val_track_pixel_error = 0.0
        val_argmax_pixel_error = 0.0
        val_samples = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                pred = model(imgs)
                loss = criterion(pred, targets)
                val_loss += loss.item()
                batch_size = imgs.shape[0]
                track_pixel_error = _pixel_error(
                    pred, targets, video_width, video_height, peak_blend_alpha
                )
                argmax_pixel_error = _pixel_error(pred, targets, video_width, video_height, 1.0)
                val_track_pixel_error += track_pixel_error.item() * batch_size
                val_argmax_pixel_error += argmax_pixel_error.item() * batch_size
                val_samples += batch_size

        avg_val_loss = val_loss / len(val_loader)
        avg_val_track_pixel_error = val_track_pixel_error / max(val_samples, 1)
        avg_val_argmax_pixel_error = val_argmax_pixel_error / max(val_samples, 1)

        # Step learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log to TensorBoard
        writer.add_scalar("Loss/train_coord", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_coord", avg_val_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        writer.add_scalar("Metrics/val_track_error_px", avg_val_track_pixel_error, epoch)
        writer.add_scalar("Metrics/val_argmax_error_px", avg_val_argmax_pixel_error, epoch)

        # Early stopping
        improvement = best_val_loss - avg_val_loss
        if improvement > 1e-4:
            best_val_loss = avg_val_loss
            best_val_track_pixel_error = avg_val_track_pixel_error
            best_val_argmax_pixel_error = avg_val_argmax_pixel_error
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            status = "★ BEST"
        else:
            patience_counter += 1
            status = f"({patience_counter}/{patience})"

        pbar.set_postfix(
            {
                "train_loss": f"{avg_train_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
                "val_px_track": f"{avg_val_track_pixel_error:.1f}",
                "val_px_argmax": f"{avg_val_argmax_pixel_error:.1f}",
                "status": status,
            }
        )

        if patience_counter >= patience:
            pbar.set_postfix({"status": f"Early stopping at epoch {epoch+1}"})
            break

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(
            "Restored best model "
            f"(val_loss: {best_val_loss:.4f}, "
            f"val_track_error_px: {best_val_track_pixel_error:.1f}, "
            f"val_argmax_error_px: {best_val_argmax_pixel_error:.1f})"
        )

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_checkpoint(
        output_path,
        model.state_dict(),
        backbone=backbone,
        input_mode=input_mode,
        heatmap_sigma=heatmap_sigma,
        img_size=IMG_SIZE,
        video_size=(video_width, video_height),
    )
    writer.close()
    logger.info(f"Model saved to {output_path}")
    logger.info(f"TensorBoard logs saved to {logdir}")
