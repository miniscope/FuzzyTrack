"""Tracking inference functions."""

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import logger
from .checkpoints import load_checkpoint
from .config import IMG_SIZE
from .models import MouseHeatmapCNN
from .video import get_video_info


@dataclass
class WarmupState:
    enabled: bool
    warmup_frames: int
    min_confident_frames: int
    heatmap_min_confidence: float
    coords: list[np.ndarray]


def extract_coords_from_heatmap(
    heatmap: np.ndarray, peak_blend_alpha: float = 0.0
) -> tuple[np.ndarray, float]:
    """Extract coordinates from a heatmap output.

    peak_blend_alpha blends between weighted-average coordinates (0.0)
    and argmax peak coordinates (1.0).
    """
    if heatmap.ndim == 3:
        heatmap = heatmap[0]

    heatmap = np.maximum(heatmap, 0.0)
    h, w = heatmap.shape
    n_pixels = h * w

    total = np.sum(heatmap)
    if total > 0:
        heatmap_norm = heatmap / total
        entropy = -np.sum(heatmap_norm * np.log(heatmap_norm + 1e-10))
        max_entropy = np.log(n_pixels)
        confidence = 1.0 - (entropy / max_entropy)
        confidence = float(np.clip(confidence, 0.0, 1.0))
    else:
        confidence = 0.0

    if total > 0:
        y_coords, x_coords = np.ogrid[:h, :w]
        avg_x = np.sum(heatmap * x_coords) / total
        avg_y = np.sum(heatmap * y_coords) / total
        avg_x = np.clip(avg_x, 0.0, w - 1.0)
        avg_y = np.clip(avg_y, 0.0, h - 1.0)

        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        alpha = float(np.clip(peak_blend_alpha, 0.0, 1.0))
        hm_x = (1.0 - alpha) * avg_x + alpha * peak_x
        hm_y = (1.0 - alpha) * avg_y + alpha * peak_y
    else:
        hm_x = (w - 1) / 2.0
        hm_y = (h - 1) / 2.0
        confidence = 0.0

    norm_x = np.clip(hm_x / (w - 1) if w > 1 else 0.5, 0.0, 1.0)
    norm_y = np.clip(hm_y / (h - 1) if h > 1 else 0.5, 0.0, 1.0)
    return np.array([norm_x, norm_y], dtype=np.float32), float(confidence)


def _load_tracking_model(cnn_model_path: str, backbone: str, device: torch.device):
    state_dict, _metadata = load_checkpoint(
        cnn_model_path,
        device=device,
        expected_backbone=backbone,
    )
    cnn = MouseHeatmapCNN(backbone=backbone)
    logger.info("Loading heatmap CNN model...")

    cnn.load_state_dict(state_dict)
    cnn.to(device)
    cnn.eval()
    logger.info("CNN model loaded successfully.")
    return cnn


def _open_io(video_path: str, output_video: str, output_csv: str):
    video_info = get_video_info(video_path, require_square=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video: {video_path}")

    fps = video_info.fps
    width = video_info.width
    height = video_info.height
    total_frames = video_info.total_frames
    logger.info(f"Input video: {width}x{height} @ {fps} fps, {total_frames} frames")

    for path in [output_video, output_csv]:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_width = width * 3
    out_video = cv2.VideoWriter(output_video, fourcc, fps, (output_width, height))
    if not out_video.isOpened():
        cap.release()
        raise RuntimeError(f"Error creating output video: {output_video}")

    logger.info(f"Output video: {output_video}")
    return cap, out_video, width, height, total_frames


def _make_warmup_state(
    enable_warmup: bool,
    warmup_frames: int,
    min_warmup_confident_frames: int,
    heatmap_min_confidence: float,
) -> WarmupState:
    min_confident_frames = min_warmup_confident_frames
    if not enable_warmup:
        warmup_frames = 0
        min_confident_frames = 0
    return WarmupState(
        enabled=enable_warmup,
        warmup_frames=warmup_frames,
        min_confident_frames=min_confident_frames,
        heatmap_min_confidence=heatmap_min_confidence,
        coords=[],
    )


def _prepare_heatmap_vis(heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert a raw heatmap into an 8-bit image for QC rendering."""
    heatmap_vis = np.maximum(heatmap, 0.0)
    heatmap_vis = (heatmap_vis / (heatmap_vis.max() + 1e-8) * 255).astype(np.uint8)
    return cv2.resize(heatmap_vis, (width, height), interpolation=cv2.INTER_NEAREST)


def _annotate_heatmap_panel(
    panel: np.ndarray,
    title: str,
    pixel_x: int,
    pixel_y: int,
    heatmap_conf: float | None,
) -> np.ndarray:
    """Add overlays to a rendered heatmap QC panel."""
    cv2.circle(panel, (pixel_x, pixel_y), 8, (255, 255, 255), 2)
    cv2.putText(
        panel,
        title,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        f"Conf: {heatmap_conf:.3f}" if heatmap_conf is not None else "Conf: N/A",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return panel


def _extract_model_coords(
    model_out: torch.Tensor,
    frame_idx: int,
    warmup: WarmupState,
    peak_blend_alpha: float,
) -> tuple[np.ndarray, float | None, np.ndarray | None]:
    heatmap = model_out[0, 0].cpu().numpy()
    raw_coords, heatmap_conf = extract_coords_from_heatmap(
        heatmap, peak_blend_alpha=peak_blend_alpha
    )
    if warmup.enabled and frame_idx < warmup.warmup_frames:
        if heatmap_conf >= warmup.heatmap_min_confidence:
            warmup.coords.append(raw_coords.copy())
            raw_coords = np.mean(warmup.coords, axis=0)
        elif len(warmup.coords) >= warmup.min_confident_frames:
            raw_coords = np.mean(warmup.coords, axis=0)
        elif warmup.coords:
            raw_coords = warmup.coords[-1].copy()
    return raw_coords, heatmap_conf, heatmap


def _refine_coords(
    raw_coords: np.ndarray,
    prev_refined_coords: np.ndarray | None,
    smoothing: float,
    max_speed: float | None,
) -> np.ndarray:
    if max_speed is not None and prev_refined_coords is not None:
        movement = raw_coords - prev_refined_coords
        movement_norm = np.linalg.norm(movement)
        if movement_norm > max_speed:
            movement = movement * (max_speed / movement_norm)
            raw_coords = prev_refined_coords + movement

    if prev_refined_coords is not None and smoothing < 1.0:
        raw_coords = smoothing * raw_coords + (1.0 - smoothing) * prev_refined_coords

    return np.clip(raw_coords, 0.0, 1.0)


def _build_result_row(
    pixel_x: int, pixel_y: int, heatmap_conf: float | None
) -> dict[str, int | float]:
    return {
        "x": pixel_x,
        "y": pixel_y,
        "likelihood": heatmap_conf if heatmap_conf is not None else 0.0,
    }


def _write_tracking_csv(
    results: list[dict[str, int | float]],
    output_csv: str,
    output_scorer: str,
    output_bodypart: str,
) -> None:
    columns = ["x", "y", "likelihood"]
    header_tuples = [(output_scorer, output_bodypart, col) for col in columns]
    multi_index = pd.MultiIndex.from_tuples(header_tuples, names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(results, columns=columns)
    df.columns = multi_index
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv)


def track_video(
    video_path: str,
    cnn_model_path: str,
    output_video: str,
    output_csv: str,
    max_speed: float | None = None,
    smoothing: float = 0.5,
    peak_blend_alpha: float = 0.25,
    backbone: str = "resnet18",
    heatmap_min_confidence: float = 0.05,
    enable_warmup: bool = True,
    warmup_frames: int = 30,
    min_warmup_confident_frames: int = 10,
    output_scorer: str = "FuzzyTrack",
    output_bodypart: str = "LED",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Backbone: {backbone}")
    logger.info(f"Smoothing factor: {smoothing}" + (" (no smoothing)" if smoothing >= 1.0 else ""))
    logger.info(f"Peak blend alpha: {peak_blend_alpha}")

    cnn = _load_tracking_model(cnn_model_path, backbone, device)
    cap, out_video, width, height, total_frames = _open_io(video_path, output_video, output_csv)

    ret, first_frame = cap.read()
    if not ret:
        logger.error("Error reading first frame.")
        cap.release()
        out_video.release()
        return

    prev_gray = cv2.resize(first_frame, IMG_SIZE)
    prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)

    warmup = _make_warmup_state(
        enable_warmup=enable_warmup,
        warmup_frames=warmup_frames,
        min_warmup_confident_frames=min_warmup_confident_frames,
        heatmap_min_confidence=heatmap_min_confidence,
    )

    results = []
    frame_idx = 0
    prev_refined_coords = None
    logger.info("Starting tracking...")
    pbar = tqdm(total=total_frames, desc="Tracking", unit="frame")

    frame = first_frame
    while True:
        curr_small = cv2.resize(frame, IMG_SIZE)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, prev_gray)
        input_img = diff.astype(np.float32) / 255.0
        input_tensor = torch.tensor(input_img).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            model_out = cnn(input_tensor)

        raw_coords, heatmap_conf, heatmap = _extract_model_coords(
            model_out, frame_idx, warmup, peak_blend_alpha
        )
        raw_coords = _refine_coords(raw_coords, prev_refined_coords, smoothing, max_speed)
        prev_refined_coords = raw_coords.copy()

        norm_x, norm_y = raw_coords
        pixel_x = int(np.clip(norm_x * (width - 1), 0, width - 1)) if width > 0 else 0
        pixel_y = int(np.clip(norm_y * (height - 1), 0, height - 1)) if height > 0 else 0
        results.append(_build_result_row(pixel_x, pixel_y, heatmap_conf))

        cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        heatmap_vis = _prepare_heatmap_vis(heatmap, width, height)
        heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_INFERNO)
        heatmap_blurred = cv2.GaussianBlur(heatmap_vis, (0, 0), sigmaX=6, sigmaY=6)
        heatmap_blurred_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_INFERNO)
        heatmap_colored = _annotate_heatmap_panel(
            heatmap_colored, "Heatmap", pixel_x, pixel_y, heatmap_conf
        )
        heatmap_blurred_colored = _annotate_heatmap_panel(
            heatmap_blurred_colored, "Heatmap (Blurred)", pixel_x, pixel_y, heatmap_conf
        )
        out_video.write(np.hstack([frame, heatmap_colored, heatmap_blurred_colored]))

        pbar.update(1)
        prev_gray = curr_gray
        frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            break

    pbar.close()
    cap.release()
    out_video.release()
    _write_tracking_csv(results, output_csv, output_scorer, output_bodypart)
    logger.info("Tracking complete!")
    logger.info(f"  - Video saved to: {output_video}")
    logger.info(f"  - CSV saved to: {output_csv}")
    logger.info(f"  - Total frames processed: {frame_idx}")
