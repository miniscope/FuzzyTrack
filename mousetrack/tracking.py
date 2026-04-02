"""Tracking inference functions."""
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import IMG_SIZE
from .models import MouseCNN, MouseHeatmapCNN
from . import logger


@dataclass
class WarmupState:
    enabled: bool
    warmup_frames: int
    min_confident_frames: int
    heatmap_min_confidence: float
    coords: list[np.ndarray]


def extract_coords_from_heatmap(heatmap: np.ndarray, use_weighted_avg: bool = True) -> tuple[np.ndarray, float]:
    """Extract coordinates from a heatmap output."""
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

    if use_weighted_avg:
        if total > 0:
            y_coords, x_coords = np.ogrid[:h, :w]
            hm_x = np.sum(heatmap * x_coords) / total
            hm_y = np.sum(heatmap * y_coords) / total
            hm_x = np.clip(hm_x, 0.0, w - 1.0)
            hm_y = np.clip(hm_y, 0.0, h - 1.0)
        else:
            hm_x = (w - 1) / 2.0
            hm_y = (h - 1) / 2.0
            confidence = 0.0
    else:
        hm_y, hm_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    norm_x = np.clip(hm_x / (w - 1) if w > 1 else 0.5, 0.0, 1.0)
    norm_y = np.clip(hm_y / (h - 1) if h > 1 else 0.5, 0.0, 1.0)
    return np.array([norm_x, norm_y], dtype=np.float32), float(confidence)


def _load_tracking_model(cnn_model_path: str, backbone: str, use_heatmap: bool, device: torch.device):
    if use_heatmap:
        cnn = MouseHeatmapCNN(backbone=backbone)
        logger.info("Loading heatmap CNN model...")
    else:
        cnn = MouseCNN(backbone=backbone)
        logger.info("Loading coordinate CNN model...")

    cnn.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn.to(device)
    cnn.eval()
    logger.info("CNN model loaded successfully.")
    return cnn


def _open_io(video_path: str, output_video: str, output_csv: str, use_heatmap: bool):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Input video: {width}x{height} @ {fps} fps, {total_frames} frames")

    for path in [output_video, output_csv]:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_width = width * 2 if use_heatmap else width
    out_video = cv2.VideoWriter(output_video, fourcc, fps, (output_width, height))
    if not out_video.isOpened():
        cap.release()
        raise RuntimeError(f"Error creating output video: {output_video}")

    logger.info(f"Output video: {output_video}")
    return cap, out_video, width, height, total_frames


def _make_warmup_state(
    use_heatmap: bool,
    enable_warmup: bool,
    warmup_frames_heatmap: int,
    warmup_frames_regression: int,
    min_warmup_confident_frames_heatmap: int,
    min_warmup_confident_frames_regression: int,
    heatmap_min_confidence: float,
) -> WarmupState:
    warmup_frames = warmup_frames_heatmap if use_heatmap else warmup_frames_regression
    min_confident_frames = (
        min_warmup_confident_frames_heatmap if use_heatmap else min_warmup_confident_frames_regression
    )
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


def _extract_model_coords(
    model_out: torch.Tensor,
    use_heatmap: bool,
    frame_idx: int,
    warmup: WarmupState,
) -> tuple[np.ndarray, float | None, np.ndarray | None]:
    heatmap_conf = None
    heatmap = None
    if use_heatmap:
        heatmap = model_out[0, 0].cpu().numpy()
        raw_coords, heatmap_conf = extract_coords_from_heatmap(heatmap, use_weighted_avg=True)
        if warmup.enabled and frame_idx < warmup.warmup_frames:
            if heatmap_conf >= warmup.heatmap_min_confidence:
                warmup.coords.append(raw_coords.copy())
                raw_coords = np.mean(warmup.coords, axis=0)
            elif len(warmup.coords) >= warmup.min_confident_frames:
                raw_coords = np.mean(warmup.coords, axis=0)
            elif warmup.coords:
                raw_coords = warmup.coords[-1].copy()
    else:
        raw_coords = model_out[0].cpu().numpy()
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


def _build_result_row(pixel_x: int, pixel_y: int, heatmap_conf: float | None) -> dict[str, int | float]:
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
    max_speed: Optional[float] = None,
    smoothing: float = 0.5,
    backbone: str = "resnet18",
    use_heatmap: bool = True,
    heatmap_min_confidence: float = 0.05,
    enable_warmup: bool = True,
    warmup_frames_heatmap: int = 30,
    warmup_frames_regression: int = 5,
    min_warmup_confident_frames_heatmap: int = 10,
    min_warmup_confident_frames_regression: int = 3,
    output_scorer: str = "FuzzyTrack",
    output_bodypart: str = "LED",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Backbone: {backbone}")
    logger.info(f"Smoothing factor: {smoothing}" + (" (no smoothing)" if smoothing >= 1.0 else ""))

    cnn = _load_tracking_model(cnn_model_path, backbone, use_heatmap, device)
    try:
        cap, out_video, width, height, total_frames = _open_io(video_path, output_video, output_csv, use_heatmap)
    except RuntimeError as exc:
        logger.error(str(exc))
        return

    ret, first_frame = cap.read()
    if not ret:
        logger.error("Error reading first frame.")
        cap.release()
        out_video.release()
        return

    prev_gray = cv2.resize(first_frame, IMG_SIZE)
    prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)

    warmup = _make_warmup_state(
        use_heatmap=use_heatmap,
        enable_warmup=enable_warmup,
        warmup_frames_heatmap=warmup_frames_heatmap,
        warmup_frames_regression=warmup_frames_regression,
        min_warmup_confident_frames_heatmap=min_warmup_confident_frames_heatmap,
        min_warmup_confident_frames_regression=min_warmup_confident_frames_regression,
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

        raw_coords, heatmap_conf, heatmap = _extract_model_coords(model_out, use_heatmap, frame_idx, warmup)
        raw_coords = _refine_coords(raw_coords, prev_refined_coords, smoothing, max_speed)
        prev_refined_coords = raw_coords.copy()

        norm_x, norm_y = raw_coords
        pixel_x = int(norm_x * width)
        pixel_y = int(norm_y * height)
        results.append(_build_result_row(pixel_x, pixel_y, heatmap_conf))

        cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"Frame: {frame_idx}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if use_heatmap:
            heatmap_vis = (heatmap / (heatmap.max() + 1e-8) * 255).astype(np.uint8)
            heatmap_vis = cv2.resize(heatmap_vis, (width, height), interpolation=cv2.INTER_NEAREST)
            heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
            cv2.circle(heatmap_colored, (pixel_x, pixel_y), 8, (255, 255, 255), 2)
            cv2.putText(heatmap_colored, "Heatmap", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(
                heatmap_colored,
                f"Conf: {heatmap_conf:.3f}" if heatmap_conf is not None else "Conf: N/A",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            out_video.write(np.hstack([frame, heatmap_colored]))
        else:
            out_video.write(frame)

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
