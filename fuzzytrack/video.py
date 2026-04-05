"""Video metadata helpers and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: int
    total_frames: int


def get_video_info(video_path: str, require_square: bool = True) -> VideoInfo:
    """Open a video and return validated metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video dimensions from: {video_path}")
    if total_frames <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    if require_square and width != height:
        raise NotImplementedError(
            f"Non-square videos are not supported yet: {video_path} ({width}x{height})."
        )

    return VideoInfo(width=width, height=height, fps=fps, total_frames=total_frames)


def mask_video(video_path: str, mask_path: str, output_path: str) -> None:
    """Apply a binary PNG mask to a video and write a masked copy."""
    video_info = get_video_info(video_path, require_square=False)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask image: {mask_path}")
    if mask.shape != (video_info.height, video_info.width):
        raise ValueError(
            f"Mask dimensions {mask.shape[1]}x{mask.shape[0]} do not match video "
            f"{video_info.width}x{video_info.height}: {mask_path}"
        )

    binary_mask = (mask > 0).astype(np.uint8)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    pbar = tqdm(total=video_info.total_frames, desc="Masking", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        masked = frame * binary_mask[:, :, None]
        writer.write(masked.astype(np.uint8))
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
