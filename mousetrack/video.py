"""Video metadata helpers and validation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2


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
