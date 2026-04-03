"""Checkpoint save/load helpers."""

from __future__ import annotations

from typing import Any

import torch


def save_checkpoint(
    output_path: str,
    state_dict: dict[str, torch.Tensor],
    *,
    backbone: str,
    use_heatmap: bool,
    heatmap_sigma: float | None,
    img_size: tuple[int, int],
    video_size: tuple[int, int],
) -> None:
    """Save model weights together with metadata needed for compatibility checks."""
    checkpoint = {
        "state_dict": state_dict,
        "metadata": {
            "backbone": backbone,
            "use_heatmap": use_heatmap,
            "heatmap_sigma": heatmap_sigma,
            "img_size": img_size,
            "video_size": video_size,
        },
    }
    torch.save(checkpoint, output_path)


def load_checkpoint(
    checkpoint_path: str,
    *,
    device: torch.device,
    expected_backbone: str | None = None,
    expected_use_heatmap: bool | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load checkpoint weights and validate stored metadata when available."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata: dict[str, Any] = {}

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = checkpoint.get("metadata", {})
    else:
        state_dict = checkpoint

    saved_backbone = metadata.get("backbone")
    if expected_backbone is not None and saved_backbone and saved_backbone != expected_backbone:
        raise RuntimeError(
            f"Checkpoint backbone mismatch: model uses '{saved_backbone}', "
            f"config requests '{expected_backbone}'."
        )

    saved_use_heatmap = metadata.get("use_heatmap")
    if expected_use_heatmap is not None and saved_use_heatmap is not None:
        if saved_use_heatmap != expected_use_heatmap:
            saved_mode = "heatmap" if saved_use_heatmap else "regression"
            requested_mode = "heatmap" if expected_use_heatmap else "regression"
            raise RuntimeError(
                f"Checkpoint mode mismatch: model was trained for {saved_mode}, "
                f"config requests {requested_mode}."
            )

    return state_dict, metadata
