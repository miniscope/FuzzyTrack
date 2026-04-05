"""Configuration constants and loading."""

import os

import yaml

# Image processing
IMG_SIZE = (224, 224)  # Standard ResNet size

# Heatmap settings
HEATMAP_SIZE = (56, 56)  # Downsampled heatmap (224/4 = 56)
HEATMAP_SIGMA = 3.0  # Gaussian sigma for heatmap generation (default, can be overridden by config)

# Default configuration values
DEFAULT_CONFIG = {
    "model": {
        "backbone": "resnet18",
    },
    "heatmap": {
        "sigma": HEATMAP_SIGMA,
    },
    "training": {
        "batch_size": 16,
        "epochs": 200,
        "patience": 20,
        "learning_rate": 5e-4,
        "val_split": 0.2,
    },
    "tracking": {
        "smoothing": 0.5,
        "enable_warmup": True,
        "warmup_frames": 30,
        "min_warmup_confident_frames": 10,
        "output_scorer": "FuzzyTrack",
        "output_bodypart": "LED",
        "heatmap_min_confidence": 0.05,
        "max_speed": None,
    },
}


def _deep_merge_dict(base: dict, override: dict) -> dict:
    for key, val in override.items():
        if isinstance(base.get(key), dict) and isinstance(val, dict):
            _deep_merge_dict(base[key], val)
        else:
            base[key] = val
    return base


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file, recursively merged with defaults."""
    config = {
        section: values.copy() if isinstance(values, dict) else values
        for section, values in DEFAULT_CONFIG.items()
    }

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}

        _deep_merge_dict(config, user_config)

    return config
