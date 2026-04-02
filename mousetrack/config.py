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
    'model': {
        'backbone': 'resnet18',
        'use_heatmap': True,
    },
    'heatmap': {
        'sigma': HEATMAP_SIGMA,
    },
    'training': {
        'batch_size': 16,
        'epochs': 200,
        'patience': 20,
        'learning_rate': 5e-4,
        'val_split': 0.2,
    },
    'tracking': {
        'smoothing': 0.5,
        'min_confidence': 0.5,
        'min_confidence_forbidden': 0.8,
        'min_frames_same': 1,
        'min_frames_forbidden': 3,
        'enable_warmup': True,
        'warmup_frames_heatmap': 30,
        'warmup_frames_regression': 5,
        'min_warmup_confident_frames_heatmap': 10,
        'min_warmup_confident_frames_regression': 3,
        'max_speed': None,
        'enable_zones': False,
    },
}


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file, merged with defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}

        # Deep merge user config into defaults
        for section, values in user_config.items():
            if section in config and isinstance(values, dict):
                config[section] = {**config[section], **values}
            else:
                config[section] = values

    return config
