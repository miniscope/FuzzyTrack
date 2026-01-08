"""Configuration constants."""

# Image processing
IMG_SIZE = (224, 224)  # Standard ResNet size

# Heatmap settings
HEATMAP_SIZE = (56, 56)  # Downsampled heatmap (224/4 = 56)
HEATMAP_SIGMA = 2.0  # Gaussian blur sigma for heatmap generation
