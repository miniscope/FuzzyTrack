# FuzzyTrack

Mouse tracking package for fuzzy videos.

## Installation

```bash
uv sync
```

## Configuration

All parameters can be set in `config/model_config.yaml`:

```yaml
model:
  backbone: resnet18  # resnet18 or resnet50
  use_heatmap: true   # true for heatmap mode, false for regression

heatmap:
  sigma: 3.0  # Gaussian sigma for target (larger = more tolerance for scattered targets)

training:
  batch_size: 16
  epochs: 200
  patience: 20
  learning_rate: 0.0005
  val_split: 0.2

tracking:
  smoothing: 0.5          # EMA smoothing (0.0-1.0, lower=smoother)
  min_confidence: 0.5     # Min confidence for zone transitions
  min_confidence_forbidden: 0.8
  min_frames_same: 1
  # max_speed: 0.1        # Uncomment to limit movement speed
```

CLI options override config values. The `--config` option is required for training and tracking.

## Workflow

### 1. Define Zones (one-time setup)
```bash
mousetrack define-zones --video assets/video.mp4
```
**Generates:** `config/zone_polygons.yaml` (zone boundaries)

**Note**: You also need `config/zone_graph.yaml` with zone connections. Create it manually or copy from an example.

### 2. Annotate Video
```bash
mousetrack annotate --video assets/video.mp4 --output assets/annotations.csv
```
Click mouse positions and select zones for training data.

**Generates:** `assets/annotations.csv` (training annotations)

### 3. Train CNN (coordinate prediction)
```bash
# Single video
mousetrack train-cnn -c config/model_config.yaml -v assets/video.mp4 -a assets/annotations.csv

# Multiple videos
mousetrack train-cnn -c config/model_config.yaml \
  -v assets/video1.mp4 -a assets/annotations1.csv \
  -v assets/video2.mp4 -a assets/annotations2.csv

# Override config with CLI options
mousetrack train-cnn -c config/model_config.yaml -v assets/video.mp4 -a assets/annotations.csv --heatmap-sigma 4.0
```
**Generates:**
- `models/mouse_cnn_heatmap.pth` or `models/mouse_cnn_regression.pth` (trained model)
- `runs/mouse_tracker_{heatmap|regression}_{timestamp}/` (TensorBoard logs)

### 4. Run Tracking
```bash
# Basic tracking
mousetrack track -c config/model_config.yaml -v assets/video.mp4

# Override smoothing (lower=smoother, 1.0=no smoothing)
mousetrack track -c config/model_config.yaml -v assets/video.mp4 --smoothing 0.3

# Custom model and output paths
mousetrack track -c config/model_config.yaml \
  -v assets/video.mp4 \
  --cnn-model models/mouse_cnn_heatmap.pth \
  --output-csv output/tracking.csv \
  --output-video output/tracking.mp4
```
**Generates:**
- `output/tracking_{heatmap|regression}_{timestamp}.csv` (tracking data in DLC format)
- `output/tracking_{heatmap|regression}_{timestamp}.mp4` (annotated video)

## Pipeline Summary

```
Video → CNN → EMA Smoothing → Geometric Classification → Zones
```

- **CNN**: Finds mouse coordinates from frames (supports direct regression or heatmap)
- **EMA Smoothing**: Exponential moving average reduces jitter from noise/scattering
- **Geometric Classification**: Classifies zones using proximity to polygons/polylines
- **Hysteresis**: Prevents rapid zone switching for stability
