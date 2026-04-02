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
  backbone: resnet50  # resnet18 or resnet50
  use_heatmap: true   # true for heatmap mode, false for regression

heatmap:
  sigma: 3.0  # Gaussian sigma for target heatmap

training:
  batch_size: 8
  epochs: 1000
  patience: 50
  learning_rate: 0.0005
  val_split: 0.2
  num_workers: 8
  pin_memory: true
  cache_frames: true

tracking:
  smoothing: 0.2          # EMA smoothing (0.0-1.0, lower=smoother, 1.0=no smoothing)
  min_confidence: 0.4     # Min confidence for valid zone transitions
  min_confidence_forbidden: 0.9
  min_frames_same: 1
  min_frames_forbidden: 10
  heatmap_min_confidence: 0.4
  # max_speed: 0.1        # Uncomment to limit movement speed
```

The `--config` option is required for training and tracking. Most training and tracking parameters are currently read from config rather than exposed as CLI flags.

## Workflow

### 1. Define Zones (one-time setup)
```bash
mousetrack define-zones --video assets/video.mp4
```
**Generates:** `config/zone_polygons.yaml` (zone boundaries)

**Note**: You also need `config/zone_graph.yaml` with zone connections. Create it manually or copy from an example.

### 2. Annotate Video
```bash
mousetrack annotate --video assets/video.mp4
```
Annotates sampled frames with mouse coordinates. Output is written automatically to a CSV with the same stem as the video.

**Generates:** `assets/video.csv` (training annotations)

### 3. Train CNN (coordinate prediction)
```bash
# Single video
mousetrack train-cnn -c config/model_config.yaml -v assets/video.mp4 -a assets/video.csv

# Multiple videos
mousetrack train-cnn -c config/model_config.yaml \
  -v assets/video1.mp4 -a assets/video1.csv \
  -v assets/video2.mp4 -a assets/video2.csv

# Dataset directory mode: each subdirectory contains one .mp4 and one .csv
mousetrack train-cnn -c config/model_config.yaml --data-root assets/dataset
```
**Generates:**
- `models/mouse_cnn_heatmap.pth` or `models/mouse_cnn_regression.pth` (trained model)
- `runs/mouse_tracker_{heatmap|regression}_{timestamp}/` (TensorBoard logs)

### 4. Run Tracking
```bash
# Basic tracking
mousetrack track -c config/model_config.yaml -v assets/video.mp4

# Custom model and output base path
mousetrack track -c config/model_config.yaml \
  -v assets/video.mp4 \
  --cnn-model models/mouse_cnn_heatmap.pth \
  --output output/tracking

# Enable zone classification during tracking
mousetrack track -c config/model_config.yaml \
  -v assets/video.mp4 \
  --enable-zones
```
**Generates:**
- `output/tracking_{heatmap|regression}_{timestamp}.csv` (tracking data in DLC format)
- `output/tracking_{heatmap|regression}_{timestamp}.mp4` (annotated video)

When zone classification is enabled, the tracking CSV uses a DLC-style 3-level header and includes:
- `x`, `y`, `likelihood`
- `x_pinned`, `y_pinned`
- `zone`
- `arm_position`

### 5. Detect Zones from Existing Tracking CSV
```bash
mousetrack detect-zones \
  --input-csv output/tracking_heatmap_20260402_120000.csv \
  --output-csv output/tracking_with_zones.csv \
  --zone-polygons config/zone_polygons.yaml \
  --zone-graph config/zone_graph.yaml \
  --config config/model_config.yaml
```
Adds zone labels to an existing tracking CSV using the configured hysteresis thresholds.
The output remains DLC-style and adds `x_pinned`, `y_pinned`, `zone`, and `arm_position`.

## Pipeline Summary

```
Video → CNN → EMA Smoothing → Geometric Classification → Zones
```

- **CNN**: Finds mouse coordinates from frames (supports direct regression or heatmap)
- **EMA Smoothing**: Exponential moving average reduces jitter from noise/scattering
- **Geometric Classification**: Classifies zones using proximity to polygons/polylines
- **Hysteresis**: Prevents rapid zone switching for stability
