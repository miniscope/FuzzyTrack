# 3D Maze Track

Mouse tracking package for 3D maze videos.

## Installation

```bash
uv sync
```

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
# Single video (heatmap mode - default)
mousetrack train-cnn --video assets/video.mp4 --annotations assets/annotations.csv

# Multiple videos
mousetrack train-cnn \
  -v assets/video1.mp4 -a assets/annotations1.csv \
  -v assets/video2.mp4 -a assets/annotations2.csv

# Regression mode (instead of heatmap)
mousetrack train-cnn --video assets/video.mp4 --annotations assets/annotations.csv --no-heatmap
```
**Generates:**
- `models/mouse_cnn_heatmap.pth` or `models/mouse_cnn_regression.pth` (trained model)
- `runs/mouse_tracker_{heatmap|regression}_{timestamp}/` (TensorBoard logs)

### 4. Run Tracking
```bash
# Heatmap mode (default)
mousetrack track --video assets/video.mp4

# Regression mode
mousetrack track --video assets/video.mp4 --no-heatmap

# Adjust smoothing (default 0.5, lower=smoother, 1.0=no smoothing)
mousetrack track --video assets/video.mp4 --smoothing 0.3

# Custom paths
mousetrack track \
  --video assets/video.mp4 \
  --cnn-model models/mouse_cnn_heatmap.pth \
  --output-csv output/tracking_heatmap.csv \
  --output-video output/tracking_heatmap.mp4
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
