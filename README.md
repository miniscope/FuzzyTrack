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
Creates `config/zone_polygons.yaml` with zone boundaries.

**Note**: You also need `config/zone_graph.yaml` with zone connections. Create it manually or copy from an example.

### 2. Annotate Video
```bash
mousetrack annotate --video assets/video.mp4 --output assets/annotations.csv
```
Click mouse positions and select zones for training data.

### 3. Train CNN (coordinate prediction)
```bash
mousetrack train-cnn --video assets/video.mp4 --annotations assets/annotations.csv --output models/mouse_cnn.pth
```

### 4. Run Tracking
```bash
mousetrack track \
  --video assets/video.mp4 \
  --cnn-model models/mouse_cnn.pth \
  --output-csv output/tracking_results.csv \
  --output-video output/tracking_results.mp4
```

## Pipeline Summary

```
Video → CNN → Geometric Classification → Zones
```

- **CNN**: Finds mouse coordinates from frames (supports direct regression or heatmap)
- **Geometric Classification**: Classifies zones using proximity to polygons/polylines
- **Hysteresis**: Prevents rapid zone switching for stability
