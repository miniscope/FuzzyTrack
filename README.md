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
  enable_warmup: true
  warmup_frames_heatmap: 30
  warmup_frames_regression: 5
  min_warmup_confident_frames_heatmap: 10
  min_warmup_confident_frames_regression: 3
  output_scorer: 3DMazeTrack
  output_bodypart: LED
  heatmap_min_confidence: 0.4
  # max_speed: 0.1        # Uncomment to limit movement speed
```

The `--config` option is required for training and tracking. Most training and tracking parameters are currently read from config rather than exposed as CLI flags.

## Workflow

FuzzyTrack is the canonical source for raw DLC-style tracking output (`x`, `y`, `likelihood`).
Maze zone detection and 1D serialization are intentionally handled in `placecell`.

### 1. Annotate Video
```bash
mousetrack annotate --video assets/video.mp4
```
Annotates sampled frames with mouse coordinates. Output is written automatically to a CSV with the same stem as the video.

**Generates:** `assets/video.csv` (training annotations)

### 2. Train CNN (coordinate prediction)
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

### 3. Run Tracking
```bash
# Basic tracking
mousetrack track -c config/model_config.yaml -v assets/video.mp4

# Custom model and output base path
mousetrack track -c config/model_config.yaml \
  -v assets/video.mp4 \
  --cnn-model models/mouse_cnn_heatmap.pth \
  --output output/tracking
```
**Generates:**
- `output/tracking_{heatmap|regression}_{timestamp}.csv` (tracking data in DLC format)
- `output/tracking_{heatmap|regression}_{timestamp}.mp4` (annotated video)

Tracking CSV schema:
- `x`, `y`, `likelihood`
- DLC header defaults are `scorer=3DMazeTrack` and `bodypart=LED`
- both header values are configurable via `tracking.output_scorer` and `tracking.output_bodypart`

## Output Contract

- Output CSVs use a DLC-style 3-level header: `scorer`, `bodyparts`, `coords`
- The tracked coordinate columns are always `x`, `y`, and `likelihood`
- The first processed frame is included in the output; because tracking uses frame differences, frame 0 is effectively a zero-motion initialization frame
- Heatmap models report an entropy-based `likelihood`; regression models currently write `0.0` in that column
- The tracking video is a QC artifact; the CSV is the canonical output for downstream analysis

## Placecell Integration

Recommended release workflow:

```bash
# 1. Track in FuzzyTrack
mousetrack track -c config/model_config.yaml -v assets/video.mp4

# 2. In placecell data config:
# behavior_position: output/tracking_heatmap_YYYYMMDD_HHMMSS.csv
# bodypart: LED
#
# 3. Run zone detection and 1D analysis in placecell
```

This keeps tracking and analysis responsibilities separate and avoids divergence between two independent zone-detection implementations.

## Pipeline Summary

```
Video -> CNN -> EMA Smoothing -> DLC CSV
```

- **CNN**: Finds mouse coordinates from frames (supports direct regression or heatmap)
- **EMA Smoothing**: Reduces jitter from noise/scattering
- **DLC CSV**: Exports raw coordinates for downstream analysis in `placecell`
