# FuzzyTrack

Tracking package for fuzzy videos.

## Installation

```bash
uv sync
```

Requires Python 3.11 or newer.

For development tools:

```bash
uv sync --group dev
make lint
make format
```

## Configuration

Runtime configs:

- `config/model_config_cuda.yaml` for CUDA training/inference
- `config/model_config_mac.yaml` for local macOS use

All parameters can be set in either file:

```yaml
model:
  backbone: resnet50  # resnet18 or resnet50

heatmap:
  sigma: 3.0  # Gaussian sigma in heatmap pixels on the 56x56 target heatmap

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
  warmup_frames: 30
  min_warmup_confident_frames: 10
  output_scorer: FuzzyTrack
  output_bodypart: LED
  heatmap_min_confidence: 0.4
  # max_speed: 0.1        # Uncomment to limit movement speed
```

Current limitation: only square videos are supported. Non-square inputs fail fast with a `NotImplementedError`.

The `--config` option is required for training and tracking. Most training and tracking parameters are currently read from config rather than exposed as CLI flags.

## Workflow

FuzzyTrack is the canonical source for raw DLC-style tracking output (`x`, `y`, `likelihood`).
Maze zone detection and 1D serialization are intentionally handled in `placecell`.

### 1. Annotate Video
```bash
fuzzytrack annotate --video assets/video.mp4
```
Annotates sampled frames with mouse coordinates. Output is written automatically to a labels CSV with the same stem as the video.

**Generates:** `assets/video_labels.csv` (training annotations)

### 2. Train CNN (coordinate prediction)
```bash
# Single video (auto-discovers assets/video_labels.csv)
fuzzytrack train-cnn -c config/model_config_cuda.yaml -v assets/video.mp4

# Multiple videos (auto-discovers matching *_labels.csv files)
fuzzytrack train-cnn -c config/model_config_cuda.yaml \
  -v assets/video1.mp4 \
  -v assets/video2.mp4

# Explicit labels also work
fuzzytrack train-cnn -c config/model_config_cuda.yaml \
  -v assets/video1.mp4 -a assets/video1_labels.csv \
  -v assets/video2.mp4 -a assets/video2_labels.csv

# Dataset directory mode: each subdirectory contains one .mp4 and one *_labels.csv
fuzzytrack train-cnn -c config/model_config_cuda.yaml --data-root assets/dataset
```
**Generates:**
- `models/mouse_cnn_heatmap.pth` (trained model)
- `runs/fuzzytrack_heatmap_{timestamp}/` (TensorBoard logs)

### 3. Run Tracking
```bash
# Basic tracking
fuzzytrack track -c config/model_config_cuda.yaml -v assets/video.mp4

# Custom model and output base path
fuzzytrack track -c config/model_config_cuda.yaml \
  -v assets/video.mp4 \
  --cnn-model models/mouse_cnn_heatmap.pth \
  --output output/tracking
```
**Generates:**
- `output/<video_stem>_tracking_heatmap_{timestamp}.csv` (tracking data in DLC format)
- `output/<video_stem>_tracking_heatmap_{timestamp}.mp4` (annotated video)

Tracking CSV schema:
- `x`, `y`, `likelihood`
- DLC header defaults are `scorer=FuzzyTrack` and `bodypart=LED`
- both header values are configurable via `tracking.output_scorer` and `tracking.output_bodypart`

## Output Contract

- Output CSVs use a DLC-style 3-level header: `scorer`, `bodyparts`, `coords`
- The tracked coordinate columns are always `x`, `y`, and `likelihood`
- The first processed frame is included in the output; because tracking uses frame differences, frame 0 is effectively a zero-motion initialization frame
- `likelihood` is entropy-based heatmap confidence
- The tracking video is a QC artifact; the CSV is the canonical output for downstream analysis
- Model checkpoints now store training metadata and tracking validates backbone compatibility when available

## Placecell Integration

Recommended release workflow:

```bash
# 1. Track in FuzzyTrack
fuzzytrack track -c config/model_config_cuda.yaml -v assets/video.mp4

# 2. In placecell data config:
# behavior_position: output/<video_stem>_tracking_heatmap_YYYYMMDD_HHMMSS.csv
# bodypart: LED
#
# 3. Run zone detection and 1D analysis in placecell
```

This keeps tracking and analysis responsibilities separate and avoids divergence between two independent zone-detection implementations.

## Pipeline Summary

```
Video -> CNN -> EMA Smoothing -> DLC CSV
```

- **CNN**: Finds mouse coordinates from frames via heatmap prediction
- **EMA Smoothing**: Reduces jitter from noise/scattering
- **DLC CSV**: Exports raw coordinates for downstream analysis in `placecell`
