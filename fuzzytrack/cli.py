"""Command-line interface for FuzzyTrack."""

from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from .annotation import annotate_video
from .config import load_config
from .tracking import track_video
from .training import train_cnn as train_cnn_func
from .video import get_video_info

LABEL_SUFFIX = "_labels.csv"


def _raise_click_error(func, /, *args, **kwargs):
    """Run a command handler and normalize common failures for CLI output."""
    try:
        return func(*args, **kwargs)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.group()
def main():
    """FuzzyTrack CLI."""
    pass


def _default_labels_path(video_path: Path) -> Path:
    """Return the default annotation CSV path for a video."""
    return video_path.with_name(f"{video_path.stem}{LABEL_SUFFIX}")


def _resolve_video_label_pairs(
    video: tuple[str, ...], annotations: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve explicit or auto-discovered video/label pairs."""
    if not video:
        raise click.BadParameter("Must provide at least one --video")
    if annotations and len(video) != len(annotations):
        raise click.BadParameter(
            f"Number of videos ({len(video)}) must match number of annotations ({len(annotations)})"
        )
    if annotations:
        return video, annotations

    discovered_annotations = []
    missing_annotations = []
    for video_path_str in video:
        video_path = Path(video_path_str)
        labels_path = _default_labels_path(video_path)
        if labels_path.exists():
            discovered_annotations.append(str(labels_path))
        else:
            missing_annotations.append(str(labels_path))

    if missing_annotations:
        missing = ", ".join(missing_annotations)
        raise click.BadParameter(
            "No --annotations were provided and the following label files were not found: "
            f"{missing}"
        )

    annotations = tuple(discovered_annotations)
    click.echo(f"Auto-discovered {len(annotations)} label file(s)")
    return video, annotations


def _print_coverage_report(video_path: str, annotations_path: str, grid_size: int) -> None:
    """Print a spatial coverage report for one video/labels pair."""
    video_info = get_video_info(video_path, require_square=True)
    frame_width = video_info.width
    frame_height = video_info.height
    df = pd.read_csv(annotations_path)

    required_columns = {"frame_idx", "x", "y"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Annotation CSV is missing required column(s): {missing} ({annotations_path})"
        )
    if df.empty:
        raise ValueError(f"Annotation CSV is empty: {annotations_path}")

    grid_counts: dict[tuple[int, int], int] = {}
    for row in df.loc[:, ["x", "y"]].itertuples(index=False):
        grid_x = min(int(row.x / frame_width * grid_size), grid_size - 1)
        grid_y = min(int(row.y / frame_height * grid_size), grid_size - 1)
        grid_counts[(grid_x, grid_y)] = grid_counts.get((grid_x, grid_y), 0) + 1

    total_cells = grid_size * grid_size
    covered_cells = len(grid_counts)
    click.echo(f"\n{Path(video_path).name}")
    click.echo(f"  Labels: {Path(annotations_path).name}")
    click.echo(f"  Samples: {len(df)}")
    click.echo(
        f"  Covered cells: {covered_cells}/{total_cells} ({100 * covered_cells / total_cells:.1f}%)"
    )
    click.echo("  Grid counts:")
    for gy in range(grid_size):
        row_counts = " ".join(f"{grid_counts.get((gx, gy), 0):3d}" for gx in range(grid_size))
        click.echo(f"    Row {gy}: {row_counts}")

    min_samples = 3
    undersampled = [
        (gx, gy)
        for gy in range(grid_size)
        for gx in range(grid_size)
        if grid_counts.get((gx, gy), 0) < min_samples
    ]
    if undersampled:
        click.echo(f"  Cells with < {min_samples} samples: {len(undersampled)}")
        for gx, gy in undersampled[:5]:
            x_range = (
                f"{int(gx * frame_width / grid_size)}-{int((gx + 1) * frame_width / grid_size)}"
            )
            y_range = (
                f"{int(gy * frame_height / grid_size)}-{int((gy + 1) * frame_height / grid_size)}"
            )
            click.echo(
                f"    Cell [{gx},{gy}]: x={x_range}, y={y_range} ({grid_counts.get((gx, gy), 0)} samples)"
            )


@main.command()
@click.option(
    "--video",
    "-v",
    multiple=True,
    type=click.Path(exists=True),
    help="Video file(s) - can specify multiple",
)
@click.option(
    "--annotations",
    "-a",
    multiple=True,
    type=click.Path(exists=True),
    help="Annotations CSV(s) - must match videos",
)
@click.option(
    "--data-root",
    "-d",
    type=click.Path(exists=True),
    help="Root directory containing subdirs, each with a .mp4 and a *_labels.csv file",
)
@click.option(
    "--config", "-c", required=True, type=click.Path(exists=True), help="Config YAML file"
)
@click.option("--output", "-o", default=None, type=click.Path(), help="Output model path")
def train_cnn(video, annotations, data_root, config, output):
    """Train CNN model."""
    if data_root:
        if video or annotations:
            raise click.BadParameter(
                "Cannot use --data-root together with --video or --annotations"
            )

        data_root_path = Path(data_root)
        video_list = []
        annotations_list = []

        for subdir in sorted(data_root_path.iterdir()):
            if not subdir.is_dir():
                continue

            video_files = sorted(subdir.glob("*.mp4"))
            label_files = sorted(subdir.glob(f"*{LABEL_SUFFIX}"))

            if video_files and label_files:
                video_list.append(str(video_files[0]))
                annotations_list.append(str(label_files[0]))
                click.echo(f"Found: {subdir.name}")
            else:
                missing = []
                if not video_files:
                    missing.append("*.mp4")
                if not label_files:
                    missing.append(f"*{LABEL_SUFFIX}")
                click.echo(
                    f"Warning: Skipping {subdir.name} - missing {', '.join(missing)}", err=True
                )

        if not video_list:
            raise click.BadParameter(
                f"No valid subdirectories found in {data_root}. Each subdirectory should contain a .mp4 and a *{LABEL_SUFFIX} file"
            )

        video = tuple(video_list)
        annotations = tuple(annotations_list)
        click.echo(f"Loaded {len(video)} video/annotation pairs from {data_root}")
    else:
        video, annotations = _resolve_video_label_pairs(video, annotations)

    cfg = load_config(config)

    backbone = cfg["model"]["backbone"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]
    learning_rate = cfg["training"]["learning_rate"]
    val_split = cfg["training"]["val_split"]
    heatmap_sigma = cfg["heatmap"]["sigma"]
    peak_blend_alpha = cfg["tracking"].get("peak_blend_alpha", 0.25)
    num_workers = cfg["training"].get("num_workers", 4)
    pin_memory = cfg["training"].get("pin_memory", True)
    cache_frames = cfg["training"].get("cache_frames", True)

    if output is None:
        output = "models/mouse_cnn_heatmap.pth"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f"runs/fuzzytrack_heatmap_{timestamp}"

    _raise_click_error(
        train_cnn_func,
        video_paths=list(video),
        annotations_paths=list(annotations),
        output_path=output,
        batch_size=batch_size,
        max_epochs=epochs,
        patience=patience,
        val_split=val_split,
        learning_rate=learning_rate,
        logdir=logdir,
        backbone=backbone,
        heatmap_sigma=heatmap_sigma,
        peak_blend_alpha=peak_blend_alpha,
        num_workers=num_workers,
        pin_memory=pin_memory,
        cache_frames=cache_frames,
    )


@main.command()
@click.option("--video", "-v", required=True, type=click.Path(exists=True), help="Video file")
@click.option(
    "--config", "-c", required=True, type=click.Path(exists=True), help="Config YAML file"
)
@click.option(
    "--cnn-model", "-m", default=None, type=click.Path(exists=True), help="CNN model path"
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output path (base name for .mp4 and .csv, or directory)",
)
def track(video, config, cnn_model, output):
    """Run tracking and export raw DLC coordinates."""
    import os

    cfg = load_config(config)
    video_stem = Path(video).stem

    backbone = cfg["model"]["backbone"]
    smoothing = cfg["tracking"]["smoothing"]
    peak_blend_alpha = cfg["tracking"].get("peak_blend_alpha", 0.25)
    enable_warmup = cfg["tracking"].get("enable_warmup", True)
    warmup_frames = cfg["tracking"].get("warmup_frames", 30)
    min_warmup_confident_frames = cfg["tracking"].get("min_warmup_confident_frames", 10)
    output_scorer = cfg["tracking"].get("output_scorer", "FuzzyTrack")
    output_bodypart = cfg["tracking"].get("output_bodypart", "LED")
    max_speed = cfg["tracking"].get("max_speed")
    heatmap_min_confidence = cfg["tracking"].get("heatmap_min_confidence", 0.05)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if cnn_model is None:
        cnn_model = "models/mouse_cnn_heatmap.pth"

    if output is not None:
        output_path = Path(output)
        looks_like_directory = output.endswith(os.sep) or output_path.suffix == ""
        if os.path.isdir(output) or looks_like_directory:
            base_name = f"{video_stem}_tracking_heatmap_{timestamp}"
            output_video = os.path.join(output, f"{base_name}.mp4")
            output_csv = os.path.join(output, f"{base_name}.csv")
        else:
            base_path = output.rsplit(".", 1)[0] if "." in os.path.basename(output) else output
            output_video = f"{base_path}.mp4"
            output_csv = f"{base_path}.csv"
    else:
        output_video = f"output/{video_stem}_tracking_heatmap_{timestamp}.mp4"
        output_csv = f"output/{video_stem}_tracking_heatmap_{timestamp}.csv"

    _raise_click_error(
        track_video,
        video_path=video,
        cnn_model_path=cnn_model,
        output_video=output_video,
        output_csv=output_csv,
        max_speed=max_speed,
        smoothing=smoothing,
        peak_blend_alpha=peak_blend_alpha,
        backbone=backbone,
        heatmap_min_confidence=heatmap_min_confidence,
        enable_warmup=enable_warmup,
        warmup_frames=warmup_frames,
        min_warmup_confident_frames=min_warmup_confident_frames,
        output_scorer=output_scorer,
        output_bodypart=output_bodypart,
    )


@main.command()
@click.option("--video", "-v", required=True, type=click.Path(exists=True), help="Video file")
@click.option(
    "--input-csv",
    "-i",
    default=None,
    type=click.Path(exists=True),
    help="Existing CSV to append to (default: auto-discover from video path)",
)
@click.option("--num-samples", "-n", default=300, type=int, help="Number of frames to annotate")
def annotate(video, input_csv, num_samples):
    """Annotate video frames (coordinates only)."""
    video_path = Path(video)
    csv_path = _default_labels_path(video_path)
    legacy_csv_path = video_path.with_suffix(".csv")

    if input_csv is None:
        if csv_path.exists():
            input_csv = str(csv_path)
            click.echo(f"Found existing annotations: {input_csv}")
        elif legacy_csv_path.exists():
            input_csv = str(legacy_csv_path)
            click.echo(
                f"Found legacy annotations: {input_csv} (will save merged labels to {csv_path})"
            )

    _raise_click_error(
        annotate_video,
        video_path=video,
        output_csv=str(csv_path),
        num_samples=num_samples,
        input_csv=input_csv,
    )


@main.command("coverage")
@click.option(
    "--video",
    "-v",
    multiple=True,
    type=click.Path(exists=True),
    help="Video file(s) - can specify multiple",
)
@click.option(
    "--annotations",
    "-a",
    multiple=True,
    type=click.Path(exists=True),
    help="Label CSV(s) - defaults to auto-discovered *_labels.csv",
)
@click.option("--grid-size", default=4, type=click.IntRange(min=2), help="Coverage grid size")
def coverage(video, annotations, grid_size):
    """Report annotation spatial coverage for one or more videos."""
    video, annotations = _resolve_video_label_pairs(video, annotations)
    for video_path, annotations_path in zip(video, annotations, strict=True):
        _raise_click_error(
            _print_coverage_report,
            video_path=video_path,
            annotations_path=annotations_path,
            grid_size=grid_size,
        )
