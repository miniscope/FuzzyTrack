"""Command-line interface for FuzzyTrack."""

from datetime import datetime
from pathlib import Path

import click

from .annotation import annotate_video
from .config import load_config
from .tracking import track_video
from .training import train_cnn as train_cnn_func

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
        if not video:
            raise click.BadParameter(
                "Must provide either --data-root OR at least one --video"
            )
        if annotations and len(video) != len(annotations):
            raise click.BadParameter(
                f"Number of videos ({len(video)}) must match number of annotations ({len(annotations)})"
            )
        if not annotations:
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

    cfg = load_config(config)

    backbone = cfg["model"]["backbone"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]
    learning_rate = cfg["training"]["learning_rate"]
    val_split = cfg["training"]["val_split"]
    heatmap_sigma = cfg["heatmap"]["sigma"]
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
