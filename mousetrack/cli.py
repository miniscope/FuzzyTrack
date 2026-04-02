"""Command-line interface for mouse tracking."""
from datetime import datetime
from pathlib import Path

import click

from .annotation import annotate_video
from .config import load_config
from .tracking import track_video
from .training import train_cnn as train_cnn_func


@click.group()
def main():
    """Mouse tracking CLI."""
    pass


@main.command()
@click.option('--video', '-v', multiple=True, type=click.Path(exists=True), help='Video file(s) - can specify multiple')
@click.option('--annotations', '-a', multiple=True, type=click.Path(exists=True), help='Annotations CSV(s) - must match videos')
@click.option('--data-root', '-d', type=click.Path(exists=True), help='Root directory containing subdirs, each with a .mp4 and .csv file')
@click.option('--config', '-c', required=True, type=click.Path(exists=True), help='Config YAML file')
@click.option('--output', '-o', default=None, type=click.Path(), help='Output model path')
def train_cnn(video, annotations, data_root, config, output):
    """Train CNN model."""
    if data_root:
        if video or annotations:
            raise click.BadParameter("Cannot use --data-root together with --video or --annotations")

        data_root_path = Path(data_root)
        video_list = []
        annotations_list = []

        for subdir in sorted(data_root_path.iterdir()):
            if not subdir.is_dir():
                continue

            video_files = list(subdir.glob('*.mp4'))
            csv_files = list(subdir.glob('*.csv'))

            if video_files and csv_files:
                video_list.append(str(video_files[0]))
                annotations_list.append(str(csv_files[0]))
                click.echo(f"Found: {subdir.name}")
            else:
                missing = []
                if not video_files:
                    missing.append('*.mp4')
                if not csv_files:
                    missing.append('*.csv')
                click.echo(f"Warning: Skipping {subdir.name} - missing {', '.join(missing)}", err=True)

        if not video_list:
            raise click.BadParameter(
                f"No valid subdirectories found in {data_root}. Each subdirectory should contain a .mp4 and .csv file"
            )

        video = tuple(video_list)
        annotations = tuple(annotations_list)
        click.echo(f"Loaded {len(video)} video/annotation pairs from {data_root}")
    else:
        if not video or not annotations:
            raise click.BadParameter("Must provide either --data-root OR both --video and --annotations")
        if len(video) != len(annotations):
            raise click.BadParameter(
                f"Number of videos ({len(video)}) must match number of annotations ({len(annotations)})"
            )

    cfg = load_config(config)

    use_heatmap = cfg['model']['use_heatmap']
    backbone = cfg['model']['backbone']
    batch_size = cfg['training']['batch_size']
    epochs = cfg['training']['epochs']
    patience = cfg['training']['patience']
    learning_rate = cfg['training']['learning_rate']
    val_split = cfg['training']['val_split']
    heatmap_sigma = cfg['heatmap']['sigma']
    num_workers = cfg['training'].get('num_workers', 4)
    pin_memory = cfg['training'].get('pin_memory', True)
    cache_frames = cfg['training'].get('cache_frames', True)

    if output is None:
        output = 'models/mouse_cnn_heatmap.pth' if use_heatmap else 'models/mouse_cnn_regression.pth'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_type = 'heatmap' if use_heatmap else 'regression'
    logdir = f'runs/mouse_tracker_{model_type}_{timestamp}'

    train_cnn_func(
        video_paths=list(video),
        annotations_paths=list(annotations),
        output_path=output,
        use_heatmap=use_heatmap,
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
@click.option('--video', '-v', required=True, type=click.Path(exists=True), help='Video file')
@click.option('--config', '-c', required=True, type=click.Path(exists=True), help='Config YAML file')
@click.option('--cnn-model', '-m', default=None, type=click.Path(exists=True), help='CNN model path')
@click.option('--output', '-o', default=None, type=click.Path(), help='Output path (base name for .mp4 and .csv, or directory)')
def track(video, config, cnn_model, output):
    """Run tracking and export raw DLC coordinates."""
    import os

    cfg = load_config(config)

    use_heatmap = cfg['model']['use_heatmap']
    backbone = cfg['model']['backbone']
    smoothing = cfg['tracking']['smoothing']
    enable_warmup = cfg['tracking'].get('enable_warmup', True)
    warmup_frames_heatmap = cfg['tracking'].get('warmup_frames_heatmap', 30)
    warmup_frames_regression = cfg['tracking'].get('warmup_frames_regression', 5)
    min_warmup_confident_frames_heatmap = cfg['tracking'].get('min_warmup_confident_frames_heatmap', 10)
    min_warmup_confident_frames_regression = cfg['tracking'].get('min_warmup_confident_frames_regression', 3)
    output_scorer = cfg['tracking'].get('output_scorer', '3DMazeTrack')
    output_bodypart = cfg['tracking'].get('output_bodypart', 'LED')
    max_speed = cfg['tracking'].get('max_speed')
    heatmap_min_confidence = cfg['tracking'].get('heatmap_min_confidence', 0.05)

    model_type = 'heatmap' if use_heatmap else 'regression'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if cnn_model is None:
        cnn_model = f'models/mouse_cnn_{model_type}.pth'

    if output is not None:
        if os.path.isdir(output):
            base_name = f'tracking_{model_type}_{timestamp}'
            output_video = os.path.join(output, f'{base_name}.mp4')
            output_csv = os.path.join(output, f'{base_name}.csv')
        else:
            base_path = output.rsplit('.', 1)[0] if '.' in os.path.basename(output) else output
            output_video = f'{base_path}.mp4'
            output_csv = f'{base_path}.csv'
    else:
        output_video = f'output/tracking_{model_type}_{timestamp}.mp4'
        output_csv = f'output/tracking_{model_type}_{timestamp}.csv'

    track_video(
        video_path=video,
        cnn_model_path=cnn_model,
        output_video=output_video,
        output_csv=output_csv,
        max_speed=max_speed,
        smoothing=smoothing,
        backbone=backbone,
        use_heatmap=use_heatmap,
        heatmap_min_confidence=heatmap_min_confidence,
        enable_warmup=enable_warmup,
        warmup_frames_heatmap=warmup_frames_heatmap,
        warmup_frames_regression=warmup_frames_regression,
        min_warmup_confident_frames_heatmap=min_warmup_confident_frames_heatmap,
        min_warmup_confident_frames_regression=min_warmup_confident_frames_regression,
        output_scorer=output_scorer,
        output_bodypart=output_bodypart,
    )


@main.command()
@click.option('--video', '-v', required=True, type=click.Path(exists=True), help='Video file')
@click.option('--input-csv', '-i', default=None, type=click.Path(exists=True), help='Existing CSV to append to (default: auto-discover from video path)')
@click.option('--num-samples', '-n', default=300, type=int, help='Number of frames to annotate')
def annotate(video, input_csv, num_samples):
    """Annotate video frames (coordinates only)."""
    video_path = Path(video)
    csv_path = video_path.with_suffix('.csv')

    if input_csv is None and csv_path.exists():
        input_csv = str(csv_path)
        click.echo(f"Found existing annotations: {input_csv}")

    annotate_video(
        video_path=video,
        output_csv=str(csv_path),
        num_samples=num_samples,
        input_csv=input_csv,
    )
