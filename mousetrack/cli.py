"""Command-line interface for mouse tracking."""
from datetime import datetime
import click

from .training import train_cnn as train_cnn_func
from .tracking import track_video
from .annotation import annotate_video
from .zones import define_zones
from .config import load_config


@click.group()
def main():
    """Mouse tracking CLI."""
    pass


@main.command()
@click.option('--video', '-v', required=True, multiple=True, type=click.Path(exists=True), help='Video file(s) - can specify multiple')
@click.option('--annotations', '-a', required=True, multiple=True, type=click.Path(exists=True), help='Annotations CSV(s) - must match videos')
@click.option('--config', '-c', required=True, type=click.Path(exists=True), help='Config YAML file')
@click.option('--output', '-o', default=None, type=click.Path(), help='Output model path')
@click.option('--no-heatmap', is_flag=True, default=None, help='Use direct coordinate regression instead of heatmap')
@click.option('--batch-size', default=None, type=int, help='Batch size')
@click.option('--epochs', default=None, type=int, help='Maximum number of training epochs')
@click.option('--patience', default=None, type=int, help='Early stopping patience')
@click.option('--learning-rate', '--lr', default=None, type=float, help='Learning rate')
@click.option('--val-split', default=None, type=float, help='Validation split ratio')
@click.option('--backbone', default=None, type=click.Choice(['resnet18', 'resnet50']), help='Backbone architecture')
@click.option('--heatmap-sigma', default=None, type=float, help='Gaussian sigma for heatmap target')
def train_cnn(video, annotations, config, output, no_heatmap, batch_size, epochs, patience, learning_rate, val_split, backbone, heatmap_sigma):
    """Train CNN model."""
    if len(video) != len(annotations):
        raise click.BadParameter(f"Number of videos ({len(video)}) must match number of annotations ({len(annotations)})")

    # Load config and apply CLI overrides
    cfg = load_config(config)

    use_heatmap = cfg['model']['use_heatmap'] if no_heatmap is None else (not no_heatmap)
    backbone = backbone or cfg['model']['backbone']
    batch_size = batch_size or cfg['training']['batch_size']
    epochs = epochs or cfg['training']['epochs']
    patience = patience or cfg['training']['patience']
    learning_rate = learning_rate or cfg['training']['learning_rate']
    val_split = val_split or cfg['training']['val_split']
    heatmap_sigma = heatmap_sigma or cfg['heatmap']['sigma']
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
@click.option('--output-video', '-o', default=None, type=click.Path(), help='Output video path')
@click.option('--output-csv', default=None, type=click.Path(), help='Output CSV path')
@click.option('--zone-polygons', default='config/zone_polygons.yaml', type=click.Path(), help='Zone polygons YAML')
@click.option('--zone-graph', default='config/zone_graph.yaml', type=click.Path(), help='Zone graph YAML')
@click.option('--max-speed', default=None, type=float, help='Max movement speed (normalized 0-1 per frame)')
@click.option('--min-confidence', default=None, type=float, help='Min confidence for valid transitions')
@click.option('--min-confidence-forbidden', default=None, type=float, help='Min confidence for forbidden transitions')
@click.option('--min-frames-same', default=None, type=int, help='Min frames in same zone before switching')
@click.option('--smoothing', default=None, type=float, help='EMA smoothing factor (0.0-1.0, lower=smoother)')
@click.option('--backbone', default=None, type=click.Choice(['resnet18', 'resnet50']), help='Backbone architecture (must match training)')
@click.option('--no-heatmap', is_flag=True, default=None, help='Use direct coordinate regression instead of heatmap')
def track(video, config, cnn_model, output_video, output_csv, zone_polygons, zone_graph, max_speed, min_confidence, min_confidence_forbidden, min_frames_same, smoothing, backbone, no_heatmap):
    """Run tracking."""
    # Load config and apply CLI overrides
    cfg = load_config(config)

    use_heatmap = cfg['model']['use_heatmap'] if no_heatmap is None else (not no_heatmap)
    backbone = backbone or cfg['model']['backbone']
    smoothing = smoothing if smoothing is not None else cfg['tracking']['smoothing']
    min_confidence = min_confidence if min_confidence is not None else cfg['tracking']['min_confidence']
    min_confidence_forbidden = min_confidence_forbidden if min_confidence_forbidden is not None else cfg['tracking']['min_confidence_forbidden']
    min_frames_same = min_frames_same if min_frames_same is not None else cfg['tracking']['min_frames_same']
    max_speed = max_speed if max_speed is not None else cfg['tracking']['max_speed']

    model_type = 'heatmap' if use_heatmap else 'regression'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if cnn_model is None:
        cnn_model = f'models/mouse_cnn_{model_type}.pth'
    if output_video is None:
        output_video = f'output/tracking_{model_type}_{timestamp}.mp4'
    if output_csv is None:
        output_csv = f'output/tracking_{model_type}_{timestamp}.csv'

    track_video(
        video_path=video,
        cnn_model_path=cnn_model,
        output_video=output_video,
        output_csv=output_csv,
        zone_polygons_file=zone_polygons,
        zone_graph_file=zone_graph,
        min_confidence=min_confidence,
        min_confidence_forbidden=min_confidence_forbidden,
        min_frames_same=min_frames_same,
        max_speed=max_speed,
        smoothing=smoothing,
        backbone=backbone,
        use_heatmap=use_heatmap,
    )


@main.command()
@click.option('--video', '-v', required=True, type=click.Path(exists=True), help='Video file')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output CSV')
def annotate(video, output):
    """Annotate video frames (coordinates only)."""
    annotate_video(
        video_path=video,
        output_csv=output,
        num_samples=300,
    )


@main.command()
@click.option('--video', '-v', required=True, type=click.Path(exists=True), help='Video file')
@click.option('--output', '-o', default='config/zone_polygons.yaml', type=click.Path(), help='Output YAML')
def define_zones_cmd(video, output):
    """Define zone polygons."""
    define_zones(
        video_path=video,
        output_file=output,
    )


def cli_main():
    main()


if __name__ == '__main__':
    cli_main()
