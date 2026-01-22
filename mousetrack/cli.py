"""Command-line interface for mouse tracking."""
from datetime import datetime
import click

from .training import train_cnn as train_cnn_func
from .tracking import track_video
from .annotation import annotate_video
from .zones import define_zones


@click.group()
def main():
    """Mouse tracking CLI."""
    pass


@main.command()
@click.option('--video', '-v', required=True, multiple=True, type=click.Path(exists=True), help='Video file(s) - can specify multiple')
@click.option('--annotations', '-a', required=True, multiple=True, type=click.Path(exists=True), help='Annotations CSV(s) - must match videos')
@click.option('--output', '-o', default=None, type=click.Path(), help='Output model (default: models/mouse_cnn_heatmap.pth or models/mouse_cnn_regression.pth)')
@click.option('--no-heatmap', is_flag=True, default=False, help='Use direct coordinate regression instead of heatmap')
@click.option('--batch-size', default=16, type=int, help='Batch size')
@click.option('--epochs', default=200, type=int, help='Maximum number of training epochs')
@click.option('--patience', default=20, type=int, help='Early stopping patience')
@click.option('--learning-rate', '--lr', default=5e-4, type=float, help='Learning rate')
@click.option('--val-split', default=0.2, type=float, help='Validation split ratio')
def train_cnn(video, annotations, output, no_heatmap, batch_size, epochs, patience, learning_rate, val_split):
    """Train CNN model."""
    if len(video) != len(annotations):
        raise click.BadParameter(f"Number of videos ({len(video)}) must match number of annotations ({len(annotations)})")

    use_heatmap = not no_heatmap
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
    )


@main.command()
@click.option('--video', '-v', required=True, type=click.Path(exists=True), help='Video file')
@click.option('--cnn-model', '-c', default=None, type=click.Path(exists=True), help='CNN model (default: models/mouse_cnn_heatmap.pth or models/mouse_cnn_regression.pth)')
@click.option('--output-video', '-o', default=None, type=click.Path(), help='Output video (default: output/tracking_heatmap.mp4 or output/tracking_regression.mp4)')
@click.option('--output-csv', default=None, type=click.Path(), help='Output CSV (default: output/tracking_heatmap.csv or output/tracking_regression.csv)')
@click.option('--zone-polygons', default='config/zone_polygons.yaml', type=click.Path(), help='Zone polygons YAML')
@click.option('--zone-graph', default='config/zone_graph.yaml', type=click.Path(), help='Zone graph YAML')
@click.option('--max-speed', type=float, help='Max movement speed (normalized 0-1 per frame)')
@click.option('--min-confidence', default=0.5, type=float, help='Min confidence for valid transitions (tube-room, room-tube)')
@click.option('--min-confidence-forbidden', default=0.8, type=float, help='Min confidence for forbidden transitions (tube-tube, room-room)')
@click.option('--min-frames-same', default=1, type=int, help='Min frames in same zone before switching')
@click.option('--no-heatmap', is_flag=True, default=False, help='Use direct coordinate regression instead of heatmap')
def track(video, cnn_model, output_video, output_csv, zone_polygons, zone_graph, max_speed, min_confidence, min_confidence_forbidden, min_frames_same, no_heatmap):
    """Run tracking."""
    use_heatmap = not no_heatmap
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
