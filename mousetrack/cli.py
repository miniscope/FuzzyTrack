"""Command-line interface for mouse tracking."""
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
@click.option('--video', '-v', required=True, type=click.Path(exists=True), help='Video file')
@click.option('--annotations', '-a', required=True, type=click.Path(exists=True), help='Annotations CSV')
@click.option('--output', '-o', default='models/mouse_cnn.pth', type=click.Path(), help='Output model')
@click.option('--no-heatmap', is_flag=True, default=False, help='Use direct coordinate regression instead of heatmap')
@click.option('--batch-size', default=16, type=int, help='Batch size')
@click.option('--epochs', default=200, type=int, help='Maximum number of training epochs')
@click.option('--patience', default=20, type=int, help='Early stopping patience')
@click.option('--learning-rate', '--lr', default=5e-4, type=float, help='Learning rate')
@click.option('--val-split', default=0.2, type=float, help='Validation split ratio')
def train_cnn(video, annotations, output, no_heatmap, batch_size, epochs, patience, learning_rate, val_split):
    """Train CNN model."""
    train_cnn_func(
        video_path=video,
        annotations_path=annotations,
        output_path=output,
        use_heatmap=not no_heatmap,
        batch_size=batch_size,
        max_epochs=epochs,
        patience=patience,
        val_split=val_split,
        learning_rate=learning_rate,
        logdir="runs/mouse_tracker",
    )


@main.command()
@click.option('--video', '-v', required=True, type=click.Path(exists=True), help='Video file')
@click.option('--cnn-model', '-c', required=True, type=click.Path(exists=True), help='CNN model')
@click.option('--output-video', '-o', default='output/tracking_results.mp4', type=click.Path(), help='Output video')
@click.option('--output-csv', default='output/tracking_results.csv', type=click.Path(), help='Output CSV')
@click.option('--zone-polygons', default='config/zone_polygons.yaml', type=click.Path(), help='Zone polygons YAML')
@click.option('--zone-graph', default='config/zone_graph.yaml', type=click.Path(), help='Zone graph YAML')
@click.option('--max-speed', type=float, help='Max movement speed (normalized 0-1 per frame)')
@click.option('--min-confidence', default=0.5, type=float, help='Min confidence for valid transitions (tube-room, room-tube)')
@click.option('--min-confidence-forbidden', default=0.8, type=float, help='Min confidence for forbidden transitions (tube-tube, room-room)')
@click.option('--min-frames-same', default=1, type=int, help='Min frames in same zone before switching')
@click.option('--no-heatmap', is_flag=True, default=False, help='Use direct coordinate regression instead of heatmap')
def track(video, cnn_model, output_video, output_csv, zone_polygons, zone_graph, max_speed, min_confidence, min_confidence_forbidden, min_frames_same, no_heatmap):
    """Run tracking."""
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
        use_heatmap=not no_heatmap,
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
