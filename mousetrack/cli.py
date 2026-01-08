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
def train_cnn(video, annotations, output):
    """Train CNN model."""
    train_cnn_func(
        video_path=video,
        annotations_path=annotations,
        output_path=output,
    )


@main.command()
@click.option('--cnn-model', '-c', required=True, type=click.Path(exists=True), help='CNN model file')
@click.option('--video', '-v', required=True, type=click.Path(exists=True), help='Video file')
@click.option('--annotations', '-a', required=True, type=click.Path(exists=True), help='Annotations CSV')
@click.option('--output', '-o', default='models/mouse_lstm.pth', type=click.Path(), help='Output model')
@click.option('--sequence-length', default=10, type=int, help='Sequence length for LSTM')
@click.option('--batch-size', default=32, type=int, help='Batch size')
@click.option('--epochs', default=2000, type=int, help='Maximum number of training epochs')
@click.option('--patience', default=100, type=int, help='Early stopping patience')
@click.option('--learning-rate', '--lr', default=1e-3, type=float, help='Learning rate')
@click.option('--hidden-size', default=64, type=int, help='LSTM hidden size')
@click.option('--num-layers', default=2, type=int, help='Number of LSTM layers')
@click.option('--dropout', default=0.1, type=float, help='Dropout rate')
def train_lstm(cnn_model, video, annotations, output, sequence_length, batch_size, epochs, patience, learning_rate, hidden_size, num_layers, dropout):
    """Train LSTM model."""
    from .training import train_lstm as train_lstm_func
    train_lstm_func(
        cnn_model_path=cnn_model,
        video_path=video,
        annotations_path=annotations,
        output_path=output,
        sequence_length=sequence_length,
        batch_size=batch_size,
        max_epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


@main.command()
@click.option('--video', '-v', required=True, type=click.Path(exists=True), help='Video file')
@click.option('--cnn-model', '-c', required=True, type=click.Path(exists=True), help='CNN model')
@click.option('--lstm-model', '-l', type=click.Path(exists=True), help='LSTM model (optional)')
@click.option('--output-video', '-o', default='output/tracking_results.mp4', type=click.Path(), help='Output video')
@click.option('--output-csv', default='output/tracking_results.csv', type=click.Path(), help='Output CSV')
@click.option('--zone-polygons', default='config/zone_polygons.yaml', type=click.Path(), help='Zone polygons YAML')
@click.option('--zone-graph', default='config/zone_graph.yaml', type=click.Path(), help='Zone graph YAML')
@click.option('--max-speed', type=float, help='Max movement speed (normalized 0-1 per frame)')
def track(video, cnn_model, lstm_model, output_video, output_csv, zone_polygons, zone_graph, max_speed):
    """Run tracking."""
    track_video(
        video_path=video,
        cnn_model_path=cnn_model,
        lstm_model_path=lstm_model,
        output_video=output_video,
        output_csv=output_csv,
        zone_polygons_file=zone_polygons,
        zone_graph_file=zone_graph,
        max_speed=max_speed,
    )


@main.command()
@click.option('--video', '-v', required=True, type=click.Path(exists=True), help='Video file')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output CSV')
@click.option('--zone-polygons', default='config/zone_polygons.yaml', type=click.Path(), help='Zone polygons YAML')
def annotate(video, output, zone_polygons):
    """Annotate video frames."""
    annotate_video(
        video_path=video,
        output_csv=output,
        num_samples=200,
        zone_polygons_file=zone_polygons,
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
