"""Tracking inference functions."""
import torch
import cv2
import numpy as np
import pandas as pd
import yaml
import os
from typing import Optional, Dict
from tqdm import tqdm

from .models import MouseCNN, MouseHeatmapCNN
from .config import IMG_SIZE, HEATMAP_SIZE
from .geometry import load_zone_geometry, get_zone_probabilities, closest_point_on_polyline, position_along_polyline
from . import logger


def load_zone_graph(zone_graph_file: str = 'config/zone_graph.yaml') -> Dict:
    if os.path.exists(zone_graph_file):
        try:
            with open(zone_graph_file, 'r') as f:
                data = yaml.safe_load(f)
                zones = data.get('zones', {})
                
                bidirectional_zones = {}
                for zone, zone_data in zones.items():
                    connections = zone_data.get('connections', [])
                    bidirectional_zones[zone] = {
                        'type': zone_data.get('type', 'unknown'),
                        'connections': connections.copy()
                    }
                
                for zone, zone_data in zones.items():
                    connections = zone_data.get('connections', [])
                    for connected_zone in connections:
                        if connected_zone not in bidirectional_zones:
                            zone_type = 'tube' if connected_zone.startswith('Tube_') else 'room'
                            bidirectional_zones[connected_zone] = {
                                'type': zone_type,
                                'connections': []
                            }
                        if zone not in bidirectional_zones[connected_zone]['connections']:
                            bidirectional_zones[connected_zone]['connections'].append(zone)
                
                return bidirectional_zones
        except Exception as e:
            logger.warning(f"Could not load {zone_graph_file}: {e}")
    return {}


def is_valid_transition(current_zone: str, new_zone: str, zone_graph: Dict) -> bool:
    if current_zone == new_zone:
        return True
    
    if not zone_graph:
        if (current_zone.startswith('Tube_') and new_zone.startswith('Tube_')) or \
           (current_zone.startswith('Room_') and new_zone.startswith('Room_')):
            return False
        return True
    
    if current_zone in zone_graph and new_zone in zone_graph:
        connections = zone_graph[current_zone].get('connections', [])
        return new_zone in connections
    
    return True


def extract_coords_from_heatmap(heatmap: np.ndarray, use_weighted_avg: bool = True) -> tuple[np.ndarray, float]:
    """Extract coordinates from heatmap.
    
    Args:
        heatmap: Heatmap array, shape (1, H, W) or (H, W)
        use_weighted_avg: If True, use weighted average (centroid). If False, use argmax.
    
    Returns:
        Tuple of (coordinates, confidence) where confidence is the peak value or entropy-based measure
    """
    # heatmap shape: (1, H, W) or (H, W)
    if heatmap.ndim == 3:
        heatmap = heatmap[0]
    
    h, w = heatmap.shape
    
    total = np.sum(heatmap)
    confidence = np.max(heatmap) if total > 0 else 0.0
    
    if use_weighted_avg:
        # Use weighted average (centroid) for more stable extraction
        # This is more robust than argmax, especially for noisy heatmaps
        if total > 0:
            # Create coordinate grids (0-indexed, so center of pixel 0 is at 0.5)
            y_coords, x_coords = np.ogrid[:h, :w]
            # Weighted average
            hm_x = np.sum(heatmap * x_coords) / total
            hm_y = np.sum(heatmap * y_coords) / total
            
            hm_x = np.clip(hm_x, 0.0, w - 1.0)
            hm_y = np.clip(hm_y, 0.0, h - 1.0)
        else:
            # Fallback to center if heatmap is all zeros
            hm_x = (w - 1) / 2.0
            hm_y = (h - 1) / 2.0
            confidence = 0.0
    else:
        # Find peak location (argmax)
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        hm_y, hm_x = max_idx
    
    norm_x = np.clip(hm_x / (w - 1) if w > 1 else 0.5, 0.0, 1.0)
    norm_y = np.clip(hm_y / (h - 1) if h > 1 else 0.5, 0.0, 1.0)
    
    return np.array([norm_x, norm_y], dtype=np.float32), float(confidence)


def track_video(
    video_path: str,
    cnn_model_path: str,
    output_video: str,
    output_csv: str,
    zone_polygons_file: str,
    zone_graph_file: str,
    min_confidence: float,
    min_confidence_forbidden: float,
    min_frames_same: int,
    max_speed: Optional[float],
    use_heatmap: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load CNN model
    if use_heatmap:
        cnn = MouseHeatmapCNN()
        logger.info("Loading heatmap CNN model...")
    else:
        cnn = MouseCNN()
        logger.info("Loading coordinate CNN model...")
    
    cnn.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn.to(device)
    cnn.eval()
    logger.info("CNN model loaded successfully.")
    
    # Load zone geometry
    zone_polygons = load_zone_geometry(zone_polygons_file)
    if not zone_polygons:
        logger.warning(f"No zone polygons found in {zone_polygons_file}")
    else:
        logger.info(f"Loaded {len(zone_polygons)} zone polygons")
    
    # Load zone graph
    zone_graph = load_zone_graph(zone_graph_file)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Input video: {width}x{height} @ {fps} fps, {total_frames} frames")

    # Create output directories if needed
    for path in [output_video, output_csv]:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not out_video.isOpened():
        logger.error(f"Error creating output video: {output_video}")
        cap.release()
        return
    
    logger.info(f"Output video: {output_video}")
    
    # Prepare for processing
    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Error reading first frame.")
        cap.release()
        out_video.release()
        return
    
    prev_gray = cv2.resize(prev_frame, IMG_SIZE)
    prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
    
    # State variables
    current_zone = None
    frames_in_current_zone = 0
    prev_refined_coords = None  # Previous refined coordinates for max_speed constraint
    
    # Warm-up period for stable initialization
    warmup_frames = 30 if use_heatmap else 5  # Much longer warm-up for heatmap
    warmup_zone_candidates = {}  # Track zone candidates during warm-up
    warmup_coords = []  # Track coordinates during warm-up for smoothing
    heatmap_min_confidence = 0.3  # Minimum heatmap confidence to accept prediction (higher = more strict)
    min_warmup_confident_frames = 10 if use_heatmap else 3  # Need at least N confident predictions before starting
    
    results = []
    frame_idx = 0
    
    logger.info("Starting tracking...")
    
    pbar = tqdm(total=total_frames, desc="Tracking", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessing
        curr_small = cv2.resize(frame, IMG_SIZE)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, prev_gray)
        input_img = diff.astype(np.float32) / 255.0
        input_tensor = torch.tensor(input_img).unsqueeze(0).unsqueeze(0).to(device)
        
        # CNN inference
        with torch.no_grad():
            model_out = cnn(input_tensor)
        
        # Extract coordinates from model output
        heatmap_conf = None  # Only set for heatmap models
        if use_heatmap:
            # Model outputs heatmap, extract coordinates using weighted average for stability
            heatmap = model_out[0, 0].cpu().numpy()  # (H, W)
            raw_coords, heatmap_conf = extract_coords_from_heatmap(heatmap, use_weighted_avg=True)

            if frame_idx < warmup_frames:
                if heatmap_conf >= heatmap_min_confidence:
                    warmup_coords.append(raw_coords.copy())
                    raw_coords = np.mean(warmup_coords, axis=0)
                elif len(warmup_coords) >= min_warmup_confident_frames:
                    raw_coords = np.mean(warmup_coords, axis=0)
                elif len(warmup_coords) > 0:
                    raw_coords = warmup_coords[-1].copy()
        else:
            # Model outputs coordinates directly
            raw_coords = model_out[0].cpu().numpy()
        
        # Apply max_speed constraint if specified
        if max_speed is not None and prev_refined_coords is not None:
            movement = raw_coords - prev_refined_coords
            movement_norm = np.linalg.norm(movement)
            if movement_norm > max_speed:
                # Scale down movement to max_speed
                movement = movement * (max_speed / movement_norm)
                raw_coords = prev_refined_coords + movement
        
        raw_coords = np.clip(raw_coords, 0.0, 1.0)
        
        # Update previous refined coordinates for next iteration
        prev_refined_coords = raw_coords.copy()
        
        norm_x, norm_y = raw_coords
        pred_x = norm_x * width
        pred_y = norm_y * height
        
        if zone_polygons:
            zone_probs = get_zone_probabilities(pred_x, pred_y, zone_polygons, tube_max_distance=20.0, soft_boundary=True, normalize=True)
            if zone_probs:
                new_zone = max(zone_probs, key=zone_probs.get)
                pred_confidence = zone_probs[new_zone]
            else:
                new_zone = None
                pred_confidence = 0.0
        else:
            new_zone = None
            pred_confidence = 0.0
        
        if current_zone is not None and new_zone is not None:
            if not is_valid_transition(current_zone, new_zone, zone_graph):
                if zone_polygons:
                    valid_alternatives = []
                    for alt_zone, alt_prob in zone_probs.items():
                        if alt_prob > 0 and is_valid_transition(current_zone, alt_zone, zone_graph):
                            valid_alternatives.append((alt_zone, alt_prob))
                    
                    if valid_alternatives:
                        best_alt_zone = None
                        best_alt_prob = 0.0
                        for alt_zone, alt_prob in valid_alternatives:
                            if alt_prob > best_alt_prob:
                                best_alt_prob = alt_prob
                                best_alt_zone = alt_zone
                        if best_alt_prob >= min_confidence:
                            new_zone = best_alt_zone
                            pred_confidence = best_alt_prob
                        else:
                            new_zone = current_zone
                            pred_confidence = zone_probs.get(current_zone, 0.0)
                    else:
                        new_zone = current_zone
                        pred_confidence = zone_probs.get(current_zone, 0.0)
        
        if current_zone is None:
            if frame_idx < warmup_frames:
                if use_heatmap and len(warmup_coords) < min_warmup_confident_frames:
                    pass
                elif new_zone and pred_confidence >= min_confidence:
                    if new_zone not in warmup_zone_candidates:
                        warmup_zone_candidates[new_zone] = 0
                    warmup_zone_candidates[new_zone] += 1
                
                if frame_idx == warmup_frames - 1:
                    if warmup_zone_candidates:
                        best_zone_name = None
                        best_count = 0
                        for zone_name, count in warmup_zone_candidates.items():
                            if count > best_count:
                                best_count = count
                                best_zone_name = zone_name
                        best_candidate = (best_zone_name, best_count)
                        min_consistency = 0.7 if use_heatmap else 0.6
                        if best_candidate[1] >= warmup_frames * min_consistency:
                            current_zone = best_candidate[0]
                            frames_in_current_zone = warmup_frames
                        else:
                            required_conf = max(min_confidence, 0.6) if use_heatmap else min_confidence
                            if new_zone and pred_confidence >= required_conf:
                                current_zone = new_zone
                                frames_in_current_zone = 1
            else:
                if use_heatmap and len(warmup_coords) < min_warmup_confident_frames:
                    pass
                elif new_zone and pred_confidence >= min_confidence:
                    current_zone = new_zone
                    frames_in_current_zone = 1
        elif new_zone == current_zone:
            frames_in_current_zone += 1
        else:
            is_valid = is_valid_transition(current_zone, new_zone, zone_graph)
            required_confidence = min_confidence_forbidden if not is_valid else min_confidence
            
            if pred_confidence >= required_confidence and frames_in_current_zone >= min_frames_same:
                current_zone = new_zone
                frames_in_current_zone = 1
            else:
                frames_in_current_zone += 1
        
        pred_label = current_zone if current_zone else "Unknown"

        pixel_x = int(norm_x * width)
        pixel_y = int(norm_y * height)

        # Calculate tube position if in a tube
        tube_position = None
        if pred_label.startswith('Tube_') and pred_label in zone_polygons:
            tube_position = position_along_polyline((pixel_x, pixel_y), zone_polygons[pred_label])

        # Get likelihood/confidence for this frame
        # Use zone confidence if available, otherwise heatmap confidence (None for non-heatmap models)
        likelihood = pred_confidence if pred_confidence > 0 else heatmap_conf

        result_row = {
            'x': pixel_x,
            'y': pixel_y,
            'likelihood': likelihood,
            'zone': pred_label,
            'tube_position': tube_position,
        }

        results.append(result_row)
        
        if pred_label in zone_polygons:
            points = zone_polygons[pred_label]
            zone_color = (100, 200, 100)
            overlay_alpha = 0.3
            if pred_label.startswith('Room_'):
                polygon = np.array(points, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon], zone_color)
                cv2.addWeighted(overlay, overlay_alpha, frame, 1 - overlay_alpha, 0, frame)
                cv2.polylines(frame, [polygon], True, zone_color, 3)
            elif pred_label.startswith('Tube_'):
                polyline = np.array(points, dtype=np.int32)
                cv2.polylines(frame, [polyline], False, zone_color, 3)
                closest_pt = closest_point_on_polyline((pixel_x, pixel_y), polyline)
                cv2.circle(frame, tuple(closest_pt), 18, (0, 255, 0), -1)
        
        cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 0, 255), -1)
        
        cv2.putText(frame, f"Frame: {frame_idx}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out_video.write(frame)
        
        pbar.update(1)
        prev_gray = curr_gray
        frame_idx += 1
    
    pbar.close()
    cap.release()
    out_video.release()
    
    # Save CSV in DLC format with multi-level header
    # Format: scorer, bodyparts, coords as first 3 rows, then data
    scorer = "3DMazeTrack"
    bodypart = "LED"
    columns = ['x', 'y', 'likelihood', 'zone', 'tube_position']

    # Create multi-index columns
    header_tuples = [(scorer, bodypart, col) for col in columns]
    multi_index = pd.MultiIndex.from_tuples(header_tuples, names=['scorer', 'bodyparts', 'coords'])

    df = pd.DataFrame(results, columns=columns)
    df.columns = multi_index

    df.to_csv(output_csv)
    logger.info("Tracking complete!")
    logger.info(f"  - Video saved to: {output_video}")
    logger.info(f"  - CSV saved to: {output_csv}")
    logger.info(f"  - Total frames processed: {frame_idx}")
    
