"""Tracking inference functions."""
import torch
import cv2
import numpy as np
import pandas as pd
import yaml
import os
from typing import Optional, Dict
from tqdm import tqdm

from .models import MouseCNN, CoordinateLSTM
from .config import IMG_SIZE
from .geometry import load_zone_geometry, get_zone_probabilities, closest_point_on_polyline
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
        is_tube = lambda z: z.startswith('Tube_')
        is_room = lambda z: z.startswith('Room_')
        if (is_tube(current_zone) and is_tube(new_zone)) or \
           (is_room(current_zone) and is_room(new_zone)):
            return False
        return True
    
    if current_zone in zone_graph and new_zone in zone_graph:
        connections = zone_graph[current_zone].get('connections', [])
        return new_zone in connections
    
    return True


def track_video(
    video_path: str,
    cnn_model_path: str,
    lstm_model_path: Optional[str] = None,
    output_video: str = 'output/tracking_results.mp4',
    output_csv: str = 'output/tracking_results.csv',
    zone_polygons_file: str = 'config/zone_polygons.yaml',
    zone_graph_file: str = 'config/zone_graph.yaml',
    min_confidence: float = 0.3,
    min_frames_same: int = 5,
    enable_transition_filter: bool = True,
    enable_zone_overlay: bool = True,
    zone_color: tuple = (100, 200, 100),
    overlay_alpha: float = 0.3,
    max_speed: Optional[float] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CNN model
    cnn = MouseCNN()
    cnn.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn.to(device)
    cnn.eval()
    logger.info("CNN model loaded successfully.")
    
    # Load LSTM model if provided
    lstm = None
    if lstm_model_path and os.path.exists(lstm_model_path):
        lstm = CoordinateLSTM()
        lstm.load_state_dict(torch.load(lstm_model_path, map_location=device))
        lstm.to(device)
        lstm.eval()
        logger.info("LSTM model loaded successfully.")
    
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
    coord_history = []  # For LSTM if used
    prev_refined_coords = None  # Previous refined coordinates for max_speed constraint
    
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
            coord_out = cnn(input_tensor)
        
        raw_coords = coord_out[0].cpu().numpy()
        
        # LSTM refinement if available
        if lstm is not None:
            coord_history.append(raw_coords)
            # Keep reasonable history (sequence_length frames)
            max_history = 20  # Reasonable max history
            if len(coord_history) > max_history:
                coord_history.pop(0)
            
            if len(coord_history) >= 2:  # Need at least 2 for sequence
                # Convert list of numpy arrays to single numpy array first
                coord_array = np.array(coord_history, dtype=np.float32)  # (T, 2)
                
                # Compute velocity features
                velocities = np.zeros_like(coord_array)
                if len(coord_array) > 1:
                    velocities[1:] = coord_array[1:] - coord_array[:-1]
                
                # Combine coordinates and velocity: [x, y, vx, vy]
                features = np.concatenate([coord_array, velocities], axis=-1)  # (T, 4)
                seq_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, 4)
                
                with torch.no_grad():
                    refined_seq = lstm(seq_tensor)
                raw_coords = refined_seq[0, -1].detach().cpu().numpy()  # Use last refined coordinate
        
        # Apply max_speed constraint if specified (works for both CNN-only and LSTM-refined)
        if max_speed is not None and prev_refined_coords is not None:
            movement = raw_coords - prev_refined_coords
            movement_norm = np.linalg.norm(movement)
            if movement_norm > max_speed:
                # Scale down movement to max_speed
                movement = movement * (max_speed / movement_norm)
                raw_coords = prev_refined_coords + movement
        
        # Update previous refined coordinates for next iteration
        prev_refined_coords = raw_coords.copy()
        
        # Use raw coordinates directly (no smoothing)
        norm_x, norm_y = raw_coords
        pred_x = norm_x * width
        pred_y = norm_y * height
        
        # Zone classification: geometric proximity
        # Use normalized probabilities for state tracking (better for crossing routes)
        active_zones = []
        if zone_polygons:
            # Get normalized probabilities for state tracking (like old version)
            zone_probs = get_zone_probabilities(pred_x, pred_y, zone_polygons, tube_max_distance=20.0, soft_boundary=True, normalize=True)
            
            # Select highest probability zone for state tracking (like old version)
            new_zone = max(zone_probs.items(), key=lambda x: x[1])[0] if zone_probs else None
            pred_confidence = zone_probs.get(new_zone, 0.0) if new_zone else 0.0
            
            # Get all zones above confidence threshold (for visualization)
            for zone_name, prob in zone_probs.items():
                if prob >= min_confidence:
                    active_zones.append((zone_name, prob))
            
            # Sort by confidence (highest first)
            active_zones.sort(key=lambda x: x[1], reverse=True)
        else:
            new_zone = None
            pred_confidence = 0.0
        
        # Zone transition filter
        if enable_transition_filter and current_zone is not None and new_zone is not None:
            if not is_valid_transition(current_zone, new_zone, zone_graph):
                if zone_polygons:
                    valid_alternatives = []
                    for alt_zone, alt_prob in zone_probs.items():
                        if alt_prob > 0 and is_valid_transition(current_zone, alt_zone, zone_graph):
                            valid_alternatives.append((alt_zone, alt_prob))
                    
                    if valid_alternatives:
                        valid_alternatives.sort(key=lambda x: x[1], reverse=True)
                        best_alt_zone, best_alt_prob = valid_alternatives[0]
                        if best_alt_prob >= min_confidence:
                            new_zone = best_alt_zone
                            pred_confidence = best_alt_prob
                        else:
                            # Keep current zone and use its probability, but ensure it's at least min_confidence
                            # to prevent oscillation at cross sections
                            new_zone = current_zone
                            current_prob = zone_probs.get(current_zone, 0.0)
                            pred_confidence = max(current_prob, min_confidence)
                    else:
                        # Keep current zone and use its probability, but ensure it's at least min_confidence
                        new_zone = current_zone
                        current_prob = zone_probs.get(current_zone, 0.0)
                        pred_confidence = max(current_prob, min_confidence)
        
        # Hysteresis
        if current_zone is None:
            if new_zone:
                current_zone = new_zone
                frames_in_current_zone = 1
        elif new_zone == current_zone:
            frames_in_current_zone += 1
        else:
            if pred_confidence >= min_confidence and frames_in_current_zone >= min_frames_same:
                current_zone = new_zone
                frames_in_current_zone = 1
            else:
                frames_in_current_zone += 1
        
        pred_label = current_zone if current_zone else "Unknown"
        
        # Convert to pixel coordinates
        pixel_x = int(norm_x * width)
        pixel_y = int(norm_y * height)
        
        # Store results
        result_row = {
            'frame': frame_idx,
            'zone': pred_label,
            'x': pixel_x,
            'y': pixel_y
        }
        
        results.append(result_row)
        
        # Visualization: show all active zones
        if enable_zone_overlay and zone_polygons:
            for zone_name, zone_conf in active_zones:
                if zone_name in zone_polygons:
                    points = zone_polygons[zone_name]
                    if zone_name.startswith('Room_'):
                        polygon = np.array(points, dtype=np.int32)
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [polygon], zone_color)
                        cv2.addWeighted(overlay, overlay_alpha, frame, 1 - overlay_alpha, 0, frame)
                        cv2.polylines(frame, [polygon], True, zone_color, 3)
                    elif zone_name.startswith('Tube_'):
                        polyline = np.array(points, dtype=np.int32)
                        cv2.polylines(frame, [polyline], False, zone_color, 3)
                        # Draw closest point on polyline to detected coordinate
                        closest_pt = closest_point_on_polyline((pixel_x, pixel_y), polyline)
                        cv2.circle(frame, tuple(closest_pt), 18, (0, 255, 0), -1)  # Green filled circle
        
        # Draw position (red, filled)
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
    
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logger.info("Tracking complete!")
    logger.info(f"  - Video saved to: {output_video}")
    logger.info(f"  - CSV saved to: {output_csv}")
    logger.info(f"  - Total frames processed: {frame_idx}")
    
