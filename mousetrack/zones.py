"""Zone definition tool."""
import cv2
import yaml
import numpy as np
from typing import Dict, List

from . import logger


ZONES = ['Room_1', 'Room_2', 'Room_3', 'Tube_1', 'Tube_2', 'Tube_3', 'Tube_4']

ZONE_COLORS = {
    'Room_1': (100, 200, 100),
    'Room_2': (100, 200, 200),
    'Room_3': (200, 200, 100),
    'Tube_1': (100, 100, 255),
    'Tube_2': (255, 100, 100),
    'Tube_3': (100, 255, 255),
    'Tube_4': (255, 200, 100),
}


def define_zones(
    video_path: str,
    output_file: str = 'config/zone_polygons.yaml',
):
    """
    Interactive tool to define zone polygons by clicking points on the video.
    
    Args:
        video_path: Path to input video
        output_file: Path to output YAML file
    """
    # Global state
    current_zone_idx = 0
    current_polygon = []
    polygons = {zone: [] for zone in ZONES}
    frame = None
    frame_idx = 0
    cap = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append([x, y])
            logger.info(f"  Added point ({x}, {y}) - {len(current_polygon)} points total")
            redraw_frame()
    
    def redraw_frame():
        nonlocal frame, current_polygon, current_zone_idx
        display_frame = frame.copy()
        current_zone = ZONES[current_zone_idx]
        
        # Draw all completed polygons/polylines
        for zone, poly in polygons.items():
            if len(poly) >= 2:
                color = ZONE_COLORS.get(zone, (128, 128, 128))
                poly_array = np.array(poly, dtype=np.int32)
                if zone.startswith('Room_') and len(poly) >= 3:
                    cv2.fillPoly(display_frame, [poly_array], color)
                    cv2.polylines(display_frame, [poly_array], True, color, 2)
                elif zone.startswith('Tube_'):
                    cv2.polylines(display_frame, [poly_array], False, color, 2)
        
        # Draw current polygon/polyline being defined
        if len(current_polygon) >= 2:
            color = ZONE_COLORS.get(current_zone, (128, 128, 128))
            points = np.array(current_polygon, dtype=np.int32)
            is_closed = current_zone.startswith('Room_')
            cv2.polylines(display_frame, [points], is_closed, color, 2)
            for pt in current_polygon:
                cv2.circle(display_frame, tuple(pt), 5, color, -1)
        
        cv2.putText(display_frame, f"Defining: {current_zone}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Points: {len(current_polygon)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "LEFT CLICK: Add point | ENTER: Finish | 'n': Next | 'r': Reset | 's': Save",
                    (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Zone Definition Tool', display_frame)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video: {video_path}")
        return
    
    # Load existing polygons
    try:
        with open(output_file, 'r') as f:
            existing = yaml.safe_load(f) or {}
            for zone in ZONES:
                if zone in existing and existing[zone]:
                    polygons[zone] = existing[zone]
                    logger.info(f"Loaded existing polygon for {zone} ({len(existing[zone])} points)")
    except FileNotFoundError:
        logger.info("No existing polygons found. Starting fresh.")
    except Exception as e:
        logger.warning(f"Could not load existing polygons: {e}")
    
    ret, frame = cap.read()
    if not ret:
        logger.error("Error reading video")
        return
    
    cv2.namedWindow('Zone Definition Tool', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Zone Definition Tool', mouse_callback)
    
    current_zone = ZONES[current_zone_idx]
    current_polygon = polygons[current_zone].copy() if polygons[current_zone] else []
    
    logger.info("\n=== Zone Definition Tool ===")
    logger.info(f"Current zone: {current_zone}")
    logger.info("Instructions:")
    logger.info("  - LEFT CLICK: Add point to polygon")
    logger.info("  - ENTER: Finish current polygon and move to next zone")
    logger.info("  - 'n': Next zone (without finishing current)")
    logger.info("  - 'p': Previous zone")
    logger.info("  - 'r': Reset current polygon")
    logger.info("  - 's': Save polygons to file")
    logger.info("  - 'q': Quit")
    logger.info("")
    
    redraw_frame()
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == 13 or key == 10:  # ENTER
            min_points = 3 if current_zone.startswith('Room_') else 2
            if len(current_polygon) >= min_points:
                final_poly = current_polygon.copy()
                if current_zone.startswith('Tube_') and len(final_poly) > 1:
                    if final_poly[-1] == final_poly[0]:
                        final_poly = final_poly[:-1]
                polygons[current_zone] = final_poly
                logger.info(f"Finished {current_zone} with {len(final_poly)} points")
                current_zone_idx = (current_zone_idx + 1) % len(ZONES)
                current_zone = ZONES[current_zone_idx]
                current_polygon = polygons[current_zone].copy() if polygons[current_zone] else []
                logger.info(f"\nNow defining: {current_zone}")
            else:
                logger.info(f"Need at least {min_points} points! Currently have {len(current_polygon)}")
            redraw_frame()
        
        elif key == ord('n'):
            current_zone_idx = (current_zone_idx + 1) % len(ZONES)
            current_zone = ZONES[current_zone_idx]
            current_polygon = polygons[current_zone].copy() if polygons[current_zone] else []
            logger.info(f"Switched to: {current_zone}")
            redraw_frame()
        
        elif key == ord('p'):
            current_zone_idx = (current_zone_idx - 1) % len(ZONES)
            current_zone = ZONES[current_zone_idx]
            current_polygon = polygons[current_zone].copy() if polygons[current_zone] else []
            logger.info(f"Switched to: {current_zone}")
            redraw_frame()
        
        elif key == ord('r'):
            current_polygon = []
            logger.info(f"Reset {current_zone}")
            redraw_frame()
        
        elif key == ord('s'):
            save_data = {}
            for zone, poly in polygons.items():
                min_points = 3 if zone.startswith('Room_') else 2
                if len(poly) >= min_points:
                    final_poly = poly.copy()
                    if zone.startswith('Tube_') and len(final_poly) > 1:
                        if final_poly[-1] == final_poly[0]:
                            final_poly = final_poly[:-1]
                    save_data[zone] = final_poly
            
            with open(output_file, 'w') as f:
                for zone, points in save_data.items():
                    f.write(f"{zone}:\n")
                    for point in points:
                        f.write(f"  - [{point[0]}, {point[1]}]\n")
            logger.info(f"\nSaved {len(save_data)} zones to {output_file}")
            for zone, poly in save_data.items():
                logger.info(f"  - {zone}: {len(poly)} points")
        
        elif key == ord('q'):
            break
    
    # Save on exit
    save_data = {}
    for zone, poly in polygons.items():
        min_points = 3 if zone.startswith('Room_') else 2
        if len(poly) >= min_points:
            final_poly = poly.copy()
            if zone.startswith('Tube_') and len(final_poly) > 1:
                if final_poly[-1] == final_poly[0]:
                    final_poly = final_poly[:-1]
            save_data[zone] = final_poly
    
    if save_data:
        with open(output_file, 'w') as f:
            for zone, points in save_data.items():
                f.write(f"{zone}:\n")
                for point in points:
                    f.write(f"  - [{point[0]}, {point[1]}]\n")
        logger.info(f"\nSaved {len(save_data)} zones to {output_file}")
    
    cap.release()
    cv2.destroyAllWindows()
