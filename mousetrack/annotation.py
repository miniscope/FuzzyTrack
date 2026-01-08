"""Annotation tool for creating training data."""
import cv2
import pandas as pd
import random
import numpy as np
from typing import Optional

from .geometry import load_zone_geometry


KEY_MAP = {
    ord('1'): 'Room_1', ord('2'): 'Room_2', ord('3'): 'Room_3',
    ord('q'): 'Tube_1', ord('w'): 'Tube_2', ord('e'): 'Tube_3', ord('r'): 'Tube_4'
}

ZONE_COLOR = (100, 200, 100)  # Light green
OVERLAY_ALPHA = 0.3


def annotate_video(
    video_path: str,
    output_csv: str,
    num_samples: int = 200,
    zone_polygons_file: str = 'config/zone_polygons.yaml',
):
    """
    Interactive annotation tool for creating training data.
    
    Args:
        video_path: Path to input video
        output_csv: Path to output CSV file
        num_samples: Number of random frames to annotate
        zone_polygons_file: Path to zone polygons YAML file
    """
    # Global state for mouse callback
    current_click = None
    selected_zone = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_click
        if event == cv2.EVENT_LBUTTONDOWN:
            current_click = (x, y)
            print(f"Clicked at: {current_click}")
    
    # Load zone polygons
    zone_polygons = load_zone_geometry(zone_polygons_file)
    if zone_polygons:
        print(f"Loaded {len(zone_polygons)} zone polygons for highlighting")
    else:
        print(f"Warning: No zone polygons found in {zone_polygons_file}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Pick random frame indices
    indices = sorted(random.sample(range(total_frames), num_samples))
    
    events = []
    
    print("--- INSTRUCTIONS ---")
    print("1. CLICK on the mouse/light center.")
    print("2. PRESS key to select zone:")
    print("   - '1', '2', '3' for Rooms")
    print("   - 'q', 'w', 'e', 'r' for Tubes")
    print("3. Press ENTER to confirm annotation")
    print("   (Press 'n' to skip frame, 'ESC' to quit)")
    
    cv2.namedWindow('Annotator', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Annotator', mouse_callback)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        current_click = None
        selected_zone = None
        
        while True:
            display_frame = frame.copy()
            
            # Highlight selected zone
            if selected_zone and selected_zone in zone_polygons:
                points = zone_polygons[selected_zone]
                color = ZONE_COLOR
                
                if selected_zone.startswith('Room_'):
                    polygon = np.array(points, dtype=np.int32)
                    overlay = display_frame.copy()
                    cv2.fillPoly(overlay, [polygon], color)
                    cv2.addWeighted(overlay, OVERLAY_ALPHA, display_frame, 1 - OVERLAY_ALPHA, 0, display_frame)
                    cv2.polylines(display_frame, [polygon], True, color, 3)
                elif selected_zone.startswith('Tube_'):
                    polyline = np.array(points, dtype=np.int32)
                    cv2.polylines(display_frame, [polyline], False, color, 3)
            
            # Show crosshair if clicked
            if current_click:
                cv2.circle(display_frame, current_click, 5, (0, 0, 255), -1)
            
            # Status text
            status_text = []
            if current_click:
                status_text.append(f"Position: {current_click}")
            else:
                status_text.append("Click to set position")
            
            if selected_zone:
                status_text.append(f"Zone: {selected_zone} [Press ENTER to confirm]")
                cv2.putText(display_frame, selected_zone, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                status_text.append("Press key to select zone")
            
            cv2.putText(display_frame, f"Frame: {idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            y_offset = 60
            for i, text in enumerate(status_text):
                cv2.putText(display_frame, text, (10, y_offset + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Annotator', display_frame)
            
            key = cv2.waitKey(20) & 0xFF
            
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return
            
            if key == ord('n'):  # Skip
                print("Skipped.")
                break
            
            # Zone selection
            if key in KEY_MAP:
                selected_zone = KEY_MAP[key]
                print(f"Selected zone: {selected_zone}")
                continue
            
            # Enter to confirm
            if key == 13 or key == 10:  # ENTER
                if current_click is None:
                    print(">> Please CLICK the mouse position first!")
                    continue
                if selected_zone is None:
                    print(">> Please SELECT a zone first (press '1'-'3' or 'q'-'r')!")
                    continue
                
                events.append({
                    'frame_idx': idx,
                    'label': selected_zone,
                    'x': current_click[0],
                    'y': current_click[1]
                })
                print(f"✓ Saved: {selected_zone} at {current_click}")
                break
    
    # Save
    df = pd.DataFrame(events)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(events)} annotations to {output_csv}")
    cap.release()
    cv2.destroyAllWindows()
