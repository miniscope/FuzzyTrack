"""Annotation tool for creating training data."""
import cv2
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from . import logger


def annotate_video(
    video_path: str,
    output_csv: str,
    num_samples: int = 400,
    input_csv: str = None,
):
    """
    Interactive annotation tool for creating training data (coordinates only).

    Args:
        video_path: Path to input video
        output_csv: Path to output CSV file
        num_samples: Number of random frames to annotate
        input_csv: Optional existing CSV to append to (default: create new)
    """
    # Load existing annotations if provided
    existing_frames = set()
    existing_events = []

    if input_csv:
        try:
            existing_df = pd.read_csv(input_csv)
            existing_events = existing_df.to_dict('records')
            existing_frames = set(existing_df['frame_idx'].values)
            logger.info(f"Loaded {len(existing_events)} existing annotations from {input_csv}")
        except Exception as e:
            logger.warning(f"Could not load existing CSV: {e}. Starting fresh.")

    # Global state for mouse callback
    current_click = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_click
        if event == cv2.EVENT_LBUTTONDOWN:
            current_click = (x, y)
            logger.info(f"Clicked at: {current_click}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Track spatial coverage of annotations to avoid center-biased labels.
    grid_size = 4
    grid_counts = defaultdict(int)

    for event in existing_events:
        x, y = event['x'], event['y']
        grid_x = min(int(x / frame_width * grid_size), grid_size - 1)
        grid_y = min(int(y / frame_height * grid_size), grid_size - 1)
        grid_counts[(grid_x, grid_y)] += 1

    # Pick random frame indices (excluding already annotated frames)
    available_frames = [f for f in range(total_frames) if f not in existing_frames]
    if len(available_frames) < num_samples:
        logger.warning(f"Only {len(available_frames)} unannotated frames available (requested {num_samples})")
        num_samples = len(available_frames)

    indices = sorted(random.sample(available_frames, num_samples)) if available_frames else []

    events = []
    
    logger.info("--- INSTRUCTIONS ---")
    logger.info("1. CLICK on the mouse/light center.")
    logger.info("2. Press ENTER to confirm annotation")
    logger.info("   (Press 'n' to skip frame, 'ESC' to quit)")
    
    cv2.namedWindow('Annotator', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Annotator', mouse_callback)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        current_click = None
        
        while True:
            display_frame = frame.copy()
            
            # Show crosshair if clicked
            if current_click:
                cv2.circle(display_frame, current_click, 5, (0, 0, 255), -1)

            # Draw a compact coverage heatmap in the corner.
            grid_viz_size = 120
            grid_cell_size = grid_viz_size // grid_size
            grid_viz = np.zeros((grid_viz_size, grid_viz_size, 3), dtype=np.uint8)
            max_count = max(grid_counts.values()) if grid_counts else 1
            for gx in range(grid_size):
                for gy in range(grid_size):
                    count = grid_counts.get((gx, gy), 0)
                    intensity = int(255 * min(count / max(max_count, 5), 1.0))
                    color = (0, intensity, 0)
                    x1 = gx * grid_cell_size
                    y1 = gy * grid_cell_size
                    x2 = x1 + grid_cell_size
                    y2 = y1 + grid_cell_size
                    cv2.rectangle(grid_viz, (x1, y1), (x2, y2), color, -1)
                    cv2.rectangle(grid_viz, (x1, y1), (x2, y2), (100, 100, 100), 1)

            x_offset = frame.shape[1] - grid_viz_size - 10
            y_offset_viz = 10
            display_frame[
                y_offset_viz:y_offset_viz + grid_viz_size,
                x_offset:x_offset + grid_viz_size,
            ] = grid_viz
            cv2.putText(
                display_frame,
                "Coverage",
                (x_offset, y_offset_viz - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            
            # Status text
            status_text = []
            if current_click:
                status_text.append(f"Position: {current_click}")
                status_text.append("Press ENTER to confirm")
            else:
                status_text.append("Click to set position")
            
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
                logger.info("Skipped.")
                break
            
            # Enter to confirm
            if key == 13 or key == 10:  # ENTER
                if current_click is None:
                    logger.info(">> Please CLICK the mouse position first!")
                    continue

                x, y = current_click
                events.append({
                    'frame_idx': idx,
                    'x': x,
                    'y': y
                })

                grid_x = min(int(x / frame_width * grid_size), grid_size - 1)
                grid_y = min(int(y / frame_height * grid_size), grid_size - 1)
                grid_counts[(grid_x, grid_y)] += 1

                logger.info(f"Saved: {current_click} (grid cell [{grid_x}, {grid_y}])")
                break
    
    # Combine with existing annotations
    all_events = existing_events + events

    # Save
    df = pd.DataFrame(all_events)
    if not df.empty:
        df = df.sort_values('frame_idx')
    df.to_csv(output_csv, index=False)

    if existing_events:
        logger.info(f"Saved {len(events)} new annotations (total: {len(all_events)}) to {output_csv}")
    else:
        logger.info(f"Saved {len(events)} annotations to {output_csv}")

    logger.info("\n--- Spatial Coverage Report ---")
    total_cells = grid_size * grid_size
    covered_cells = len(grid_counts)
    logger.info(f"Grid cells covered: {covered_cells}/{total_cells} ({100 * covered_cells / total_cells:.1f}%)")
    logger.info("Samples per grid cell:")
    for gy in range(grid_size):
        row_str = ""
        for gx in range(grid_size):
            count = grid_counts.get((gx, gy), 0)
            row_str += f"{count:3d} "
        logger.info(f"  Row {gy}: {row_str}")

    min_samples = 3
    undersampled = [
        (gx, gy)
        for gx in range(grid_size)
        for gy in range(grid_size)
        if grid_counts.get((gx, gy), 0) < min_samples
    ]
    if undersampled:
        logger.info(f"\n{len(undersampled)} cells have < {min_samples} samples")
        logger.info("Consider annotating more frames in these regions:")
        for gx, gy in undersampled[:5]:
            x_range = f"{int(gx * frame_width / grid_size)}-{int((gx + 1) * frame_width / grid_size)}"
            y_range = f"{int(gy * frame_height / grid_size)}-{int((gy + 1) * frame_height / grid_size)}"
            logger.info(
                f"  Cell [{gx},{gy}]: x={x_range}, y={y_range} ({grid_counts.get((gx, gy), 0)} samples)"
            )

    cap.release()
    cv2.destroyAllWindows()
