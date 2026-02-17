"""Annotation tool for creating training data."""
import cv2
import pandas as pd
import random
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
                
                events.append({
                    'frame_idx': idx,
                    'x': current_click[0],
                    'y': current_click[1]
                })
                logger.info(f"Saved: {current_click}")
                break
    
    # Combine with existing annotations
    all_events = existing_events + events

    # Save
    df = pd.DataFrame(all_events)
    df = df.sort_values('frame_idx')  # Sort by frame index
    df.to_csv(output_csv, index=False)

    if existing_events:
        logger.info(f"Saved {len(events)} new annotations (total: {len(all_events)}) to {output_csv}")
    else:
        logger.info(f"Saved {len(events)} annotations to {output_csv}")

    cap.release()
    cv2.destroyAllWindows()
