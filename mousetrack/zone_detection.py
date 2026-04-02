"""Zone detection from position tracking data."""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from .geometry import (
    load_zone_geometry,
    get_zone_probabilities,
    closest_point_on_polyline,
    position_along_polyline,
    is_arm_zone,
)
from .tracking import load_zone_graph, is_valid_transition
from . import logger


def detect_zones_from_csv(
    input_csv: str,
    output_csv: str,
    zone_polygons_file: str,
    zone_graph_file: str,
    min_confidence: float = 0.5,
    min_confidence_forbidden: float = 0.8,
    min_frames_same: int = 1,
    min_frames_forbidden: int = 3,
):
    """
    Detect zones from position tracking CSV.

    Args:
        input_csv: Input CSV with x, y coordinates (DLC format)
        output_csv: Output CSV with zone information added
        zone_polygons_file: Path to zone polygons YAML
        zone_graph_file: Path to zone graph YAML
        min_confidence: Minimum confidence for zone transitions
        min_confidence_forbidden: Minimum confidence for forbidden transitions
        min_frames_same: Minimum frames in zone before allowing transition
        min_frames_forbidden: Minimum consecutive frames for forbidden transition
    """
    logger.info(f"Loading position data from {input_csv}")

    # Read input CSV (DLC format)
    df = pd.read_csv(input_csv, header=[0, 1, 2], index_col=0)

    # Extract coordinates
    # DLC format: (scorer, bodypart, coords)
    # Get the first scorer and bodypart
    scorer = df.columns.get_level_values(0)[0]
    bodypart = df.columns.get_level_values(1)[0]

    x_col = (scorer, bodypart, 'x')
    y_col = (scorer, bodypart, 'y')

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Could not find x, y columns in input CSV. Expected columns: {x_col}, {y_col}")

    x_coords = df[x_col].values
    y_coords = df[y_col].values

    logger.info(f"Loaded {len(x_coords)} position samples")

    # Load zone geometry
    zone_polygons = load_zone_geometry(zone_polygons_file)
    if not zone_polygons:
        logger.warning(f"No zone polygons found in {zone_polygons_file}")
        return

    logger.info(f"Loaded {len(zone_polygons)} zone polygons")

    # Load zone graph
    zone_graph = load_zone_graph(zone_graph_file)

    # Zone detection state
    current_zone = None
    frames_in_current_zone = 0
    forbidden_candidate_zone = None
    forbidden_consecutive_frames = 0

    # Results
    zones = []
    x_pinned_list = []
    y_pinned_list = []
    arm_positions = []

    logger.info("Detecting zones...")

    for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
        # Get zone probabilities
        zone_probs = get_zone_probabilities(
            x, y, zone_polygons,
            arm_max_distance=60.0,
            soft_boundary=True,
            normalize=True
        )

        if zone_probs:
            new_zone = max(zone_probs, key=zone_probs.get)
            pred_confidence = zone_probs[new_zone]
        else:
            new_zone = None
            pred_confidence = 0.0

        # Zone transition logic (same as tracking.py)
        if current_zone is not None and new_zone is not None:
            if not is_valid_transition(current_zone, new_zone, zone_graph):
                # Invalid transition - check for valid alternatives
                if zone_polygons:
                    valid_alternatives = []
                    for alt_zone, alt_prob in zone_probs.items():
                        if alt_prob > 0 and alt_zone != new_zone and is_valid_transition(current_zone, alt_zone, zone_graph):
                            valid_alternatives.append((alt_zone, alt_prob))

                    if valid_alternatives:
                        best_alt_zone = None
                        best_alt_prob = 0.0
                        for alt_zone, alt_prob in valid_alternatives:
                            if alt_prob > best_alt_prob:
                                best_alt_prob = alt_prob
                                best_alt_zone = alt_zone
                        if best_alt_prob >= min_confidence and best_alt_prob > pred_confidence:
                            new_zone = best_alt_zone
                            pred_confidence = best_alt_prob

        if current_zone is None:
            # Initialize zone
            if new_zone and pred_confidence >= min_confidence:
                current_zone = new_zone
                frames_in_current_zone = 1
        elif new_zone == current_zone:
            frames_in_current_zone += 1
            forbidden_candidate_zone = None
            forbidden_consecutive_frames = 0
        else:
            is_valid = is_valid_transition(current_zone, new_zone, zone_graph)

            if is_valid:
                # Valid transition
                if pred_confidence >= min_confidence and frames_in_current_zone >= min_frames_same:
                    current_zone = new_zone
                    frames_in_current_zone = 1
                    forbidden_candidate_zone = None
                    forbidden_consecutive_frames = 0
                else:
                    frames_in_current_zone += 1
            else:
                # Forbidden transition
                if pred_confidence >= min_confidence_forbidden:
                    if new_zone == forbidden_candidate_zone:
                        forbidden_consecutive_frames += 1
                    else:
                        forbidden_candidate_zone = new_zone
                        forbidden_consecutive_frames = 1

                    if forbidden_consecutive_frames >= min_frames_forbidden and frames_in_current_zone >= min_frames_same:
                        current_zone = new_zone
                        frames_in_current_zone = 1
                        forbidden_candidate_zone = None
                        forbidden_consecutive_frames = 0
                    else:
                        frames_in_current_zone += 1
                else:
                    forbidden_candidate_zone = None
                    forbidden_consecutive_frames = 0
                    frames_in_current_zone += 1

        pred_label = current_zone if current_zone else "Unknown"

        # Calculate arm-pinned coordinates
        arm_position = None
        pinned_x, pinned_y = x, y
        if is_arm_zone(pred_label) and pred_label in zone_polygons:
            arm_position = position_along_polyline((x, y), zone_polygons[pred_label])
            pinned_pt = closest_point_on_polyline((x, y), zone_polygons[pred_label])
            pinned_x, pinned_y = float(pinned_pt[0]), float(pinned_pt[1])

        zones.append(pred_label)
        x_pinned_list.append(pinned_x)
        y_pinned_list.append(pinned_y)
        arm_positions.append(arm_position)

    logger.info("Zone detection complete!")

    # Create output DataFrame
    output_data = {
        (scorer, bodypart, 'x'): x_coords,
        (scorer, bodypart, 'y'): y_coords,
        (scorer, bodypart, 'x_pinned'): x_pinned_list,
        (scorer, bodypart, 'y_pinned'): y_pinned_list,
        (scorer, bodypart, 'zone'): zones,
        (scorer, bodypart, 'arm_position'): arm_positions,
    }

    # Include likelihood if it exists in input
    likelihood_col = (scorer, bodypart, 'likelihood')
    if likelihood_col in df.columns:
        output_data[likelihood_col] = df[likelihood_col].values

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv)

    logger.info(f"Zone detection results saved to {output_csv}")

    # Print zone statistics
    zone_counts = pd.Series(zones).value_counts()
    logger.info("Zone distribution:")
    for zone, count in zone_counts.items():
        logger.info(f"  {zone}: {count} frames ({100 * count / len(zones):.1f}%)")
