"""Geometric zone classification based on coordinates."""
import os
import numpy as np
import cv2
import yaml
from typing import Dict, Tuple, Optional, List

from . import logger


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """Check if a point is inside a polygon using OpenCV's pointPolygonTest."""
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0  # >= 0 means inside or on edge


def distance_to_polyline(point: Tuple[float, float], polyline: np.ndarray, max_distance: float = 20.0) -> bool:
    """
    Check if a point is near a polyline (within max_distance pixels).
    For tubes, we check distance to the center line.
    """
    # Find minimum distance to any segment of the polyline
    min_dist = float('inf')
    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
        # Distance from point to line segment
        dist = _point_to_segment_distance(point, p1, p2)
        min_dist = min(min_dist, dist)
    
    return min_dist <= max_distance


def _point_to_segment_distance(point: Tuple[float, float], seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """Calculate distance from a point to a line segment."""
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)
    
    # Vector from seg_start to seg_end
    seg_vec = seg_end - seg_start
    # Vector from seg_start to point
    point_vec = point - seg_start
    
    # Project point_vec onto seg_vec
    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq == 0:
        # Segment is a point
        return np.linalg.norm(point - seg_start)
    
    t = np.clip(np.dot(point_vec, seg_vec) / seg_len_sq, 0.0, 1.0)
    # Closest point on segment
    closest = seg_start + t * seg_vec
    return np.linalg.norm(point - closest)


def closest_point_on_polyline(point: Tuple[float, float], polyline: np.ndarray) -> np.ndarray:
    """
    Find the closest point on a polyline to a given point.
    
    Args:
        point: (x, y) coordinate
        polyline: Array of points defining the polyline
    
    Returns:
        Closest point on the polyline as (x, y) numpy array
    """
    point = np.array(point, dtype=np.float32)
    polyline = np.array(polyline, dtype=np.float32)
    
    min_dist = float('inf')
    closest_point = None
    
    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
        
        # Vector from p1 to p2
        seg_vec = p2 - p1
        # Vector from p1 to point
        point_vec = point - p1
        
        # Project point_vec onto seg_vec
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq == 0:
            # Segment is a point
            candidate = p1
        else:
            t = np.clip(np.dot(point_vec, seg_vec) / seg_len_sq, 0.0, 1.0)
            candidate = p1 + t * seg_vec
        
        dist = np.linalg.norm(point - candidate)
        if dist < min_dist:
            min_dist = dist
            closest_point = candidate
    
    # If polyline has only one point
    if closest_point is None and len(polyline) > 0:
        closest_point = polyline[0]
    
    return closest_point.astype(np.int32)


def load_zone_geometry(zone_polygons_file: str = 'config/zone_polygons.yaml') -> Dict[str, np.ndarray]:
    """Load zone polygons from YAML file."""
    polygons = {}
    if os.path.exists(zone_polygons_file):
        try:
            with open(zone_polygons_file, 'r') as f:
                data = yaml.safe_load(f)
                for zone, points in data.items():
                    if points:
                        polygons[zone] = np.array(points, dtype=np.int32)
        except Exception as e:
            logger.warning(f"Could not load {zone_polygons_file}: {e}")
    return polygons


def classify_zone_geometric(
    x: float, 
    y: float, 
    zone_polygons: Dict[str, np.ndarray],
    tube_max_distance: float = 20.0
) -> Optional[str]:
    """
    Classify which zone a point (x, y) belongs to using geometry.
    
    Args:
        x, y: Coordinates in pixel space
        zone_polygons: Dictionary mapping zone names to polygon/polyline arrays
        tube_max_distance: Maximum distance from polyline center for tube classification
    
    Returns:
        Zone name if found, None otherwise
    """
    point = (float(x), float(y))
    
    # First check rooms (closed polygons) - these take priority
    for zone_name, polygon in zone_polygons.items():
        if zone_name.startswith('Room_'):
            if point_in_polygon(point, polygon):
                return zone_name
    
    # Then check tubes (polylines) - check distance to center line
    for zone_name, polyline in zone_polygons.items():
        if zone_name.startswith('Tube_'):
            if distance_to_polyline(point, polyline, tube_max_distance):
                return zone_name
    
    return None  # Point doesn't match any zone


def get_zone_probabilities(
    x: float,
    y: float,
    zone_polygons: Dict[str, np.ndarray],
    tube_max_distance: float = 40.0,
    soft_boundary: bool = True,
    normalize: bool = True
) -> Dict[str, float]:
    point = (float(x), float(y))
    probs = {zone: 0.0 for zone in zone_polygons.keys()}
    
    for zone_name, polygon in zone_polygons.items():
        if zone_name.startswith('Room_'):
            if point_in_polygon(point, polygon):
                probs[zone_name] = 1.0
            elif soft_boundary:
                dist = cv2.pointPolygonTest(polygon, point, True)
                if dist > -tube_max_distance:
                    # Room probability decays much faster when outside
                    # Use squared decay: probability drops quickly as distance increases
                    # When dist is negative (outside), use faster decay
                    normalized_dist = max(0.0, 1.0 + dist / tube_max_distance)  # 0 to 1
                    # Square it to make it decay faster: 1.0 at edge, 0.0 at max distance
                    probs[zone_name] = normalized_dist * normalized_dist
    
    for zone_name, polyline in zone_polygons.items():
        if zone_name.startswith('Tube_'):
            min_dist = float('inf')
            for i in range(len(polyline) - 1):
                p1 = polyline[i]
                p2 = polyline[i + 1]
                dist = _point_to_segment_distance(point, p1, p2)
                min_dist = min(min_dist, dist)
            
            if min_dist <= tube_max_distance:
                if soft_boundary:
                    probs[zone_name] = max(0.0, 1.0 - min_dist / tube_max_distance)
                else:
                    probs[zone_name] = 1.0
    
    if normalize:
        total = sum(probs.values())
        if total > 0:
            for zone in probs:
                probs[zone] /= total
    
    return probs
