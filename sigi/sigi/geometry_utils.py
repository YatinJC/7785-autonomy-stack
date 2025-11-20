#!/usr/bin/env python3
"""
Geometry utility functions for lidar processing.

This module provides common geometric operations used in wall detection
and robot localization, including angle normalization, vector operations,
and coordinate transformations.
"""

import numpy as np
from typing import Tuple, Optional


def normalize_angle(angle: float, range_type: str = 'pi') -> float:
    """
    Normalize an angle to a standard range.
    
    Args:
        angle: Angle in radians
        range_type: Either 'pi' for [-π, π] or 'twopi' for [0, 2π]
    
    Returns:
        Normalized angle in radians
    """
    if range_type == 'pi':
        # Normalize to [-π, π]
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
    elif range_type == 'twopi':
        # Normalize to [0, 2π]
        while angle < 0:
            angle += 2 * np.pi
        while angle >= 2 * np.pi:
            angle -= 2 * np.pi
    else:
        raise ValueError("range_type must be 'pi' or 'twopi'")
    
    return angle


def normalize_angle_diff(angle1: float, angle2: float, range_type: str = 'pi') -> float:
    """
    Calculate the normalized difference between two angles.
    
    Args:
        angle1: First angle in radians
        angle2: Second angle in radians
        range_type: Either 'pi' for [0, π] or 'twopi' for [0, 2π]
    
    Returns:
        Normalized angle difference
    """
    diff = abs(angle1 - angle2)
    two_pi = 2 * np.pi
    
    # Wrap to [0, 2π)
    diff = np.fmod(diff, two_pi)
    if diff < 0:
        diff += two_pi
    
    if range_type == 'pi':
        # Mirror values greater than π back into [0, π]
        if diff > np.pi:
            diff = two_pi - diff
    elif range_type == 'twopi':
        # Already wrapped into [0, 2π)
        pass
    else:
        raise ValueError("range_type must be 'pi' or 'twopi'")
    
    return diff


def are_angles_parallel(angle1: float, angle2: float, tolerance: float) -> bool:
    """
    Check if two angles represent parallel orientations.
    
    Angles are parallel if they differ by ~0° or ~180°.
    
    Args:
        angle1: First angle in radians
        angle2: Second angle in radians
        tolerance: Tolerance in radians
    
    Returns:
        True if angles are parallel within tolerance
    """
    diff = normalize_angle_diff(angle1, angle2, range_type='pi')
    return diff < tolerance or abs(diff - np.pi) < tolerance


def are_angles_perpendicular(angle1: float, angle2: float, tolerance: float) -> bool:
    """
    Check if two angles represent perpendicular orientations.
    
    Angles are perpendicular if they differ by ~90°.
    
    Args:
        angle1: First angle in radians
        angle2: Second angle in radians
        tolerance: Tolerance in radians
    
    Returns:
        True if angles are perpendicular within tolerance
    """
    diff = normalize_angle_diff(angle1, angle2, range_type='pi')
    return abs(diff - np.pi / 2) < tolerance


def rotate_vector(x: float, y: float, theta: float) -> Tuple[float, float]:
    """
    Rotate a 2D vector by an angle.
    
    Args:
        x: X component of vector
        y: Y component of vector
        theta: Rotation angle in radians (counter-clockwise)
    
    Returns:
        (rotated_x, rotated_y) tuple
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    rotated_x = x * cos_theta - y * sin_theta
    rotated_y = x * sin_theta + y * cos_theta
    
    return rotated_x, rotated_y


def normalize_vector(x: float, y: float) -> Tuple[float, float]:
    """
    Normalize a 2D vector to unit length.
    
    Args:
        x: X component of vector
        y: Y component of vector
    
    Returns:
        (normalized_x, normalized_y) tuple
    """
    norm = np.sqrt(x**2 + y**2)
    if norm < 1e-10:
        return 0.0, 0.0
    return x / norm, y / norm


def find_closest_cardinal(normal_x: float, normal_y: float) -> Optional[Tuple[int, int, str, float]]:
    """
    Find the closest cardinal direction to a given normal vector.
    
    Cardinal directions for cell walls (normal pointing inward):
    - Left wall (x=0): normal = (+1, 0)
    - Right wall (x=cell_size): normal = (-1, 0)
    - Bottom wall (y=0): normal = (0, +1)
    - Top wall (y=cell_size): normal = (0, -1)
    
    Args:
        normal_x: X component of normal vector (should be normalized)
        normal_y: Y component of normal vector (should be normalized)
    
    Returns:
        Tuple of (cardinal_x, cardinal_y, name, dot_product) or None
        where dot_product indicates alignment quality (1.0 = perfect)
    """
    cardinals = [
        (1, 0, 'left'),      # Left wall
        (-1, 0, 'right'),    # Right wall
        (0, 1, 'bottom'),    # Bottom wall
        (0, -1, 'top')       # Top wall
    ]
    
    best_match = None
    best_dot = -1.0
    
    for cx, cy, name in cardinals:
        dot = normal_x * cx + normal_y * cy
        if dot > best_dot:
            best_dot = dot
            best_match = (cx, cy, name, dot)
    
    return best_match


def project_point_onto_line(point_x: float, point_y: float, 
                           line_dir_x: float, line_dir_y: float) -> float:
    """
    Project a point onto a line direction.
    
    Args:
        point_x: X coordinate of point
        point_y: Y coordinate of point
        line_dir_x: X component of line direction vector
        line_dir_y: Y component of line direction vector
    
    Returns:
        Scalar projection value
    """
    return point_x * line_dir_x + point_y * line_dir_y


def calculate_line_segment_length(points_x: np.ndarray, points_y: np.ndarray,
                                  line_dir_x: float, line_dir_y: float) -> float:
    """
    Calculate the length of a line segment defined by projected points.
    
    Args:
        points_x: X coordinates of points
        points_y: Y coordinates of points
        line_dir_x: X component of line direction vector
        line_dir_y: Y component of line direction vector
    
    Returns:
        Length of the line segment
    """
    projections = points_x * line_dir_x + points_y * line_dir_y
    return float(np.max(projections) - np.min(projections))


def cluster_angles(angles: list, threshold: float) -> list:
    """
    Cluster angles that are close to each other.
    
    Handles angle wraparound correctly (e.g., 179° and -179° are close).
    
    Args:
        angles: List of angles in radians
        threshold: Maximum distance between angles in same cluster (radians)
    
    Returns:
        List of clusters, where each cluster is a list of angles
    """
    if len(angles) == 0:
        return []
    
    clusters = []
    
    for angle in angles:
        found_cluster = False
        
        for cluster in clusters:
            cluster_mean = np.mean(cluster)
            diff = abs(angle - cluster_mean)
            
            # Handle wraparound
            if diff > np.pi:
                diff = 2 * np.pi - diff
            
            if diff < threshold:
                cluster.append(angle)
                found_cluster = True
                break
        
        if not found_cluster:
            clusters.append([angle])
    
    return clusters


def select_best_cluster(clusters: list, prefer_zero: bool = True) -> list:
    """
    Select the best cluster from a list of clusters.
    
    Args:
        clusters: List of clusters (each cluster is a list of values)
        prefer_zero: If True, prefer cluster with mean closest to zero on tie
    
    Returns:
        The selected cluster (list of values)
    """
    if len(clusters) == 0:
        return []
    
    # Find maximum cluster size
    max_size = max(len(c) for c in clusters)
    largest_clusters = [c for c in clusters if len(c) == max_size]
    
    if len(largest_clusters) == 1:
        return largest_clusters[0]
    
    # Tie-breaker
    if prefer_zero:
        return min(largest_clusters, key=lambda c: abs(np.mean(c)))
    else:
        return largest_clusters[0]
