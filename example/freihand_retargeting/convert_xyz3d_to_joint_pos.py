"""
Convert FreiHAND xyz_3d to SingleHandDetector joint_pos format

This module provides functions to convert 3D keypoints from the FreiHAND dataset
to the joint_pos format output by MediaPipe SingleHandDetector.
"""

import numpy as np
from typing import Literal

# Define coordinate transformation matrices (same as in SingleHandDetector)
OPERATOR2MANO_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])

OPERATOR2MANO_LEFT = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
])


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Estimate the hand coordinate frame from 3D keypoints (same implementation as in SingleHandDetector)
    
    Args:
        keypoint_3d_array: 3D keypoints, shape (21, 3), already centered at the wrist
        
    Returns:
        3x3 rotation matrix representing the hand coordinate frame
    """
    assert keypoint_3d_array.shape == (21, 3)
    
    # Select keypoints: wrist (0), index MCP (5), middle MCP (9)
    points = keypoint_3d_array[[0, 5, 9], :]
    
    # Compute vector from palm to middle finger MCP
    x_vector = points[0] - points[2]
    
    # Fit normal vector using SVD
    points_centered = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points_centered)
    
    normal = v[2, :]
    
    # Gram-Schmidt orthogonalization
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    
    # Ensure the vector from pinky to index aligns with z axis direction
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    
    frame = np.stack([x, normal, z], axis=1)
    return frame


def convert_xyz3d_to_joint_pos(
    xyz_3d: np.ndarray, 
    hand_type: Literal["Right", "Left"] = "Right"
) -> np.ndarray:
    """
    Convert FreiHAND xyz_3d to SingleHandDetector joint_pos format
    
    This function replicates the full transformation process in SingleHandDetector.detect():
    1. Normalize coordinates to the wrist as origin
    2. Estimate hand coordinate frame
    3. Apply coordinate transformation matrix
    
    Args:
        xyz_3d: FreiHAND 3D keypoints, shape (21, 3)
        hand_type: Hand type, "Right" or "Left"
        
    Returns:
        joint_pos: Transformed 3D keypoints, same format as SingleHandDetector output
    """
    xyz_3d = np.array(xyz_3d)
    assert xyz_3d.shape == (21, 3), f"Expected shape (21, 3), got {xyz_3d.shape}"
    
    # Step 1: Normalize to wrist as origin (same as MediaPipe SingleHandDetector)
    # Corresponding code: keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
    keypoint_3d_array = xyz_3d - xyz_3d[0:1, :]
    
    # Step 2: Estimate hand coordinate frame
    # Corresponding code: mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
    mediapipe_wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)
    
    # Step 3: Apply coordinate transformation
    # Corresponding code: joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano
    operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
    joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ operator2mano
    
    return joint_pos


def simple_convert_xyz3d_to_joint_pos(xyz_3d: np.ndarray) -> np.ndarray:
    """
    Simplified conversion function (only wrist normalization)
    
    This is the simplified method used in convert_freihand_to_retargeting.py,
    which only performs wrist normalization and does not include coordinate frame transformation.
    
    Args:
        xyz_3d: FreiHAND 3D keypoints, shape (21, 3)
        
    Returns:
        joint_pos: 3D keypoints with wrist as origin
    """
    xyz_3d = np.array(xyz_3d)
    assert xyz_3d.shape == (21, 3), f"Expected shape (21, 3), got {xyz_3d.shape}"
    
    # Only perform wrist normalization
    joint_pos = xyz_3d - xyz_3d[0:1, :]
    
    return joint_pos


def compare_conversion_methods(xyz_3d: np.ndarray, hand_type: Literal["Right", "Left"] = "Right"):
    """
    Compare the results of different conversion methods
    
    Args:
        xyz_3d: FreiHAND 3D keypoints
        hand_type: Hand type
    """
    print("=== Comparison of xyz_3d to joint_pos conversion methods ===")
    
    # Original data info
    print(f"Original xyz_3d shape: {xyz_3d.shape}")
    print(f"Original xyz_3d range: [{xyz_3d.min():.3f}, {xyz_3d.max():.3f}]")
    print(f"Wrist position (xyz_3d[0]): {xyz_3d[0]}")
    
    # Method 1: Full transformation (simulate SingleHandDetector)
    joint_pos_full = convert_xyz3d_to_joint_pos(xyz_3d, hand_type)
    print(f"\nAfter full transformation, joint_pos range: [{joint_pos_full.min():.3f}, {joint_pos_full.max():.3f}]")
    print(f"After full transformation, wrist position: {joint_pos_full[0]}")
    
    # Method 2: Simplified transformation (wrist normalization only)
    joint_pos_simple = simple_convert_xyz3d_to_joint_pos(xyz_3d)
    print(f"\nAfter simplified transformation, joint_pos range: [{joint_pos_simple.min():.3f}, {joint_pos_simple.max():.3f}]")
    print(f"After simplified transformation, wrist position: {joint_pos_simple[0]}")
    
    # Compare differences
    diff = np.linalg.norm(joint_pos_full - joint_pos_simple, axis=1)
    print(f"\nDifference between the two methods:")
    print(f"Mean difference: {diff.mean():.3f}")
    print(f"Max difference: {diff.max():.3f}")
    print(f"Number of keypoints with difference > 0.01: {np.sum(diff > 0.01)}")
    
    return joint_pos_full, joint_pos_simple


if __name__ == "__main__":
    # Example: create some test data
    np.random.seed(42)
    
    # Simulate FreiHAND xyz_3d data
    xyz_3d_example = np.random.randn(21, 3) * 0.1
    xyz_3d_example[0] = [0.5, 0.3, 0.2]  # Set wrist position
    
    print("=== Example of xyz_3d to joint_pos conversion ===")
    joint_pos_full, joint_pos_simple = compare_conversion_methods(xyz_3d_example, "Right")
    
    print(f"\nUsage suggestions:")
    print(f"- If you need output fully consistent with SingleHandDetector, use convert_xyz3d_to_joint_pos()")
    print(f"- If you only need wrist-normalized coordinates, use simple_convert_xyz3d_to_joint_pos()")
    print(f"- For robot retargeting applications, the full transformation is usually more accurate")
