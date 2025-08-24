#!/usr/bin/env python3
"""
Script for single image hand detection, retargeting, and rendering.
"""

from pathlib import Path
import numpy as np
import cv2
import sapien
from sapien.asset import create_dome_envmap
import matplotlib.pyplot as plt
import tyro

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

from single_hand_detector import SingleHandDetector
from convert_xyz3d_to_joint_pos import convert_xyz3d_to_joint_pos

# Import sapien utilities
from sapien_utils import (
    reset_sapien_scene,
    destroy_sapien_objects,
    create_clean_sapien_scene,
    safe_render_robot,
    cleanup_sapien_resources
)

import sys
sys.path.append(str(Path(__file__).parent / "freihand"))
from freihand.utils.fh_utils import read_img, plot_hand, projectPoints, load_db_annotation

def load_image(image_path):
    """Load image."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    annotations = list(load_db_annotation(Path(image_path).parent.parent.parent, 'training'))
    anno = [annotations[i] for i in range(len(annotations)) if i == int(Path(image_path).stem)][0]
    K, mano_params, xyz_3d = anno
    K = np.array(K)
    xyz_3d = np.array(xyz_3d)
    keypoint_2d = projectPoints(xyz_3d, K)
    joint_pos = convert_xyz3d_to_joint_pos(xyz_3d, 'Right')

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image, image_rgb, joint_pos, keypoint_2d


def detect_hand(image_rgb, hand_type="Right"):
    """Detect hand using HandDetector."""
    print(f"Detecting hand using HandDetector (type: {hand_type})...")
    
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)
    
    # Detect hand
    num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(image_rgb)
    
    if num_box == 0:
        raise ValueError("No hand detected")
    
    print(f"Hand detected successfully, number of joints: {len(joint_pos)}")
    
    return detector, joint_pos, keypoint_2d, mediapipe_wrist_rot


def retarget_joint_pos_to_qpos(joint_pos, retargeting):
    """Convert joint_pos to robot qpos."""
    retargeting_type = retargeting.optimizer.retargeting_type
    indices = retargeting.optimizer.target_link_human_indices
    
    if retargeting_type == "POSITION":
        ref_value = joint_pos[indices, :]
    else:
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
    
    qpos = retargeting.retarget(ref_value)
    return qpos


def load_robot_in_scene(scene, config_path):
    """Load robot in the scene."""
    config = RetargetingConfig.load_from_file(config_path)
    
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    
    # Set scale
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)
    robot = loader.load(filepath)

    # Set position
    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))
    elif "inspire" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))

    return robot, config


def create_comparison_image(original_image, detected_keypoints_2d, rendered_robot, image_name, save_path=None):
    """Create a comparison image of original + rendered (following enhanced_compare_freihand_detection.py style)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Original + detected keypoints (using plot_hand function)
    axes[1].imshow(original_image)
    if detected_keypoints_2d is not None and len(detected_keypoints_2d) > 0:
        # Draw hand keypoints in the same way as enhanced_compare_freihand_detection.py
        plot_hand(axes[1], detected_keypoints_2d, color_fixed='red', linewidth=2, order='uv')
    axes[1].set_title('Original + Detected Keypoints', fontsize=12, color='red')
    axes[1].axis('off')
    
    # Rendered robot hand
    axes[2].imshow(rendered_robot)
    axes[2].set_title('Rendered Robot Hand', fontsize=12, color='blue')
    axes[2].axis('off')
    
    # Add image info
    fig.text(0.02, 0.02, f'Image: {image_name}', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison image saved: {save_path}")
    else:
        plt.show()


def process_single_image(
    image_path: str,
    robot_name: RobotName = RobotName.allegro,
    retargeting_type: RetargetingType = RetargetingType.vector,
    hand_type: HandType = HandType.right,
    output_dir: str = "single_image_output",
    hand_detector_type: str = "Right"
):
    """Process a single image: detect hand, retarget, render."""
    
    print(f"=== Single Image Processing ===")
    print(f"Input image: {image_path}")
    print(f"Robot: {robot_name}")
    print(f"Retargeting type: {retargeting_type}")
    print(f"Hand type: {hand_type}")
    print(f"HandDetector type: {hand_detector_type}")
    
    # Set output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    scene, cam = create_clean_sapien_scene()
    robot, config = load_robot_in_scene(scene, config_path)
    
    # 1. Load image
    print("\n1. Loading image...")
    image, image_rgb, joint_pos_gt, keypoint_2d_gt = load_image(image_path)
    image_name = Path(image_path).stem
    print(f"Image shape: {image.shape}")
    
    # 2. Detect hand
    print("\n2. Detecting hand...")
    detector, joint_pos, keypoint_2d, mediapipe_wrist_rot = detect_hand(image_rgb, hand_detector_type)
    print(f"Detected joint position range: [{joint_pos.min():.4f}, {joint_pos.max():.4f}]")
    
    # Convert detected keypoints to numpy array (following enhanced_compare_freihand_detection.py style)
    img_height, img_width = image_rgb.shape[:2]
    detected_keypoints_2d = detector.parse_keypoint_2d(keypoint_2d, (img_height, img_width))
    
    # 3. Set retargeting
    print("\n3. Setting retargeting...")
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    
    # 4. Perform retargeting
    print("\n4. Performing retargeting...")
    qpos = retarget_joint_pos_to_qpos(joint_pos, retargeting)
    # qpos_gt = retarget_joint_pos_to_qpos(joint_pos_gt, retargeting)
    print(f"Retargeting result range: [{qpos.min():.4f}, {qpos.max():.4f}]")
    
    # 5. Render robot
    print("\n5. Rendering robot...")
    
    # Get joint names
    retargeting_joint_names = retargeting.optimizer.robot.dof_joint_names
    
    # Render robot
    rendered_robot = safe_render_robot(scene, cam, robot, qpos, retargeting_joint_names)

    # rendered_robot_gt = safe_render_robot(scene, cam, robot, qpos_gt, retargeting_joint_names)
    
    # Clean up sapien resources
    reset_sapien_scene(scene)
    destroy_sapien_objects(scene, cam, robot, config)
    cleanup_sapien_resources()

    # 6. Create comparison image and save
    print("\n6. Creating comparison image...")
    comparison_path = output_path / f"{image_name}_comparison.png"
    create_comparison_image(image_rgb, detected_keypoints_2d, rendered_robot, image_name, str(comparison_path))
    
    # 6.1 Create comparison image (using freihand's plot_hand function)
    # comparison_path = output_path / f"{image_name}_comparison_freihand.png"
    # create_comparison_image(image_rgb, keypoint_2d_gt, rendered_robot_gt, image_name, str(comparison_path))

    # 7. Save rendered image only
    render_path = output_path / f"{image_name}_robot_hand.png"
    cv2.imwrite(str(render_path), cv2.cvtColor(rendered_robot, cv2.COLOR_RGB2BGR))
    print(f"Robot hand rendering saved: {render_path}")
    
    print(f"\n=== Processing Complete ===")
    print(f"Output directory: {output_path}")
    print(f"Comparison image: {comparison_path}")
    print(f"Rendered image: {render_path}")
    
    return {
        'joint_pos': joint_pos,
        'qpos': qpos,
        'keypoint_2d': keypoint_2d,
        'rendered_robot': rendered_robot,
        'comparison_path': str(comparison_path),
        'render_path': str(render_path)
    }


def main(
    image_path: str,
    robot_name: RobotName = RobotName.allegro,
    retargeting_type: RetargetingType = RetargetingType.vector,
    hand_type: HandType = HandType.right,
    hand_detector_type: str = "Right",
    output_dir: str = "single_image_output"
):
    """
    Single image hand detection, retargeting, and rendering.
    
    Args:
        image_path: Input image path
        robot_name: Robot name (allegro, shadow, dclaw, ability, bhand, leap, svh, inspire)
        retargeting_type: Retargeting type (vector, position)
        hand_type: Hand type (right, left)
        hand_detector_type: HandDetector hand type (Right, Left)
        output_dir: Output directory
    """
    
    try:
        result = process_single_image(
            image_path=image_path,
            robot_name=robot_name,
            retargeting_type=retargeting_type,
            hand_type=hand_type,
            output_dir=output_dir,
            hand_detector_type=hand_detector_type
        )
        print("\n✓ Processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    tyro.cli(main)
