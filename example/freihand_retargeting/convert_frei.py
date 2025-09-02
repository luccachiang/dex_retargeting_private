#!/usr/bin/env python3
"""
Retarget processed hand pose dataset to specific robot hands
This script reads the preprocessed and augmented dataset and retargets 
it to a specific robot embodiment.
"""
from pathlib import Path
import numpy as np
import sapien
from sapien.asset import create_dome_envmap
import matplotlib.pyplot as plt
import tyro
from tqdm import tqdm
import pickle
from scipy.spatial.transform import Rotation as R
import copy
import cv2

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

# Import sapien utilities
import sapien.core as sapien
sapien.set_log_level("error") # filter out warnings
from sapien_utils import (
    reset_sapien_scene,
    destroy_sapien_objects,
    create_clean_sapien_scene,
    safe_render_robot,
    cleanup_sapien_resources
)

import sys
sys.path.append(str(Path(__file__).parent / "freihand"))
from freihand.utils.fh_utils import plot_hand


def retarget_joint_pos_to_qpos(joint_pos, retargeting):
    """Convert joint_pos to robot qpos"""
    retargeting_type = retargeting.optimizer.retargeting_type
    indices = retargeting.optimizer.target_link_human_indices
    
    if retargeting_type == "POSITION":
        ref_value = joint_pos[indices, :]
    else:
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
    
    qpos = retargeting.retarget(ref_value) # may have joints more than only actuated joints
    # usually vector/dexpilot have same setting, and position has a different setting
    # use retargeting.optimizer.__dict__ to inspect
    # if not only save actuated joints here, we need to process later in the dataloader
    return qpos


def load_robot_in_scene(scene, config_path):
    """Load robot in scene"""
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
    elif "xhand" in robot_name:
        loader.scale = 1.5

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)
    robot = loader.load(filepath)

    # Set pose
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
    elif "xhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))

    return robot, config


def create_comparison_image(original_image, detected_keypoints_2d, rendered_robot, image_name, save_path=None):
    """Create comparison image of original + rendered robot"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if original_image is not None:
        original_image = cv2.cvtColor(cv2.imread(original_image), cv2.COLOR_BGR2RGB)
    
    # Original image (if available)
    if original_image is not None:
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12)
    else:
        axes[0].text(0.5, 0.5, 'No Image\n(Augmented Sample)', 
                    ha='center', va='center', fontsize=14, 
                    transform=axes[0].transAxes)
        axes[0].set_title('Augmented Sample', fontsize=12)
    axes[0].axis('off')
    
    # Original + detected keypoints (if image is available)
    if original_image is not None:
        axes[1].imshow(original_image)
        if detected_keypoints_2d is not None and len(detected_keypoints_2d) > 0:
            plot_hand(axes[1], detected_keypoints_2d, color_fixed='red', linewidth=2, order='uv')
        # Add index labels beside each keypoint
        for idx, (x, y) in enumerate(detected_keypoints_2d):
            axes[1].text(
                x + 2, y + 2,  # small offset so text doesn't overlap point
                str(idx),
                fontsize=8,
                color='yellow',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
            )
        axes[1].set_title('Original + GT Keypoints', fontsize=12, color='red')
    else:
        axes[1].text(0.5, 0.5, 'No Image\n(Augmented Sample)', 
                    ha='center', va='center', fontsize=14, 
                    transform=axes[1].transAxes)
        axes[1].set_title('Augmented Sample', fontsize=12)
    axes[1].axis('off')
    
    # Rendered robot hand
    axes[2].imshow(rendered_robot)
    axes[2].set_title('Rendered Robot Hand (GT)', fontsize=12, color='blue')
    axes[2].axis('off')
    
    # Add image info
    fig.text(0.02, 0.02, f'Image: {image_name}', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        # print(f"Comparison image saved: {save_path}")
    else:
        plt.show()


def process_single_sample(
    sample_data: dict,
    retargeting,
    robot=None,
    scene=None,
    cam=None,
    retargeting_joint_names=None,
    output_dir=None,
    save_images: bool = False,
    hand_type: HandType = HandType.right,
):
    """Process a single sample: retargeting and optional rendering"""
    
    # Extract data
    joint_pos = sample_data['joint_pos']
    # do finger scale, code adapted from lai
    joint_pos[1:5] *= 1.02
    joint_pos[5:9] *= 1.02
    joint_pos[17:21] *= 1.2
    keypoint_2d = sample_data['keypoint_2d']
    # add right2left mirror
    if hand_type == HandType.left:
        raise NotImplementedError("Left hand mirroring doesn't work so far.")
        joint_pos = copy.deepcopy(joint_pos)
        joint_pos[:, 0] = -joint_pos[:, 0]
        if keypoint_2d is not None:
            keypoint_2d = copy.deepcopy(keypoint_2d)
            keypoint_2d[:, 0] = -keypoint_2d[:, 0]
    image = sample_data.get('image', None)
    image_id = sample_data['image_id']
    
    # Retarget to robot
    # Each time we init the retargeting, it will have a last_qpos. 
    # If the qpos is not a sequential trajectory, like here we use a dataset, call reset to remove the influence of history qpos.
    # retargeting.reset()
    # TODO reset has different results from init a new instance
    qpos = retarget_joint_pos_to_qpos(joint_pos, retargeting)

    # Create result
    result = {
        'image_id': image_id,
        'joint_pos': joint_pos, # human hand (21, 3)
        'qpos': qpos,
        'keypoint_2d': keypoint_2d, # (21, 2)
        'is_augmented': sample_data.get('is_augmented', False),
        'split': sample_data.get('split', 'unknown')
    }
    
    # Render robot if requested
    if save_images and robot is not None and scene is not None:
        rendered_robot = safe_render_robot(scene, cam, robot, qpos, retargeting_joint_names)
        
        # Create comparison image
        image_name = f"{image_id:08d}"
        if sample_data.get('is_augmented', False):
            image_name += "_aug"
        
        comparison_path = Path(output_dir) / "images" / f"{image_name}_comparison.png"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        
        create_comparison_image(image, keypoint_2d, rendered_robot, image_name, str(comparison_path))
    
    return result


def main(
    processed_dataset_path: str,
    robot_name: RobotName = RobotName.allegro,
    retargeting_type: RetargetingType = RetargetingType.vector,
    hand_type: HandType = HandType.right,
    save_images: bool = False,
    output_dir: str = "retargeted_dataset",
    max_samples: int = None
):
    """
    Retarget processed dataset to specific robot hand
    
    Args:
        processed_dataset_path: Path to processed dataset pickle file
        robot_name: Robot name (allegro, shadow, dclaw, ability, bhand, leap, svh, inspire)
        retargeting_type: Retargeting type (vector, position)
        hand_type: Hand type (right, left)
        save_images: Whether to save comparison images
        output_dir: Output directory
        max_samples: Maximum number of samples to process (None for all)
    """
    
    try:
        print(f"=== Retargeting to {robot_name.name} ===")
        print(f"Retargeting type: {retargeting_type}")
        print(f"Hand type: {hand_type}")
        
        # Load processed dataset
        print(f"\nLoading processed dataset from: {processed_dataset_path}")
        with open(processed_dataset_path, 'rb') as f:
            dataset_info = pickle.load(f)
        
        # Handle both old format (direct list) and new format (with metadata)
        if isinstance(dataset_info, dict) and 'data' in dataset_info:
            processed_data = dataset_info['data']
            metadata = dataset_info.get('metadata', {})
            print(f"Dataset metadata found:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
            # Old format - direct list
            processed_data = dataset_info
            print("Dataset loaded (old format without metadata)")
        
        total_samples = len(processed_data)
        if max_samples is not None:
            total_samples = min(total_samples, max_samples)
            processed_data = processed_data[:total_samples]
        
        print(f"Dataset loaded: {total_samples} samples")
        
        # Count sample types
        original_count = sum(1 for d in processed_data if not d.get('is_augmented', False))
        augmented_count = sum(1 for d in processed_data if d.get('is_augmented', False))
        print(f"Original samples: {original_count}")
        print(f"Augmented samples: {augmented_count}")
        
        # Set up retargeting
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
        print(config_path)
        robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        
        
        # Set up rendering if needed
        robot = None
        scene = None
        cam = None
        
        if save_images:
            print("Setting up rendering...")
            scene, cam = create_clean_sapien_scene()
            robot, config = load_robot_in_scene(scene, config_path)
        
        # Process all samples
        print(f"\nProcessing {total_samples} samples...")
        results = []
        
        for i in tqdm(range(total_samples), desc="Retargeting samples"):
            sample_data = processed_data[i]

            # need to new a retargeting in each iter
            retargeting = RetargetingConfig.load_from_file(config_path).build()
            retargeting_joint_names = retargeting.optimizer.robot.dof_joint_names
            
            result = process_single_sample(
                sample_data=sample_data,
                retargeting=retargeting,
                robot=robot,
                scene=scene,
                cam=cam,
                retargeting_joint_names=retargeting_joint_names,
                output_dir=output_dir,
                save_images=save_images,
                hand_type=hand_type,
            )
            
            results.append(result)
        
        # Clean up rendering resources
        if save_images:
            print("Cleaning up rendering resources...")
            reset_sapien_scene(scene)
            destroy_sapien_objects(scene, cam, robot, config)
            cleanup_sapien_resources()
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_pkl_path = output_path / f"retargeted_{robot_name.name}_{retargeting_type.name}_{hand_type.name}.pkl"
        
        with open(output_pkl_path, "wb") as f:
            pickle.dump(results, f)
        
        print(f"\n✓ Retargeting completed successfully!")
        print(f"Results saved to: {output_pkl_path}")
        
        if save_images:
            images_dir = output_path / "images"
            print(f"Images saved to: {images_dir}")
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"Total processed samples: {len(results)}")
        print(f"Robot: {robot_name.name}")
        print(f"Retargeting type: {retargeting_type.name}")
        print(f"Hand type: {hand_type.name}")
        
    except Exception as e:
        print(f"\n❌ Retargeting failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    tyro.cli(main)