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
import gc
import psutil

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

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def force_cleanup():
    """Force garbage collection and clear caches"""
    gc.collect()
    # Clear any potential caches
    if hasattr(gc, 'set_debug'):
        gc.set_debug(0)


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
        
class DatasetIterator:
    """Memory-efficient dataset iterator that loads samples on demand"""
    def __init__(self, dataset_path, max_samples=None):
        self.dataset_path = dataset_path
        self.max_samples = max_samples
        self._total_samples = None
        
    def __len__(self):
        if self._total_samples is None:
            with open(self.dataset_path, 'rb') as f:
                dataset_info = pickle.load(f)
            if isinstance(dataset_info, dict) and 'data' in dataset_info:
                self._total_samples = len(dataset_info['data'])
            else:
                self._total_samples = len(dataset_info)
            
            if self.max_samples is not None:
                self._total_samples = min(self._total_samples, self.max_samples)
                
        return self._total_samples
    
    def __iter__(self):
        with open(self.dataset_path, 'rb') as f:
            dataset_info = pickle.load(f)
            
        if isinstance(dataset_info, dict) and 'data' in dataset_info:
            data = dataset_info['data']
        else:
            data = dataset_info
            
        total_samples = len(self)
        
        for i in range(total_samples):
            yield data[i]
            
        # Clean up the loaded data
        del data
        del dataset_info
        force_cleanup()


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
    if save_images:
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
    max_samples: int = None,
    batch_size: int = 1e3,  # New parameter for batch saving
):
    """
    Retarget processed dataset to specific robot hand with memory-efficient saving
    
    Args:
        processed_dataset_path: Path to processed dataset pickle file
        robot_name: Robot name (allegro, shadow, dclaw, ability, bhand, leap, svh, inspire)
        retargeting_type: Retargeting type (vector, position)
        hand_type: Hand type (right, left)
        save_images: Whether to save comparison images
        output_dir: Output directory
        max_samples: Maximum number of samples to process (None for all)
        batch_size: Number of samples to process before saving to disk
    """
    
    try:
        print(f"=== Retargeting to {robot_name.name} ===")
        print(f"Retargeting type: {retargeting_type}")
        print(f"Hand type: {hand_type}")
        print(f"Batch size: {batch_size}")
        
        # Load processed dataset
        print(f"\nLoading processed dataset from: {processed_dataset_path}")
        with open(processed_dataset_path, 'rb') as f:
            dataset_info = pickle.load(f)
        
        # # Handle both old format (direct list) and new format (with metadata)
        # if isinstance(dataset_info, dict) and 'data' in dataset_info:
        #     processed_data = dataset_info['data']
        #     metadata = dataset_info.get('metadata', {})
        #     print(f"Dataset metadata found:")
        #     for key, value in metadata.items():
        #         print(f"  {key}: {value}")
        # else:
        #     # Old format - direct list
        #     processed_data = dataset_info
        #     print("Dataset loaded (old format without metadata)")
        
        processed_data = DatasetIterator(processed_dataset_path, max_samples)
        total_samples = len(processed_data)
        if max_samples is not None:
            total_samples = min(total_samples, max_samples)
            processed_data = processed_data[:total_samples]
        
        print(f"Dataset loaded: {total_samples} samples")
        
        # # Count sample types
        # original_count = sum(1 for d in processed_data if not d.get('is_augmented', False))
        # augmented_count = sum(1 for d in processed_data if d.get('is_augmented', False))
        # print(f"Original samples: {original_count}")
        # print(f"Augmented samples: {augmented_count}")
        
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
        
        # Set up output paths
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        final_output_path = output_path / f"retargeted_{robot_name.name}_{retargeting_type.name}_{hand_type.name}.pkl"
        temp_dir = output_path / "temp_batches"
        temp_dir.mkdir(exist_ok=True)
        
        # Process samples in batches
        print(f"\nProcessing {total_samples} samples in batches of {batch_size}...")
        batch_files = []
        current_batch = []
        sample_iter = iter(processed_data)
        
        for i in tqdm(range(total_samples), desc="Retargeting samples"):
            # sample_data = processed_data[i]
            sample_data = next(sample_iter)

            # Create new retargeting instance for each sample
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
            
            current_batch.append(result)
            del retargeting  # Free memory
            
            # Save batch when it reaches batch_size or when it's the last sample
            if len(current_batch) >= batch_size or i == total_samples - 1:
                batch_num = len(batch_files)
                batch_file = temp_dir / f"batch_{batch_num}.pkl"
                
                with open(batch_file, "wb") as f:
                    pickle.dump(current_batch, f)
                
                batch_files.append(batch_file)
                print(f"Saved batch {batch_num} with {len(current_batch)} samples")
                
                # Clear the current batch from memory
                current_batch = []
                force_cleanup()
        
        # Clean up rendering resources
        if save_images:
            print("Cleaning up rendering resources...")
            reset_sapien_scene(scene)
            destroy_sapien_objects(scene, cam, robot, config)
            cleanup_sapien_resources()
        
        # Option 1: Memory-efficient streaming merge (recommended for very large datasets)
        print(f"\nCombining {len(batch_files)} batch files into final dataset...")
        
        # Use streaming approach to avoid loading all data into memory at once
        with open(final_output_path, "wb") as final_file:
            # Initialize empty list for pickle
            all_results = []
            
            # Load and merge batches one by one
            for batch_file in tqdm(batch_files, desc="Merging batches"):
                with open(batch_file, "rb") as f:
                    batch_data = pickle.load(f)
                    all_results.extend(batch_data)
                
                # Save intermediate state if memory usage gets too high
                # You can add memory monitoring here if needed
                
                # Remove processed batch file immediately
                batch_file.unlink()
            
            # Save final combined results
            pickle.dump(all_results, final_file)
        
        # Clean up temporary directory
        if temp_dir.exists():
            temp_dir.rmdir()
        
        print(f"\n✓ Retargeting completed successfully!")
        print(f"Results saved to: {final_output_path}")
        
        if save_images:
            images_dir = output_path / "images"
            print(f"Images saved to: {images_dir}")
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"Total processed samples: {len(all_results)}")
        print(f"Robot: {robot_name.name}")
        print(f"Retargeting type: {retargeting_type.name}")
        print(f"Hand type: {hand_type.name}")
        
        # Clear results from memory
        del all_results
        
    except Exception as e:
        print(f"\n❌ Retargeting failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temporary files in case of error
        temp_dir = Path(output_dir) / "temp_batches"
        if temp_dir.exists():
            for batch_file in temp_dir.glob("*.pkl"):
                batch_file.unlink()
            temp_dir.rmdir()
        
        return 1
    
    return 0


if __name__ == "__main__":
    tyro.cli(main)