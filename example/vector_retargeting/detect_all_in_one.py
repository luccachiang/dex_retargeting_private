import pickle
from pathlib import Path
import glob
import os

import cv2
import tqdm
import tyro
import numpy as np

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
    ROBOT_NAMES,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector


def setup_robot_retargeting(robot_name: RobotName, hand_type: HandType, robot_dir: str):
    """
    Setup retargeting for a specific robot hand configuration
    
    Args:
        robot_name: Robot name
        hand_type: Hand type (left/right)
        robot_dir: Robot assets directory path
        
    Returns:
        Retargeting object and metadata, or None if failed
    """
    try:
        config_path = get_default_config_path(robot_name, RetargetingType.dexpilot, hand_type)
        if config_path is None or not config_path.exists():
            return None, None
            
        RetargetingConfig.set_default_urdf_dir(robot_dir)
        retargeting = RetargetingConfig.load_from_file(config_path).build()
        
        meta_data = {
            'robot_name': robot_name.name,
            'hand_type': hand_type.name,
            'retargeting_type': 'dexpilot',
            'config_path': str(config_path),
            'dof': len(retargeting.optimizer.robot.dof_joint_names),
            'joint_names': retargeting.optimizer.robot.dof_joint_names,
        }
        
        return retargeting, meta_data
        
    except Exception as e:
        print(f"Warning: Failed to setup {robot_name.name}_{hand_type.name}: {str(e)}")
        return None, None


def process_single_image_all_robots(
    image_path: Path,
    detector_right: SingleHandDetector,
    detector_left: SingleHandDetector,
    robot_retargeting_configs: dict
) -> dict:
    """
    Process a single image for all robot hand configurations
    
    Args:
        image_path: Path to the image file
        detector_right: Right hand detector
        detector_left: Left hand detector  
        robot_retargeting_configs: Dictionary of all robot retargeting setups
        
    Returns:
        Dictionary containing all robot qpos data for this image
    """
    # Read image
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None
    
    # Convert color format (BGR -> RGB)
    rgb = frame[..., ::-1]
    
    # Detect both hands
    num_box_right, joint_pos_right, _, _ = detector_right.detect(rgb)
    num_box_left, joint_pos_left, _, _ = detector_left.detect(rgb)
    
    image_result = {
        'image_path': str(image_path),
        'image_name': image_path.name,
        'hand_detected': {
            'right': num_box_right > 0,
            'left': num_box_left > 0
        },
        'robots': {}
    }
    
    # Process all robot configurations
    for robot_config_key, (retargeting, meta_data) in robot_retargeting_configs.items():
        if retargeting is None:
            continue
            
        robot_name, hand_type = robot_config_key.split('_')
        
        # Select appropriate hand detection result
        joint_pos = None
        if hand_type == HandType.right:
            if num_box_right == 0:
                qpos = None
            else:
                joint_pos = joint_pos_right
        else:  # HandType.left
            if num_box_left == 0:
                qpos = None
            else:
                joint_pos = joint_pos_left
        
        # Execute retargeting if hand detected
        if joint_pos is not None:
            try:
                # Prepare retargeting input
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                
                if retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = (
                        joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    )
                
                # Execute retargeting
                qpos = retargeting.retarget(ref_value)
                
            except Exception as e:
                print(f"Warning: Retargeting failed for {robot_config_key} on {image_path.name}: {str(e)}")
                qpos = None
        else:
            qpos = None
        
        # Store result
        image_result['robots'][robot_config_key] = {
            'qpos': qpos.tolist() if qpos is not None else None,
            'success': qpos is not None
        }
    
    return image_result


def retarget_images_all_robots(
    input_path: str, 
    output_path: str,
    image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
):
    """
    Process images for all robot hands using dexpilot mode
    Data structure: each image contains qpos for all robots
    
    Args:
        input_path: Input path, can be a single image or folder containing images
        output_path: Output data file path (.pickle format)
        image_extensions: Supported image formats
    """
    input_path = Path(input_path)
    
    # Get all image files
    if input_path.is_file():
        # Single image
        if input_path.suffix.lower() in image_extensions:
            image_files = [input_path]
        else:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
    elif input_path.is_dir():
        # Folder
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(input_path / f"*{ext}")))
            image_files.extend(glob.glob(str(input_path / f"*{ext.upper()}")))
        image_files = [Path(f) for f in sorted(image_files)]
        if not image_files:
            raise ValueError(f"No supported image files found in folder {input_path}")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    print(f"Found {len(image_files)} images")
    
    # Setup robot directory
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    
    # Initialize detectors for both hands
    detector_right = SingleHandDetector(hand_type="Right", selfie=False)
    detector_left = SingleHandDetector(hand_type="Left", selfie=False)
    
    # Setup all robot retargeting configurations
    print("Setting up robot configurations...")
    robot_retargeting_configs = {}
    robot_metadata = {}
    
    for robot_name in ROBOT_NAMES:
        # Setup right hand
        retargeting_right, meta_right = setup_robot_retargeting(
            robot_name, HandType.right, str(robot_dir)
        )
        config_key_right = f"{robot_name.name}_right"
        robot_retargeting_configs[config_key_right] = (retargeting_right, meta_right)
        if meta_right:
            robot_metadata[config_key_right] = meta_right
        
        # Setup left hand
        retargeting_left, meta_left = setup_robot_retargeting(
            robot_name, HandType.left, str(robot_dir)
        )
        config_key_left = f"{robot_name.name}_left"
        robot_retargeting_configs[config_key_left] = (retargeting_left, meta_left)
        if meta_left:
            robot_metadata[config_key_left] = meta_left
    
    # Process all images
    print("Processing images...")
    images_data = []
    
    for image_file in tqdm.tqdm(image_files, desc="Processing images"):
        image_result = process_single_image_all_robots(
            image_file, detector_right, detector_left, robot_retargeting_configs
        )
        
        if image_result is not None:
            images_data.append(image_result)
    
    # Organize final output data
    output_data = {
        'metadata': {
            'input_info': {
                'input_path': str(input_path),
                'total_images': len(image_files),
                'successfully_processed_images': len(images_data),
                'processing_mode': 'all_robots_dexpilot_per_image'
            },
            'robot_configurations': robot_metadata,
            'data_structure_info': {
                'description': 'Each image contains qpos data for all robot hand configurations',
                'image_index': 'Use images_data[i] to access data for image i',
                'robot_access': 'Use images_data[i]["robots"]["robot_name_hand_type"]["qpos"] to get specific robot qpos'
            }
        },
        'data': images_data
    }
    
    # Calculate summary statistics
    total_combinations = len(robot_metadata)
    successful_detections = 0
    total_possible = len(images_data) * total_combinations
    
    for image_data in images_data:
        for robot_key, robot_result in image_data['robots'].items():
            if robot_result['success']:
                successful_detections += 1
    
    output_data['metadata']['summary'] = {
        'total_robot_hand_combinations': total_combinations,
        'total_images_processed': len(images_data),
        'total_possible_detections': total_possible,
        'successful_detections': successful_detections,
        'success_rate': successful_detections / total_possible if total_possible > 0 else 0
    }
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("wb") as f:
        pickle.dump(output_data, f)
    
    # Print summary
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {len(images_data)}/{len(image_files)} images")
    print(f"Total robot configurations: {total_combinations}")
    print(f"Successful detections: {successful_detections}/{total_possible} ({successful_detections/total_possible*100:.1f}%)")
    print(f"Results saved to: {output_path}")
    
    # Print data access example
    print(f"\n=== Data Access Example ===")
    print(f"# Load data")
    print(f"with open('{output_path}', 'rb') as f:")
    print(f"    data = pickle.load(f)")
    print(f"")
    print(f"# Access first image's allegro right hand qpos")
    print(f"qpos = data['data'][0]['robots']['allegro_right']['qpos']")
    print(f"")
    print(f"# Check if detection was successful")
    print(f"success = data['data'][0]['robots']['allegro_right']['success']")


def main(
    input_path: str,
    output_path: str,
):
    """
    Detects human hand pose from images and translates to all robot pose trajectories using dexpilot mode.
    Data is organized by image: each image contains qpos for all robot hand configurations.

    Args:
        input_path: The input path, can be a single image file or a folder containing images.
        output_path: The file path for the output data in .pickle format.
    """
    retarget_images_all_robots(input_path, output_path)


if __name__ == "__main__":
    tyro.cli(main)
