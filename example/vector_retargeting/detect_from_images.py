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
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector


def retarget_images(
    retargeting: SeqRetargeting, 
    input_path: str, 
    output_path: str, 
    config_path: str,
    image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
):
    """
    Process images or image folder for hand retargeting
    
    Args:
        retargeting: Retargeting object
        input_path: Input path, can be a single image or folder containing images
        output_path: Output data file path (.pickle format)
        config_path: Configuration file path
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

    data = []
    
    detector = SingleHandDetector(hand_type="Right", selfie=False)
    print(f"Found {len(image_files)} images, starting processing...")
    
    with tqdm.tqdm(total=len(image_files)) as pbar:
        for image_file in image_files:
            # Read image
            frame = cv2.imread(str(image_file))
            if frame is None:
                print(f"Warning: Could not read image {image_file}")
                pbar.update(1)
                continue
            
            # Convert color format (BGR -> RGB)
            rgb = frame[..., ::-1]
            
            # Detect hand
            num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(rgb)
            
            if num_box == 0:
                pbar.update(1)
                continue
            
            # Prepare retargeting input
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = (
                    joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                )
            
            # Execute retargeting
            qpos = retargeting.retarget(ref_value)
            data.append(qpos)
            pbar.update(1)
    
    if not data:
        print("Error: No images were successfully processed")
        return
    
    # Save results - same format as video version
    meta_data = dict(
        config_path=config_path,
        dof=len(retargeting.optimizer.robot.dof_joint_names),
        joint_names=retargeting.optimizer.robot.dof_joint_names,
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("wb") as f:
        pickle.dump(dict(data=data, meta_data=meta_data), f)
    
    retargeting.verbose()


def main(
    robot_name: RobotName,
    input_path: str,
    output_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
):
    """
    Detects the human hand pose from images and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        input_path: The input path, can be a single image file or a folder containing images.
        output_path: The file path for the output data in .pickle format.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
    """

    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    retarget_images(retargeting, input_path, output_path, str(config_path))


if __name__ == "__main__":
    tyro.cli(main)
