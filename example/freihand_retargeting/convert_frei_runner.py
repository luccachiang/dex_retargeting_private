#!/usr/bin/env python3
"""
Runner script that wraps the main function from convert_frei.py
This allows direct function calls instead of subprocess calls.
"""
from pathlib import Path
import sys
import traceback

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
)

# Import the main function from convert_frei.py
from convert_frei import main as convert_frei_main


def run_retarget_runner(
    processed_dataset_path: str,
    robot_name: RobotName,
    retargeting_type: RetargetingType = RetargetingType.vector,
    hand_type: HandType = HandType.right,
    save_images: bool = False,
    output_dir: str = "retargeted_dataset",
    max_samples: int = None
) -> bool:
    """
    Runner function that calls convert_frei.main() directly.
    
    Args:
        processed_dataset_path: Path to processed dataset pickle file
        robot_name: Robot name
        retargeting_type: Retargeting type (vector, position)
        hand_type: Hand type (right, left)
        save_images: Whether to save comparison images
        output_dir: Output directory
        max_samples: Maximum number of samples to process (None for all)
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        print(f"=== Starting retargeting for {robot_name.name} ===")
        
        # Call the main function directly
        result = convert_frei_main(
            processed_dataset_path=processed_dataset_path,
            robot_name=robot_name,
            retargeting_type=retargeting_type,
            hand_type=hand_type,
            save_images=save_images,
            output_dir=output_dir,
            max_samples=max_samples
        )
        
        if result == 0:
            print(f"✓ {robot_name.name} completed successfully")
            return True
        else:
            print(f"✗ {robot_name.name} failed with return code {result}")
            return False
            
    except Exception as e:
        print(f"✗ {robot_name.name} failed with exception: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # This allows the runner to be called directly from command line if needed
    import tyro
    
    def main(
        processed_dataset_path: str,
        robot_name: RobotName,
        retargeting_type: RetargetingType = RetargetingType.vector,
        hand_type: HandType = HandType.right,
        save_images: bool = False,
        output_dir: str = "retargeted_dataset",
        max_samples: int = None
    ):
        success = run_retarget_runner(
            processed_dataset_path=processed_dataset_path,
            robot_name=robot_name,
            retargeting_type=retargeting_type,
            hand_type=hand_type,
            save_images=save_images,
            output_dir=output_dir,
            max_samples=max_samples
        )
        
        return 0 if success else 1
    
    exit_code = tyro.cli(main)
    sys.exit(exit_code)