#!/usr/bin/env python3
"""
Run convert_frei.py in parallel for all robot_name using direct imports.
This version provides real-time terminal output.
"""

import multiprocessing
import time
from pathlib import Path
import tyro
from typing import List
import sys

from dex_retargeting.constants import RobotName, RetargetingType, HandType

# Import the runner function
from convert_frei_runner import run_retarget_runner


def run_retarget_for_robot(
    processed_dataset_path: str,
    robot_name: RobotName,
    retargeting_type: RetargetingType = RetargetingType.vector,
    hand_type: HandType = HandType.right,
    save_images: bool = False,
    output_dir: str = "retargeted_dataset",
    max_samples: int = None
) -> bool:
    """
    Run convert_frei.py for a single robot using direct import.
    
    Args:
        processed_dataset_path: Path to processed dataset pickle file
        robot_name: Name of the robot
        retargeting_type: Retargeting type (vector, position)
        hand_type: Hand type (right, left)
        save_images: Whether to save images
        output_dir: Output directory
        max_samples: Maximum number of samples to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    print(f"\n{'='*60}")
    print(f"Starting processing for {robot_name.name}")
    print(f"Output directory: {output_dir}/{robot_name.name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Use the runner function directly
        success = run_retarget_runner(
            processed_dataset_path=processed_dataset_path,
            robot_name=robot_name,
            retargeting_type=retargeting_type,
            hand_type=hand_type,
            save_images=save_images,
            output_dir=f"{output_dir}/{robot_name.name}",
            max_samples=max_samples
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n{'='*60}")
        if success:
            print(f"✓ {robot_name.name} COMPLETED SUCCESSFULLY")
        else:
            print(f"✗ {robot_name.name} FAILED")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"{'='*60}\n")
        
        return success
        
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"✗ {robot_name.name} FAILED WITH EXCEPTION")
        print(f"Error: {e}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"{'='*60}\n")
        
        import traceback
        traceback.print_exc()
        
        return False


def run_serial_retarget(
    processed_dataset_path: str,
    robot_names: List[RobotName],
    retargeting_type: RetargetingType,
    hand_type: HandType,
    save_images: bool,
    output_dir: str,
    max_samples: int
) -> List[bool]:
    """Run retargeting serially (one robot at a time)."""
    
    print(f"\n{'='*80}")
    print(f"RUNNING IN SERIAL MODE")
    print(f"{'='*80}")
    
    results = []
    for i, robot_name in enumerate(robot_names, 1):
        print(f"\n[{i}/{len(robot_names)}] Processing {robot_name.name}...")
        
        success = run_retarget_for_robot(
            processed_dataset_path=processed_dataset_path,
            robot_name=robot_name,
            retargeting_type=retargeting_type,
            hand_type=hand_type,
            save_images=save_images,
            output_dir=output_dir,
            max_samples=max_samples
        )
        
        results.append(success)
    
    return results


def run_parallel_retarget_worker(args):
    """Worker function for multiprocessing.Pool"""
    return run_retarget_for_robot(*args)


def run_parallel_retarget(
    processed_dataset_path: str,
    robot_names: List[RobotName] = None,
    retargeting_type: RetargetingType = RetargetingType.vector,
    hand_type: HandType = HandType.right,
    max_workers: int = None,
    save_images: bool = False,
    output_dir: str = "retargeted_dataset",
    max_samples: int = None
):
    """
    Run convert_frei.py in parallel using direct imports.
    
    Args:
        processed_dataset_path: Path to processed dataset pickle file
        robot_names: List of robots to process, if None process all robots
        retargeting_type: Retargeting type (vector, position)
        hand_type: Hand type (right, left)
        max_workers: Max number of parallel processes, if None use CPU count
        save_images: Whether to save images
        output_dir: Output directory
        max_samples: Maximum number of samples to process
    """
    
    # If no robots specified, use all available robots
    if robot_names is None:
        robot_names = [
            RobotName.allegro,
            RobotName.shadow,
            RobotName.dclaw,
            RobotName.ability,
            RobotName.bhand,
            RobotName.leap,
            RobotName.svh,
            RobotName.inspire,
            RobotName.xhand,
        ]
    
    # If no max_workers specified, use CPU count
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(robot_names))
    
    print(f"{'='*80}")
    print(f"PARALLEL RETARGETING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Processed dataset path: {processed_dataset_path}")
    print(f"Robot list: {[r.name for r in robot_names]}")
    print(f"Retargeting type: {retargeting_type.name}")
    print(f"Hand type: {hand_type.name}")
    print(f"Max parallel processes: {max_workers}")
    print(f"Output directory: {output_dir}")
    print(f"Save images: {save_images}")
    if max_samples is not None:
        print(f"Max samples per robot: {max_samples}")
    print(f"{'='*80}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    if max_workers == 1:
        # Run serially for real-time output
        results = run_serial_retarget(
            processed_dataset_path=processed_dataset_path,
            robot_names=robot_names,
            retargeting_type=retargeting_type,
            hand_type=hand_type,
            save_images=save_images,
            output_dir=output_dir,
            max_samples=max_samples
        )
    else:
        # Run in parallel
        print(f"\n{'='*80}")
        print(f"RUNNING IN PARALLEL MODE ({max_workers} processes)")
        print(f"Note: Real-time output may be interleaved between processes")
        print(f"{'='*80}")
        
        # Prepare task arguments
        tasks = []
        for robot_name in robot_names:
            task_args = (
                processed_dataset_path,
                robot_name,
                retargeting_type,
                hand_type,
                save_images,
                output_dir,
                max_samples
            )
            tasks.append(task_args)
        
        # Run in parallel
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.starmap(run_retarget_for_robot, tasks)
    
    end_time = time.time()
    
    # Calculate statistics
    successful = sum(results)
    failed = len(results) - successful
    total_time = end_time - start_time
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total elapsed time: {total_time:.2f} seconds")
    print(f"Success: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    print(f"\nDetailed results:")
    for i, (robot_name, success) in enumerate(zip(robot_names, results), 1):
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {i:2d}. {robot_name.name}: {status}")
    
    # Save execution log
    log_file = Path(output_dir) / "execution_log.txt"
    with open(log_file, "w") as f:
        f.write(f"Execution time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processed dataset path: {processed_dataset_path}\n")
        f.write(f"Robot list: {[r.name for r in robot_names]}\n")
        f.write(f"Retargeting type: {retargeting_type.name}\n")
        f.write(f"Hand type: {hand_type.name}\n")
        f.write(f"Max parallel processes: {max_workers}\n")
        if max_samples is not None:
            f.write(f"Max samples per robot: {max_samples}\n")
        f.write(f"Total elapsed time: {total_time:.2f} seconds\n")
        f.write(f"Success: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success rate: {successful/len(results)*100:.1f}%\n\n")
        
        f.write("Detailed results:\n")
        for i, (robot_name, success) in enumerate(zip(robot_names, results), 1):
            status = "Success" if success else "Failed"
            f.write(f"  {i:2d}. {robot_name.name}: {status}\n")
    
    print(f"\nExecution log saved to: {log_file}")
    print(f"{'='*80}\n")
    
    return successful, failed


def main(
    processed_dataset_path: str,
    robot_names: List[RobotName] = None,
    retargeting_type: RetargetingType = RetargetingType.vector,
    hand_type: HandType = HandType.right,
    max_workers: int = 1,  # Default to 1 for better real-time output
    save_images: bool = False,
    output_dir: str = "retargeted_dataset",
    max_samples: int = None
):
    """
    Run convert_frei.py in parallel using direct imports.
    
    Args:
        processed_dataset_path: Path to processed dataset pickle file
        robot_names: List of robots to process, if None process all robots
        retargeting_type: Retargeting type (vector, position)
        hand_type: Hand type (right, left)
        max_workers: Max number of parallel processes (default: 1 for real-time output)
        save_images: Whether to save images
        output_dir: Output directory
        max_samples: Maximum number of samples to process
    """
    
    try:
        successful, failed = run_parallel_retarget(
            processed_dataset_path=processed_dataset_path,
            robot_names=robot_names,
            retargeting_type=retargeting_type,
            hand_type=hand_type,
            max_workers=max_workers,
            save_images=save_images,
            output_dir=output_dir,
            max_samples=max_samples
        )
        
        if failed == 0:
            print("🎉 ALL TASKS SUCCEEDED! 🎉")
            return 0
        else:
            print(f"⚠️  {failed} TASK(S) FAILED")
            return 1
            
    except KeyboardInterrupt:
        print("\n❌ Execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = tyro.cli(main)
    sys.exit(exit_code)