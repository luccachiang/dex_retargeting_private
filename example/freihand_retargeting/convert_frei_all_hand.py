#!/usr/bin/env python3
"""
Run convert_frei.py in parallel for all robot_name.
"""

import subprocess
import multiprocessing
import time
from pathlib import Path
import tyro
from typing import List
import os
import signal
import sys

from dex_retargeting.constants import RobotName, RetargetingType, HandType


def run_retarget_for_robot(
    processed_dataset_path: str,
    robot_name: RobotName,
    retargeting_type: RetargetingType = RetargetingType.vector,
    hand_type: HandType = HandType.right,
    save_images: bool = False,
    output_dir: str = "retargeted_dataset",
    max_samples: int = None
):
    """
    Run convert_frei.py for a single robot.
    
    Args:
        processed_dataset_path: Path to processed dataset pickle file
        robot_name: Name of the robot
        retargeting_type: Retargeting type (vector, position)
        hand_type: Hand type (right, left)
        save_images: Whether to save images
        output_dir: Output directory
        max_samples: Maximum number of samples to process
    """
    try:
        # Build command
        cmd = [
            sys.executable, "convert_frei.py",
            "--processed_dataset_path", processed_dataset_path,
            "--robot_name", robot_name.name,
            "--retargeting_type", retargeting_type.name,
            "--hand_type", hand_type.name,
            "--output_dir", f"{output_dir}/{robot_name.name}",
        ]
        
        if save_images:
            cmd.append("--save_images")
        
        if max_samples is not None:
            cmd.extend(["--max_samples", str(max_samples)])
        
        print(f"Start processing {robot_name.name}...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run command
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=36000  # 1 hour timeout
        )
        end_time = time.time()
        
        # Output result
        print(f"\n=== {robot_name.value} finished ===")
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("Stdout:")
            print(result.stdout)
        
        if result.stderr:
            print("Stderr:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✓ {robot_name.value} succeeded")
            return True
        else:
            print(f"✗ {robot_name.value} failed")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {robot_name.value} timed out")
        return False


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
    Run convert_frei.py in parallel.
    
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
    
    print(f"=== Running convert_frei.py in parallel ===")
    print(f"Processed dataset path: {processed_dataset_path}")
    print(f"Robot list: {[r.value for r in robot_names]}")
    print(f"Retargeting type: {retargeting_type.value}")
    print(f"Hand type: {hand_type.value}")
    print(f"Max parallel processes: {max_workers}")
    print(f"Output directory: {output_dir}")
    print(f"Save images: {save_images}")
    if max_samples is not None:
        print(f"Max samples per robot: {max_samples}")
    print("-" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
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
    start_time = time.time()
    
    if max_workers == 1:
        # Run serially
        print("Using serial execution mode...")
        results = []
        for task_args in tasks:
            result = run_retarget_for_robot(*task_args)
            results.append(result)
    else:
        # Run in parallel
        print(f"Using parallel execution mode (max {max_workers} processes)...")
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.starmap(run_retarget_for_robot, tasks)
    
    end_time = time.time()
    
    # Statistics
    successful = sum(results)
    failed = len(results) - successful
    
    print(f"\n=== All tasks finished ===")
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
    print(f"Success: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    # Detailed results
    print(f"\nDetailed results:")
    for i, (robot_name, success) in enumerate(zip(robot_names, results)):
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {i+1:2d}. {robot_name.value}: {status}")
    
    # Save execution log
    log_file = Path(output_dir) / "execution_log.txt"
    with open(log_file, "w") as f:
        f.write(f"Execution time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processed dataset path: {processed_dataset_path}\n")
        f.write(f"Robot list: {[r.value for r in robot_names]}\n")
        f.write(f"Retargeting type: {retargeting_type.value}\n")
        f.write(f"Hand type: {hand_type.value}\n")
        f.write(f"Max parallel processes: {max_workers}\n")
        if max_samples is not None:
            f.write(f"Max samples per robot: {max_samples}\n")
        f.write(f"Total elapsed time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Success: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success rate: {successful/len(results)*100:.1f}%\n\n")
        
        f.write("Detailed results:\n")
        for i, (robot_name, success) in enumerate(zip(robot_names, results)):
            status = "Success" if success else "Failed"
            f.write(f"  {i+1:2d}. {robot_name.value}: {status}\n")
    
    print(f"\nExecution log saved to: {log_file}")
    
    return successful, failed


def main(
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
    Run convert_frei.py in parallel.
    
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
            print("\n✓ All tasks succeeded!")
            return 0
        else:
            print(f"\n⚠ {failed} tasks failed")
            return 1
            
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        return 1


if __name__ == "__main__":
    tyro.cli(main)