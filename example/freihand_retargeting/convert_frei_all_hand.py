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

from dex_retargeting.constants import RobotName


def run_convert_frei_for_robot(
    freihand_dataset_path: str,
    robot_name: RobotName,
    output_dir: str = "data/freihand_to_dexhand",
    save_images: bool = False
):
    """
    Run convert_frei.py for a single robot.
    
    Args:
        freihand_dataset_path: Path to FreiHAND dataset
        robot_name: Name of the robot
        output_dir: Output directory
        save_images: Whether to save images
    """
    try:
        # Build command
        cmd = [
            sys.executable, "convert_frei.py",
            "--freihand_dataset_path", freihand_dataset_path,
            "--robot_name", robot_name,
            "--retargeting_type", "dexpilot",
            "--hand_type", "right",
            "--output_dir", f"{output_dir}",
        ]
        
        if save_images:
            cmd.append("--save_images")
        
        print(f"Start processing {robot_name}...")
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
        print(f"\n=== {robot_name} finished ===")
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("Stdout:")
            print(result.stdout)
        
        if result.stderr:
            print("Stderr:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✓ {robot_name} succeeded")
            return True
        else:
            print(f"✗ {robot_name} failed")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {robot_name} timed out")
        return False
    except Exception as e:
        print(f"✗ {robot_name} exception: {e}")
        return False


def run_parallel_convert_frei(
    freihand_dataset_path: str,
    robot_names: List[RobotName] = None,
    max_workers: int = None,
    output_dir: str = "data/freihand_to_dexhand",
    save_images: bool = False
):
    """
    Run convert_frei.py in parallel.
    
    Args:
        freihand_dataset_path: Path to FreiHAND dataset
        robot_names: List of robots to process, if None process all robots
        max_workers: Max number of parallel processes, if None use CPU count
        output_dir: Output directory
        save_images: Whether to save images
    """
    
    # If no robots specified, use all available robots
    if robot_names is None:
        robot_names = [
            "allegro",
            "shadow",
            "svh",
            "leap",
            "ability",
            "panda",
            "xhand",
        ]
    
    # If no max_workers specified, use CPU count
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(robot_names))
    
    print(f"=== Running convert_frei.py in parallel ===")
    print(f"FreiHAND dataset path: {freihand_dataset_path}")
    print(f"Robot list: {[r for r in robot_names]}")
    print(f"Max parallel processes: {max_workers}")
    print(f"Output directory: {output_dir}")
    print(f"Save images: {save_images}")
    print(f"Retargeting type: dexpilot")
    print(f"Hand type: right")
    print("-" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare task arguments
    tasks = []
    for robot_name in robot_names:
        task_args = (
            freihand_dataset_path,
            robot_name,
            f"{output_dir}/{robot_name}",
            save_images
        )
        tasks.append(task_args)
    
    # Run in parallel
    start_time = time.time()
    
    if max_workers == 1:
        # Run serially
        print("Using serial execution mode...")
        results = []
        for task_args in tasks:
            result = run_convert_frei_for_robot(*task_args)
            results.append(result)
    else:
        # Run in parallel
        print(f"Using parallel execution mode (max {max_workers} processes)...")
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.starmap(run_convert_frei_for_robot, tasks)
    
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
        print(f"  {i+1:2d}. {robot_name}: {status}")
    
    # Save execution log
    log_file = Path(output_dir) / "execution_log.txt"
    with open(log_file, "w") as f:
        f.write(f"Execution time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"FreiHAND dataset path: {freihand_dataset_path}\n")
        f.write(f"Robot list: {[r for r in robot_names]}\n")
        f.write(f"Max parallel processes: {max_workers}\n")
        f.write(f"Total elapsed time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Success: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success rate: {successful/len(results)*100:.1f}%\n\n")
        
        f.write("Detailed results:\n")
        for i, (robot_name, success) in enumerate(zip(robot_names, results)):
            status = "Success" if success else "Failed"
            f.write(f"  {i+1:2d}. {robot_name}: {status}\n")
    
    print(f"\nExecution log saved to: {log_file}")
    
    return successful, failed


def main(
    freihand_dataset_path: str,
    robot_names: List[str] = None,
    max_workers: int = None,
    output_dir: str = "data/freihand_to_dexhand",
    save_images: bool = False
):
    """
    Run convert_frei.py in parallel.
    
    Args:
        freihand_dataset_path: Path to FreiHAND dataset
        robot_names: List of robots to process, if None process all robots
        max_workers: Max number of parallel processes, if None use CPU count
        output_dir: Output directory
        save_images: Whether to save images
    """
    
    # Convert robot name strings to enum
    if robot_names is not None:
        try:
            robot_name_enums = [RobotName(robot_name) for robot_name in robot_names]
        except ValueError as e:
            print(f"Error: Invalid robot name - {e}")
            print(f"Available robot names: {[r.value for r in RobotName]}")
            return 1
    else:
        robot_name_enums = None
    
    try:
        successful, failed = run_parallel_convert_frei(
            freihand_dataset_path=freihand_dataset_path,
            robot_names=robot_name_enums,
            max_workers=max_workers,
            output_dir=output_dir,
            save_images=save_images
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
    except Exception as e:
        print(f"\nException occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    tyro.cli(main)
