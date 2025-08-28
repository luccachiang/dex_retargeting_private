#!/usr/bin/env python3
"""
Load and augment FreiHAND dataset for hand pose retargeting
This script loads the raw FreiHAND data and applies augmentation techniques,
then saves the processed dataset for use with multiple robot embodiments.
"""
from pathlib import Path
import numpy as np
import cv2
import tyro
from tqdm import tqdm
import pickle
import random
from typing import List, Tuple
import copy
import sys
import time
import multiprocessing as mp

sys.path.append(str(Path(__file__).parent / "freihand"))
from freihand.utils.fh_utils import read_img, plot_hand, projectPoints, load_db_annotation
from convert_xyz3d_to_joint_pos import convert_xyz3d_to_joint_pos

def process_single_sample(args):
    """Process a single sample - designed for multiprocessing"""
    i, annotations_all, freihand_dataset_path, annotations_train_len, original_count = args
    
    try:
        if i < original_count:
            # Original sample - load from image file
            idx = f"{i:08d}"
            if i < annotations_train_len:
                image_path = Path(freihand_dataset_path) / "training" / "rgb" / f"{idx}.jpg"
                split = "training"
            else:
                # Evaluation sample
                eval_idx = i - annotations_train_len
                idx = f"{eval_idx:08d}"
                image_path = Path(freihand_dataset_path) / "evaluation" / "rgb" / f"{idx}.jpg"
                split = "evaluation"
            
            if image_path.exists():
                data = load_image_data(str(image_path), annotations_all, read_img=False)
                data['split'] = split
                data['is_augmented'] = False
            else:
                # For augmented samples or missing files, create data from annotation
                K, mano_params, xyz_3d = annotations_all[i]
                joint_pos = convert_xyz3d_to_joint_pos(np.array(xyz_3d), 'Right')
                keypoint_2d = projectPoints(np.array(xyz_3d), np.array(K))
                
                data = {
                    'image_id': i,
                    'image': None,  # No image for augmented samples
                    'joint_pos': joint_pos,
                    'keypoint_2d': keypoint_2d,
                    'K': K,
                    'mano_params': mano_params,
                    'xyz_3d': xyz_3d,
                    'split': 'augmented',
                    'is_augmented': True
                }
        else:
            # Augmented sample
            K, mano_params, xyz_3d = annotations_all[i]
            joint_pos = convert_xyz3d_to_joint_pos(np.array(xyz_3d), 'Right')
            keypoint_2d = projectPoints(np.array(xyz_3d), np.array(K))
            
            data = {
                'image_id': i,
                'image': None,  # No image for augmented samples
                'joint_pos': joint_pos,
                'keypoint_2d': keypoint_2d,
                'K': K,
                'mano_params': mano_params,
                'xyz_3d': xyz_3d,
                'split': 'augmented',
                'is_augmented': True
            }
        
        return i, data
        
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        return i, None


def process_samples_multiprocessing(
    annotations_all, 
    freihand_dataset_path, 
    annotations_train_len, 
    original_count, 
    total_samples,
    num_workers=None
):
    """Process samples using multiprocessing"""
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Limit to 8 to avoid overwhelming system
    
    print(f"Using {num_workers} workers for multiprocessing")
    
    # Prepare arguments for each sample
    args_list = [
        (i, annotations_all, freihand_dataset_path, annotations_train_len, original_count)
        for i in range(total_samples)
    ]
    
    processed_data = [None] * total_samples
    
    # Use multiprocessing with progress tracking
    with mp.Pool(num_workers) as pool:
        # Use imap for better progress tracking
        results = list(tqdm(
            pool.imap(process_single_sample, args_list),
            total=total_samples,
            desc="Processing samples (multiprocessing)"
        ))
    
    # Sort results by index and extract data
    for idx, data in results:
        if data is not None:
            processed_data[idx] = data
    
    # Filter out None values (failed processing)
    processed_data = [data for data in processed_data if data is not None]
    
    return processed_data


def process_samples_sequential(
    annotations_all, 
    freihand_dataset_path, 
    annotations_train_len, 
    original_count, 
    total_samples
):
    """Process samples sequentially (fallback)"""
    processed_data = []
    
    for i in tqdm(range(total_samples), desc="Processing samples (sequential)"):
        args = (i, annotations_all, freihand_dataset_path, annotations_train_len, original_count)
        idx, data = process_single_sample(args)
        if data is not None:
            processed_data.append(data)
    
    return processed_data

def load_image_data(image_path, annotations, read_img=False):
    """Load image and extract pose data"""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    
    anno = [annotations[i] for i in range(len(annotations)) if i == int(Path(image_path).stem)][0]
    K, mano_params, xyz_3d = anno
    K = np.array(K)
    xyz_3d = np.array(xyz_3d)
    keypoint_2d = projectPoints(xyz_3d, K)
    joint_pos = convert_xyz3d_to_joint_pos(xyz_3d, 'Right')
    
    # Read image
    image_rgb = image_path
    if read_img:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return {
        'image_id': int(Path(image_path).stem),
        'image': image_rgb, # str path or img
        'joint_pos': joint_pos,
        'keypoint_2d': keypoint_2d,
        'K': K,
        'mano_params': mano_params,
        'xyz_3d': xyz_3d
    }


def hand_annotation_augmentation(hand_annotation_list: List[Tuple], 
                                augmentation_factor: int = 5,
                                enable_interpolation: bool = True,
                                enable_noise: bool = True,
                                seed: int = None) -> List[Tuple]:
    """
    Augment hand pose data using various techniques including interpolation,
    rotation, translation, scaling, and noise addition.
    
    Args:
        hand_annotation_list: List of annotations [K, mano_params, xyz_3d]
        augmentation_factor: How many times to multiply the dataset size
        enable_interpolation: Enable temporal interpolation between poses
        enable_noise: Enable noise addition
        seed: Random seed for reproducible augmentation
    
    Returns:
        Augmented list of hand annotations
    """
    
    # Set random seed for reproducible augmentation
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"Random seed set to: {seed}")
    
    print(f"Starting augmentation with {len(hand_annotation_list)} original samples...")
    
    augmented_data = copy.deepcopy(hand_annotation_list)
    original_count = len(hand_annotation_list)
    target_count = original_count * augmentation_factor
    
    # Extract xyz_3d positions for analysis
    all_xyz_3d = np.array([anno[2] for anno in hand_annotation_list])  # Shape: (N, 21, 3)
    
    # Compute global statistics for realistic augmentation ranges
    global_center = np.mean(all_xyz_3d.reshape(-1, 3), axis=0)
    global_std = np.std(all_xyz_3d.reshape(-1, 3), axis=0)
    
    print(f"Global hand pose statistics:")
    print(f"  Center: {global_center}")
    print(f"  Std: {global_std}")
    
    # Generate augmented samples
    while len(augmented_data) < target_count:
        # Choose augmentation technique randomly
        techniques = []
        if enable_interpolation and len(hand_annotation_list) >= 2:
            techniques.append('interpolation')
        if enable_noise:
            techniques.append('noise')
        
        if not techniques:
            break
            
        technique = random.choice(techniques)
        new_sample = None
        
        if technique == 'interpolation':
            new_sample = generate_interpolated_pose(hand_annotation_list)
        elif technique == 'noise':
            new_sample = generate_noisy_pose(hand_annotation_list, global_std)
        
        if new_sample is not None and is_valid_pose(new_sample[2]):
            augmented_data.append(new_sample)
    
    print(f"Augmentation complete: {len(augmented_data)} total samples ({len(augmented_data) - original_count} augmented)")
    return augmented_data


def generate_interpolated_pose(hand_annotation_list: List[Tuple]) -> Tuple:
    """Generate single interpolated pose between two random poses"""
    if len(hand_annotation_list) < 2:
        return None
    
    # Randomly select two different poses
    idx1, idx2 = random.sample(range(len(hand_annotation_list)), 2)
    pose1 = hand_annotation_list[idx1]
    pose2 = hand_annotation_list[idx2]
    
    # Random interpolation factor
    alpha = random.uniform(0.1, 0.9)  # Avoid exact endpoints
    
    # Interpolate xyz_3d
    xyz_3d_1 = np.array(pose1[2])
    xyz_3d_2 = np.array(pose2[2])
    interpolated_xyz_3d = (1 - alpha) * xyz_3d_1 + alpha * xyz_3d_2
    
    # Use camera matrix from first pose
    K = pose1[0]
    
    # Interpolate MANO parameters if they exist
    if pose1[1] is not None and pose2[1] is not None:
        if isinstance(pose1[1], dict):
            # Handle dictionary format
            interpolated_mano = {}
            for key in pose1[1].keys():
                if isinstance(pose1[1][key], (np.ndarray, list)):
                    val1 = np.array(pose1[1][key])
                    val2 = np.array(pose2[1][key])
                    interpolated_mano[key] = ((1 - alpha) * val1 + alpha * val2).tolist()
                else:
                    interpolated_mano[key] = pose1[1][key]
        elif isinstance(pose1[1], (list, np.ndarray)):
            # Handle list/array format
            val1 = np.array(pose1[1])
            val2 = np.array(pose2[1])
            interpolated_mano = ((1 - alpha) * val1 + alpha * val2).tolist()
        else:
            interpolated_mano = pose1[1]
    else:
        interpolated_mano = pose1[1]
    
    return [K, interpolated_mano, interpolated_xyz_3d.tolist()]


def generate_noisy_pose(hand_annotation_list: List[Tuple], global_std: np.ndarray) -> Tuple:
    """Generate new pose by adding random noise"""
    base_pose = random.choice(hand_annotation_list)
    K, mano_params, xyz_3d = copy.deepcopy(base_pose)
    
    xyz_3d = np.array(xyz_3d)
    
    # Generate noise (small percentage of global variation)
    noise_std = global_std * 0.02  # 2% of global std
    noise = np.random.normal(0, noise_std, xyz_3d.shape)
    
    # Apply noise
    noisy_xyz_3d = xyz_3d + noise
    
    return [K, mano_params, noisy_xyz_3d.tolist()]


def is_valid_pose(xyz_3d: List, verbose: bool = False) -> bool:
    """Check if the generated pose is valid"""
    xyz_3d = np.array(xyz_3d)
    
    # Check for correct shape (should be 21 joints x 3 coordinates)
    if xyz_3d.shape != (21, 3):
        if verbose:
            print(f"Invalid shape: {xyz_3d.shape}, expected (21, 3)")
        return False
    
    # Check for NaN or inf values
    if not np.all(np.isfinite(xyz_3d)):
        if verbose:
            print("Contains NaN or inf values")
        return False
    
    # Check if joints are too far apart (hand integrity) - more lenient
    distances = []
    for i in range(len(xyz_3d)):
        for j in range(i+1, len(xyz_3d)):
            dist = np.linalg.norm(xyz_3d[i] - xyz_3d[j])
            distances.append(dist)
    
    max_joint_distance = 0.2  # 20cm max between any two joints (more lenient)
    if np.any(np.array(distances) > max_joint_distance):
        if verbose:
            max_dist = np.max(distances)
            print(f"Joints too far apart: max distance = {max_dist:.3f}m")
        return False
    
    # Check if hand is within reasonable bounds - more lenient
    hand_span = np.max(xyz_3d, axis=0) - np.min(xyz_3d, axis=0)
    max_hand_span = 0.4  # 40cm max hand span (more lenient)
    if np.any(hand_span > max_hand_span):
        if verbose:
            print(f"Hand span too large: {hand_span}")
        return False
    
    # Check if all joints are too close together (degenerate case)
    min_hand_span = 0.01  # 1cm minimum hand span
    if np.all(hand_span < min_hand_span):
        if verbose:
            print(f"Hand span too small: {hand_span}")
        return False
    
    return True


def main(
    freihand_dataset_path: str,
    augmentation_factor: int = 5,
    enable_interpolation: bool = True,
    enable_noise: bool = True,
    seed: int = 42,
    output_dir: str = "processed_dataset",
    num_workers: int = None,
    use_multiprocessing: bool = True
):
    """
    Load and augment FreiHAND dataset
    
    Args:
        freihand_dataset_path: FreiHAND dataset path
        augmentation_factor: How many times to multiply the dataset size
        enable_interpolation: Enable temporal interpolation between poses
        enable_noise: Enable noise addition
        seed: Random seed for reproducible augmentation
        output_dir: Output directory for processed dataset
        num_workers: Number of worker processes (None for auto)
        use_multiprocessing: Whether to use multiprocessing for data loading
    """
    
    try:
        # Set global random seed at the beginning
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            print(f"Global random seed set to: {seed}")
        
        print("Loading FreiHAND annotations...")
        # Load original annotations
        annotations_train = list(load_db_annotation(freihand_dataset_path, 'training'))
        annotations_eval = list(load_db_annotation(freihand_dataset_path, 'evaluation'))
        
        print(f"Loaded {len(annotations_train)} training and {len(annotations_eval)} evaluation samples")
        
        # Apply augmentation
        annotations_all = hand_annotation_augmentation(
            annotations_train + annotations_eval,
            augmentation_factor=augmentation_factor,
            enable_interpolation=enable_interpolation,
            enable_noise=enable_noise,
            seed=seed
        )
        
        total_samples = len(annotations_all)
        print(f"Total samples after augmentation: {total_samples}")
        
        # Process all samples using multiprocessing or sequential processing
        original_count = len(annotations_train) + len(annotations_eval)
        
        if use_multiprocessing:
            try:
                processed_data = process_samples_multiprocessing(
                    annotations_all=annotations_all,
                    freihand_dataset_path=freihand_dataset_path,
                    annotations_train_len=len(annotations_train),
                    original_count=original_count,
                    total_samples=total_samples,
                    num_workers=num_workers
                )
            except Exception as e:
                print(f"Multiprocessing failed: {e}")
                print("Falling back to sequential processing...")
                use_multiprocessing = False
        
        if not use_multiprocessing:
            processed_data = process_samples_sequential(
                annotations_all=annotations_all,
                freihand_dataset_path=freihand_dataset_path,
                annotations_train_len=len(annotations_train),
                original_count=original_count,
                total_samples=total_samples
            )
        
        # Save processed dataset with metadata
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_pkl_path = output_path / "processed_freihand_dataset.pkl"
        
        # Create dataset metadata
        original_samples = sum(1 for d in processed_data if not d['is_augmented'])
        augmented_samples = sum(1 for d in processed_data if d['is_augmented'])
        dataset_info = {
            'data': processed_data,
            'metadata': {
                'augmentation_factor': augmentation_factor,
                'enable_interpolation': enable_interpolation,
                'enable_noise': enable_noise,
                'seed': seed,
                'num_workers': num_workers if use_multiprocessing else None,
                'use_multiprocessing': use_multiprocessing,
                'total_samples': total_samples,
                'original_samples': original_samples,
                'augmented_samples': augmented_samples,
                'creation_timestamp': time.time(),
            }
        }
        
        with open(output_pkl_path, "wb") as f:
            pickle.dump(dataset_info, f)
        
        print(f"\n✓ Dataset processing completed successfully!")
        print(f"Total samples: {total_samples}")
        print(f"Processing method: {'Multiprocessing' if use_multiprocessing else 'Sequential'}")
        if use_multiprocessing:
            actual_workers = num_workers if num_workers else min(mp.cpu_count(), 8)
            print(f"Workers used: {actual_workers}")
        print(f"Processed dataset saved to: {output_pkl_path}")
        print(f"Augmentation seed: {seed}")
        
        # Print statistics
        print(f"Original samples: {original_samples}")
        print(f"Augmented samples: {augmented_samples}")
        
    except Exception as e:
        print(f"\n❌ Dataset processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Set multiprocessing start method to avoid issues on different platforms
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set
    
    tyro.cli(main)