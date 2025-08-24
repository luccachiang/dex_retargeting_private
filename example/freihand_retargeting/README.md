# FreiHAND Retargeting

This directory contains tools for retargeting FreiHAND dataset to robot hand configurations using the DexRetargeting framework.

## Overview

The FreiHAND retargeting tools enable:
- Converting FreiHAND 3D hand poses to robot joint configurations
- Rendering robot hands using Sapien
- Comparing ground truth vs detected hand poses
- Batch processing of FreiHAND dataset

## Files Description

### Core Scripts

- **`convert_frei.py`**: Main script for converting FreiHAND data to robot configurations
- **`convert_frei_all_hand.py`**: Batch processing script for all robot hand types
- **`detect_and_render_single_image.py`**: Single image detection and rendering
- **`single_hand_detector.py`**: MediaPipe-based hand detection wrapper
- **`convert_xyz3d_to_joint_pos.py`**: Coordinate transformation utilities

### Utilities

- **`sapien_utils.py`**: Sapien rendering utilities and scene management
- **`freihand/`**: FreiHAND dataset utilities and evaluation scripts

## Installation

1. Install dependencies:
```bash
pip install numpy opencv-python sapien matplotlib tqdm tyro
```

2. Ensure FreiHAND dataset is available:
```bash
# Download FreiHAND dataset to your preferred location
# Dataset structure should be:
# freihand_dataset/
# ├── training/
# │   ├── rgb/
# │   └── ...
# ├── training_K.json
# ├── training_mano.json
# └── training_xyz.json
```

## Usage

### Single Image Processing

Process a single FreiHAND image:

```bash
python convert_frei.py \
    --freihand_dataset_path /path/to/freihand \
    --robot_name allegro \
    --retargeting_type vector \
    --hand_type right \
    --save_images
```

### Batch Processing

Process all robot hand types:

```bash
python convert_frei_all_hand.py \
    --freihand_dataset_path /path/to/freihand \
    --retargeting_type dexpilot \
    --hand_type right \
    --max_samples 100
```

### Single Image Detection and Rendering

```bash
python detect_and_render_single_image.py \
    --image_path /path/to/image.jpg \
    --robot_name shadow \
    --retargeting_type vector \
    --hand_type right
```

## Supported Robots

The following robot hands are supported:
- `allegro`: Allegro Hand
- `shadow`: Shadow Hand
- `svh`: Schunk SVH Hand
- `leap`: Leap Hand
- `ability`: Ability Hand
- `inspire`: Inspire Hand
- `panda`: Panda Gripper
- `xhand`: XHand

## Retargeting Types

- **`vector`**: Vector-based retargeting for teleoperation
- **`position`**: Position-based retargeting for offline processing
- **`dexpilot`**: DexPilot retargeting with finger closing prior

## Hand Types

- **`right`**: Right hand
- **`left`**: Left hand

## Output Structure

```
output_directory/
├── results_robot_name_retargeting_type_hand_type.pkl
├── images/
│   ├── comparison_sample_000000.png
│   ├── comparison_sample_000001.png
│   └── ...
└── execution_log.txt
```

## Key Features

### Coordinate Transformation

The tools handle coordinate transformations between:
- FreiHAND 3D coordinates (xyz_3d)
- MediaPipe joint positions (joint_pos)
- Robot joint configurations (qpos)

### Rendering

- Sapien-based 3D rendering
- Headless rendering support
- Comparison visualizations
- Side-by-side GT vs detection comparisons

### Batch Processing

- Parallel processing support
- Progress tracking
- Error handling and recovery
- Comprehensive logging

## Examples

### Basic Conversion

```python
from convert_frei import process_single_image

result = process_single_image(
    image_path="sample.jpg",
    robot_name="allegro",
    retargeting_type="vector",
    hand_type="right"
)
```

### Coordinate Transformation

```python
from convert_xyz3d_to_joint_pos import convert_xyz3d_to_joint_pos

# Convert FreiHAND xyz_3d to joint_pos format
joint_pos = convert_xyz3d_to_joint_pos(xyz_3d, hand_type="Right")
```

### Sapien Rendering

```python
from sapien_utils import safe_render_robot

# Render robot hand with given qpos
rendered_image = safe_render_robot(scene, cam, robot, qpos, joint_names)
```

## Troubleshooting

### Common Issues

1. **Sapien rendering errors**: Ensure GPU drivers are up to date
2. **Memory issues**: Reduce batch size or use headless rendering
3. **Coordinate transformation errors**: Check hand type and robot configuration

### Debug Mode

Enable debug output by setting environment variable:
```bash
export DEBUG_FREIHAND_RETARGETING=1
```

## Performance

- **Single image processing**: ~1-2 seconds per image
- **Batch processing**: Varies by dataset size and hardware
- **Memory usage**: ~2-4GB for typical batch sizes

## Contributing

When adding new features:
1. Follow existing code structure
2. Add appropriate error handling
3. Update documentation
4. Test with multiple robot configurations

## License

This code is part of the dex-retargeting framework. See the main repository for license information.
