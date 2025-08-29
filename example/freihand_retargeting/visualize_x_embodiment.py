from __future__ import annotations

import os
import time
import pickle
from pathlib import Path
from typing import Literal, Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import tyro
from robot_descriptions.loaders.yourdfpy import load_robot_description
import yourdfpy

import viser
from viser.extras import ViserUrdf

# Import dex_retargeting modules for proper joint mapping
from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path, DEX_RETARGETING_PATH
from dex_retargeting.retargeting_config import RetargetingConfig


@dataclass
class HandConfig:
    """Configuration for a single hand."""
    name: str
    urdf_path: Optional[str] = None
    robot_type: Optional[str] = None
    qpose: Optional[List[float]] = None
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pkl_file: Optional[str] = None  # Path to the pkl file with dataset
    robot_name: Optional[RobotName] = None  # Robot name for retargeting config
    retargeting_type: Optional[RetargetingType] = None  # Retargeting type
    hand_type: Optional[HandType] = None  # Hand type


@dataclass
class DatasetEntry:
    """Single entry from the dataset."""
    image_id: str
    joint_pos: List[float]
    qpos: List[float] 
    keypoint_2d: List[List[float]]


class MultiRobotDatasetController:
    """Controller for multiple robot instances with dataset navigation."""
    
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.robots: Dict[str, ViserUrdf] = {}
        self.slider_handles: Dict[str, List[viser.GuiInputHandle[float]]] = {}
        self.initial_configs: Dict[str, List[float]] = {}
        self.axes_handles: Dict[str, Dict] = {}
        
        # Dataset-related attributes
        self.dataset: Dict[str, List[DatasetEntry]] = {}  # robot_id -> list of poses
        self.current_index: int = 0
        self.max_index: int = 0
        self.index_slider: Optional[viser.GuiInputHandle[int]] = None
        self.auto_play: bool = False
        self.play_speed: float = 0.1  # seconds between frames
        
    def load_dataset(self, hand_configs: List[HandConfig]):
        """Load dataset from pkl files for each hand configuration."""
        print("Loading dataset files...")
        
        min_length = float('inf')
        
        for i, config in enumerate(hand_configs):
            if config.pkl_file is None:
                print(f"No pkl file specified for {config.name}, skipping dataset loading")
                continue
                
            pkl_path = Path(config.pkl_file)
            if not pkl_path.exists():
                print(f"❌ Dataset file not found: {pkl_path}")
                continue
                
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    
                print(f"Loaded {len(data)} entries for {config.name}")
                
                # Convert to DatasetEntry objects
                dataset_entries = []
                for entry in data:
                    if isinstance(entry, dict):
                        # Convert numpy arrays to lists for compatibility
                        qpos = entry.get('qpos', [])
                        if isinstance(qpos, np.ndarray):
                            qpos = qpos.tolist()
                        
                        joint_pos = entry.get('joint_pos', [])
                        if isinstance(joint_pos, np.ndarray):
                            joint_pos = joint_pos.tolist()
                        
                        keypoint_2d = entry.get('keypoint_2d', [])
                        if isinstance(keypoint_2d, np.ndarray):
                            keypoint_2d = keypoint_2d.tolist()
                        
                        dataset_entry = DatasetEntry(
                            image_id=entry.get('image_id', ''),
                            joint_pos=joint_pos,
                            qpos=qpos,
                            keypoint_2d=keypoint_2d
                        )
                        dataset_entries.append(dataset_entry)
                    
                robot_id = f"hand_{i}"
                self.dataset[robot_id] = dataset_entries
                
                # Store retargeting joint information if available
                if not hasattr(self, 'retargeting_joint_names'):
                    self.retargeting_joint_names = {}
                if not hasattr(self, 'retargeting_configs'):
                    self.retargeting_configs = {}
                
                # Load retargeting configuration if robot info is available
                if config.robot_name is not None and config.retargeting_type is not None and config.hand_type is not None:
                    try:
                        # Set up URDF directory
                        robot_dir = Path(DEX_RETARGETING_PATH) / "assets" / "robots" / "hands"
                        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
                        
                        # Get config path
                        config_path = get_default_config_path(config.robot_name, config.retargeting_type, config.hand_type)
                        
                        if config_path and config_path.exists():
                            # Load retargeting configuration
                            retargeting_config = RetargetingConfig.load_from_file(config_path)
                            retargeting = retargeting_config.build()
                            
                            # Store retargeting joint names
                            self.retargeting_joint_names[robot_id] = retargeting.joint_names
                            self.retargeting_configs[robot_id] = retargeting
                            
                            print(f"  Loaded retargeting config for {config.name}: {len(retargeting.joint_names)} joints")
                            print(f"    Joint names: {retargeting.joint_names}")
                            
                            # Check if qpos length matches joint names length
                            if dataset_entries and len(dataset_entries[0].qpos) > 0:
                                qpos_length = len(dataset_entries[0].qpos)
                                if qpos_length != len(retargeting.joint_names):
                                    print(f"  ⚠️  WARNING: QPOS length ({qpos_length}) != joint names length ({len(retargeting.joint_names)})")
                                    print(f"  Using qpos length for mapping (DexPilot format may differ)")
                                    # Use qpos length for mapping since DexPilot format may be different
                                    self.retargeting_joint_names[robot_id] = [f"joint_{j}" for j in range(qpos_length)]
                        else:
                            print(f"  Warning: Retargeting config not found for {config.name}")
                            # Fallback to inferred joints
                            if dataset_entries and len(dataset_entries[0].qpos) > 0:
                                qpos_length = len(dataset_entries[0].qpos)
                                self.retargeting_joint_names[robot_id] = [f"joint_{j}" for j in range(qpos_length)]
                                print(f"  Inferred {qpos_length} retargeting joints for {config.name}")
                    except Exception as e:
                        print(f"  Warning: Failed to load retargeting config for {config.name}: {e}")
                        # Fallback to inferred joints
                        if dataset_entries and len(dataset_entries[0].qpos) > 0:
                            qpos_length = len(dataset_entries[0].qpos)
                            self.retargeting_joint_names[robot_id] = [f"joint_{j}" for j in range(qpos_length)]
                            print(f"  Inferred {qpos_length} retargeting joints for {config.name}")
                else:
                    # Fallback to inferred joints
                    if dataset_entries and len(dataset_entries[0].qpos) > 0:
                        qpos_length = len(dataset_entries[0].qpos)
                        self.retargeting_joint_names[robot_id] = [f"joint_{j}" for j in range(qpos_length)]
                        print(f"  Inferred {qpos_length} retargeting joints for {config.name}")
                
                min_length = min(min_length, len(dataset_entries))
                
            except Exception as e:
                print(f"❌ Error loading dataset for {config.name}: {e}")
                continue
        
        if min_length != float('inf'):
            self.max_index = min_length - 1
            print(f"Dataset loaded successfully. Max index: {self.max_index}")
        else:
            print("No datasets loaded successfully")
            self.max_index = 0
        
    def add_robot(
        self,
        robot_id: str,
        urdf: yourdfpy.URDF,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        load_meshes: bool = True,
        load_collision_meshes: bool = False,
    ) -> ViserUrdf:
        """Add a robot to the scene."""
        
        # Create a transform frame for this robot
        frame_name = f"/robot_{robot_id}_frame"
        
        if any(pos != 0.0 for pos in position) or any(rot != 0.0 for rot in rotation):
            from scipy.spatial.transform import Rotation
            # Convert rotation to quaternion (wxyz format for viser)
            r = Rotation.from_euler('xyz', rotation)
            quat_xyzw = r.as_quat()  # Returns [x, y, z, w]
            quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # Convert to [w, x, y, z]
            
            print(f"  Creating transform frame at position: {position}, rotation: {rotation}")
            
            # Create a coordinate frame to visualize the transform
            self.server.scene.add_frame(
                frame_name,
                position=np.array(position),
                wxyz=np.array(quat_wxyz),
                axes_length=0.1,
                axes_radius=0.005,
            )
        else:
            # Create frame at origin
            self.server.scene.add_frame(
                frame_name,
                axes_length=0.1,
                axes_radius=0.005
            )
        
        # Create ViserUrdf wrapper as a child of the transform frame
        viser_urdf = ViserUrdf(
            self.server,
            urdf_or_path=urdf,
            root_node_name=frame_name + f"/robot_{robot_id}",  # Make it a child of the frame
            load_meshes=load_meshes,
            load_collision_meshes=load_collision_meshes,
            collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
        )
        
        self.robots[robot_id] = viser_urdf
        self.axes_handles[robot_id] = {}
        
        # Store joint mapping information for this robot
        if not hasattr(self, 'joint_mappings'):
            self.joint_mappings = {}
        
        # Get joint names from the URDF and create mapping
        try:
            # Use the correct method for yourdfpy URDF objects
            if hasattr(urdf, 'actuated_joint_names'):
                # yourdfpy URDF object
                sapien_joint_names = urdf.actuated_joint_names
            elif hasattr(urdf, 'get_active_joints'):
                # Sapien URDF object
                sapien_joint_names = [joint.get_name() for joint in urdf.get_active_joints()]
            else:
                # Fallback: try to get joint names from the URDF
                sapien_joint_names = getattr(urdf, 'joint_names', [])
                if not sapien_joint_names and hasattr(urdf, 'robot'):
                    sapien_joint_names = [j.name for j in urdf.robot.joints if j.type not in ['fixed', 'mimic']]
            
            # Get retargeting joint names if available
            retargeting_joint_names = getattr(self, 'retargeting_joint_names', {}).get(robot_id, None)
            
            if retargeting_joint_names is not None:
                # Create proper mapping from retargeting joint names to sapien joint names
                # This matches the logic used in Sapien rendering: retargeting_to_sapien
                retargeting_to_sapien = []
                for sapien_joint_name in sapien_joint_names:
                    try:
                        # Find the index of this joint in retargeting joint names
                        retargeting_index = retargeting_joint_names.index(sapien_joint_name)
                        retargeting_to_sapien.append(retargeting_index)
                    except ValueError:
                        # Joint not found in retargeting, use -1 to indicate missing
                        retargeting_to_sapien.append(-1)
                
                self.joint_mappings[robot_id] = np.array(retargeting_to_sapien)
                print(f"  Joint mapping for {robot_id}: {len(sapien_joint_names)} joints")
                print(f"    Sapien joints: {sapien_joint_names}")
                print(f"    Retargeting joints: {retargeting_joint_names}")
                print(f"    Mapping (retargeting_idx -> sapien_idx): {retargeting_to_sapien}")
                
                # Check if mapping is valid (at least some joints mapped)
                valid_mappings = [idx for idx in retargeting_to_sapien if idx >= 0]
                if len(valid_mappings) == 0:
                    print(f"  ⚠️  WARNING: No valid joint mappings found for {robot_id}")
                    print(f"  Using sequential mapping as fallback")
                    self.joint_mappings[robot_id] = np.arange(min(len(sapien_joint_names), len(retargeting_joint_names)))
            else:
                # Fallback: use simple sequential mapping
                self.joint_mappings[robot_id] = np.arange(len(sapien_joint_names))
                print(f"  Joint mapping for {robot_id}: {len(sapien_joint_names)} joints (sequential)")
                
        except Exception as e:
            print(f"Warning: Could not create joint mapping for {robot_id}: {e}")
            self.joint_mappings[robot_id] = None
        
        return viser_urdf
    
    def create_robot_control_sliders(self, robot_id: str, hand_name: str) -> Tuple[List[viser.GuiInputHandle[float]], List[float]]:
        """Create control sliders for a specific robot."""
        viser_urdf = self.robots[robot_id]
        slider_handles: List[viser.GuiInputHandle[float]] = []
        initial_config: List[float] = []
        
        with self.server.gui.add_folder(f"{hand_name} - Joint Control"):
            for joint_name, (lower, upper) in viser_urdf.get_actuated_joint_limits().items():
                lower = lower if lower is not None else -np.pi
                upper = upper if upper is not None else np.pi
                initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
                
                slider = self.server.gui.add_slider(
                    label=f"{joint_name}",
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=initial_pos,
                )
                
                # Create closure to capture robot_id and slider_handles
                def make_update_callback(rid, handles_list):
                    def update_callback(_):
                        self.robots[rid].update_cfg(
                            np.array([s.value for s in handles_list])
                        )
                    return update_callback
                
                slider.on_update(make_update_callback(robot_id, slider_handles))
                slider_handles.append(slider)
                initial_config.append(initial_pos)
        
        self.slider_handles[robot_id] = slider_handles
        self.initial_configs[robot_id] = initial_config
        
        return slider_handles, initial_config
    
    def set_robot_pose(self, robot_id: str, joint_positions: List[float]):
        """Set a specific pose for a robot."""
        if robot_id not in self.robots:
            print(f"Robot {robot_id} not found!")
            return
        
        viser_urdf = self.robots[robot_id]
        slider_handles = self.slider_handles.get(robot_id, [])
        
        # Get joint mapping information if available
        # mapping the retargeting dataset qpos to only hand actuated joints position
        rt_data_qpos2actuated_q = getattr(self, 'joint_mappings', {}).get(robot_id, None) # xhand array([ 9, 10, 11,  0,  1,  2,  3,  4,  7,  8,  5,  6])
        
        if rt_data_qpos2actuated_q is not None:
            actuated_idxes = [] # actuated position in the retargetting saved data
            for _, actuated_idx in enumerate(rt_data_qpos2actuated_q):
                if actuated_idx >= 0 and actuated_idx < len(joint_positions):
                    actuated_idxes.append(joint_positions[actuated_idx]) # joint_positions is from dataset, which is sapien joint seq
                else:
                    # Use default value (0.0) for missing joints
                    actuated_idxes.append(0.0)
            joint_positions = actuated_idxes # sapien joint seq to urdf joint seq
        else:
            # Fallback: ensure we don't exceed the number of available joints
            num_joints = len(slider_handles)
            joint_positions = joint_positions[:num_joints]
            
            # Pad with zeros if needed
            if len(joint_positions) < num_joints:
                joint_positions = joint_positions + [0.0] * (num_joints - len(joint_positions))
        
        # Update sliders and robot configuration
        for i, pos in enumerate(joint_positions):
            if i < len(slider_handles):
                slider_handles[i].value = pos
        
        # Update the robot configuration
        viser_urdf.update_cfg(np.array(joint_positions))
    
    def update_all_poses_from_dataset(self, index: int):
        """Update all robot poses from dataset at given index."""
        if index < 0 or index > self.max_index:
            return
            
        for robot_id in self.robots.keys():
            if robot_id in self.dataset:
                dataset_entries = self.dataset[robot_id]
                if index < len(dataset_entries):
                    entry = dataset_entries[index]
                    # Use qpos from the dataset
                    qpos = entry.qpos
                    if qpos is not None and len(qpos) > 0:
                        self.set_robot_pose(robot_id, qpos)
        
        self.current_index = index
    
    def create_dataset_navigation_controls(self):
        """Create controls for navigating through the dataset."""
        if self.max_index <= 0:
            return
            
        with self.server.gui.add_folder("📊 Dataset Navigation"):
            # Index slider
            self.index_slider = self.server.gui.add_slider(
                label="Pose Index",
                min=0,
                max=self.max_index,
                step=1,
                initial_value=0,
            )
            
            @self.index_slider.on_update
            def _(_):
                self.update_all_poses_from_dataset(self.index_slider.value)
            
            # Index input field
            index_input = self.server.gui.add_number(
                label="Go to Index",
                min=0,
                max=self.max_index,
                step=1,
                initial_value=0,
            )
            
            @index_input.on_update
            def _(event):
                new_index = int(event.target.value)
                if 0 <= new_index <= self.max_index:
                    self.index_slider.value = new_index
                    self.update_all_poses_from_dataset(new_index)
            
            # Navigation buttons
            prev_button = self.server.gui.add_button("⬅️ Previous")
            next_button = self.server.gui.add_button("➡️ Next")
            
            @prev_button.on_click
            def _(_):
                if self.current_index > 0:
                    new_index = self.current_index - 1
                    self.index_slider.value = new_index
                    self.update_all_poses_from_dataset(new_index)
            
            @next_button.on_click
            def _(_):
                if self.current_index < self.max_index:
                    new_index = self.current_index + 1
                    self.index_slider.value = new_index
                    self.update_all_poses_from_dataset(new_index)
            
            # Jump buttons
            first_button = self.server.gui.add_button("⏮️ First")
            last_button = self.server.gui.add_button("⏭️ Last")
            
            @first_button.on_click
            def _(_):
                self.index_slider.value = 0
                self.update_all_poses_from_dataset(0)
            
            @last_button.on_click
            def _(_):
                self.index_slider.value = self.max_index
                self.update_all_poses_from_dataset(self.max_index)
            
            # Auto-play controls
            play_button = self.server.gui.add_button("▶️ Play")
            pause_button = self.server.gui.add_button("⏸️ Pause")
            
            speed_slider = self.server.gui.add_slider(
                label="Play Speed (fps)",
                min=0.1,
                max=30.0,
                step=0.1,
                initial_value=10.0,
            )
            
            @speed_slider.on_update
            def _(_):
                self.play_speed = 1.0 / speed_slider.value  # Convert fps to seconds per frame
            
            @play_button.on_click
            def _(_):
                self.auto_play = True
                
            @pause_button.on_click
            def _(_):
                self.auto_play = False
            
            # Information display
            self.server.gui.add_text("Dataset Size", f"{self.max_index + 1} poses")
            current_index_display = self.server.gui.add_text("Current Index", "0")
            
            # Update current index display
            def update_index_display():
                current_index_display.value = str(self.current_index)
            
            # Override update method to also update display
            original_update = self.update_all_poses_from_dataset
            def wrapped_update(index):
                original_update(index)
                update_index_display()
            self.update_all_poses_from_dataset = wrapped_update
    
    def auto_play_loop(self):
        """Auto-play loop for dataset animation."""
        while True:
            if self.auto_play and self.max_index > 0:
                next_index = (self.current_index + 1) % (self.max_index + 1)
                if self.index_slider is not None:
                    self.index_slider.value = next_index
                self.update_all_poses_from_dataset(next_index)
                time.sleep(self.play_speed)
            else:
                time.sleep(0.1)  # Sleep briefly when not playing
    
    def create_preset_buttons(self):
        """Create buttons for preset hand poses."""
        with self.server.gui.add_folder("🎯 Preset Poses"):
            # Define some common hand poses
            poses = {
                "Open Hand": [0.0] * 20,  # Fully open
                "Fist": [1.2] * 20,  # Closed fingers
                "Pointing": [0.0, 0.0, 0.0, 0.0] + [1.2] * 16,  # Index finger extended
                "Peace Sign": [0.0] * 8 + [1.2] * 12,  # Index and middle extended
                "Pinch": [0.5, 1.0, 0.8, 0.8] * 5,  # Pinching pose
            }
            
            for pose_name, joint_values in poses.items():
                button = self.server.gui.add_button(f"Set All to {pose_name}")
                
                def make_pose_callback(pose_vals):
                    def callback(_):
                        for robot_id in self.robots.keys():
                            # Adjust pose length to match robot's joint count
                            robot_joint_count = len(self.slider_handles.get(robot_id, []))
                            if robot_joint_count > 0:
                                adjusted_pose = pose_vals[:robot_joint_count] + [0.0] * max(0, robot_joint_count - len(pose_vals))
                                self.set_robot_pose(robot_id, adjusted_pose)
                    return callback
                
                button.on_click(make_pose_callback(joint_values))
    
    def create_reset_button(self):
        """Create reset button for all robots."""
        reset_button = self.server.gui.add_button("🔄 Reset All Joints")
        
        @reset_button.on_click
        def _(_):
            for robot_id in self.robots.keys():
                slider_handles = self.slider_handles.get(robot_id, [])
                initial_config = self.initial_configs.get(robot_id, [])
                for slider, init_q in zip(slider_handles, initial_config):
                    slider.value = init_q


def load_robot_urdf(
    urdf_path: Optional[str] = None,
    robot_type: Optional[str] = None,
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
) -> yourdfpy.URDF:
    """Load a URDF from file or robot descriptions."""
    
    if urdf_path is not None:
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        print(f"Loading local URDF: {urdf_path}")
        urdf = yourdfpy.URDF.load(
            str(urdf_path),
            load_meshes=load_meshes,
            load_collision_meshes=load_collision_meshes,
            build_scene_graph=load_meshes,
            build_collision_scene_graph=load_collision_meshes,
        )
    elif robot_type is not None:
        print(f"Loading robot description: {robot_type}")
        urdf = load_robot_description(
            robot_type + "_description",
            load_meshes=load_meshes,
            build_scene_graph=load_meshes,
            load_collision_meshes=load_collision_meshes,
            build_collision_scene_graph=load_collision_meshes,
        )
    else:
        raise ValueError("Either urdf_path or robot_type must be provided")
    
    return urdf


def create_hand_configs_from_dataset(dataset_path: str, selected_hands: Optional[List[str]] = None) -> List[HandConfig]:
    """Create hand configurations from dataset directory structure."""
    dataset_path = Path(dataset_path)
    
    # Mapping of directory names to robot configurations
    robot_mappings = {
        'ability': {
            'name': 'Ability Hand',
            'urdf_path': os.path.join(DEX_RETARGETING_PATH, 'assets/robots/hands', 'ability_hand', "ability_hand_right_glb.urdf"),
            'position': (0.3, 0.6, 0.0),
            'robot_name': RobotName.ability,
            'retargeting_type': RetargetingType.dexpilot, # vector and dexpilot are same, position is different
            'hand_type': HandType.right
        },
        'allegro': {
            'name': 'Allegro Hand',
            'urdf_path': os.path.join(DEX_RETARGETING_PATH, 'assets/robots/hands', 'allegro_hand', "allegro_hand_right_glb.urdf"),
            'position': (-0.3, 0.0, 0.0),
            'robot_name': RobotName.allegro,
            'retargeting_type': RetargetingType.dexpilot,
            'hand_type': HandType.right
        },
        'leap': {
            'name': 'Leap Hand',
            'urdf_path': os.path.join(DEX_RETARGETING_PATH, 'assets/robots/hands', 'leap_hand', "leap_hand_right_glb.urdf"),
            'position': (0.0, 0.0, 0.0),
            'robot_name': RobotName.leap,
            'retargeting_type': RetargetingType.dexpilot,
            'hand_type': HandType.right
        },
        'panda': {
            'name': 'Panda Gripper',
            'urdf_path': os.path.join(DEX_RETARGETING_PATH, 'assets/robots/hands', 'panda_gripper', "panda_gripper_glb.urdf"),
            'position': (0.3, 0.0, 0.0),
            'robot_name': RobotName.panda,
            'retargeting_type': RetargetingType.dexpilot,
            'hand_type': HandType.right
        },
        'shadow': {
            'name': 'Shadow Hand',
            'urdf_path': os.path.join(DEX_RETARGETING_PATH, 'assets/robots/hands', 'shadow_hand', "shadow_hand_right_glb.urdf"),
            'position': (0.6, 0.0, 0.0),
            'robot_name': RobotName.shadow,
            'retargeting_type': RetargetingType.dexpilot,
            'hand_type': HandType.right
        },
        'svh': {
            'name': 'SVH Hand',
            'urdf_path': os.path.join(DEX_RETARGETING_PATH, 'assets/robots/hands', 'schunk_hand', "schunk_svh_hand_right_glb.urdf"),
            'position': (-0.3, 0.3, 0.0),
            'robot_name': RobotName.svh,
            'retargeting_type': RetargetingType.dexpilot,
            'hand_type': HandType.right
        },
        'xhand': {
            'name': 'XHand Hand',
            'urdf_path': os.path.join(DEX_RETARGETING_PATH, 'assets/robots/hands', 'xhand', "xhand_right_glb.urdf"),
            'position': (0.3, 0.3, 0.0),
            'robot_name': RobotName.xhand,
            'retargeting_type': RetargetingType.dexpilot,
            'hand_type': HandType.right
        },
        'inspire': {
            'name': 'Inspire Hand',
            'urdf_path': os.path.join(DEX_RETARGETING_PATH, 'assets/robots/hands', 'inspire_hand', "inspire_hand_right_glb.urdf"),
            'position': (0.3, 0.0, 0.0),
            'robot_name': RobotName.inspire,
            'retargeting_type': RetargetingType.dexpilot,
            'hand_type': HandType.right
        },
    }
    
    hand_configs = []
    
    # Iterate through dataset directories
    for dir_path in dataset_path.iterdir():
        if dir_path.is_dir() and dir_path.name in robot_mappings:
            # Skip if selected_hands is provided and this hand is not in the list
            if selected_hands is not None and dir_path.name not in selected_hands:
                continue
                
            # Find the pkl file in this directory
            pkl_files = list(dir_path.glob('*.pkl'))
            if pkl_files:
                pkl_file = pkl_files[0]  # Take the first pkl file found
                mapping = robot_mappings[dir_path.name]
                
                config = HandConfig(
                    name=mapping['name'],
                    robot_type=mapping.get('robot_type'),
                    urdf_path=mapping.get('urdf_path'),
                    position=mapping['position'],
                    rotation=(0.0, 0.0, 0.0),
                    pkl_file=str(pkl_file),
                    robot_name=mapping.get('robot_name'),
                    retargeting_type=mapping.get('retargeting_type'),
                    hand_type=mapping.get('hand_type')
                )
                hand_configs.append(config)
                print(f"Found dataset for {mapping['name']}: {pkl_file}")
    
    return hand_configs


def main(
    dataset_path: str,
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
    auto_start_play: bool = False,
    hands: Optional[List[str]] = None,
) -> None:
    """
    Visualize multiple hands from dataset with navigation controls.
    
    Args:
        dataset_path: Path to dataset directory containing subdirectories with pkl files.
        load_meshes: Whether to load visual meshes.
        load_collision_meshes: Whether to load collision meshes.
        auto_start_play: Whether to start auto-playing immediately.
        hands: List of hand names to load (ability, allegro, leap, panda, shadow, svh, xhand). If None, loads all available hands.
    """
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # Create hand configurations from dataset
    hand_configs = create_hand_configs_from_dataset(dataset_path, hands)
    
    if not hand_configs:
        print("No valid hand configurations found in dataset!")
        return
    
    print(f"Found {len(hand_configs)} hand configurations")
    
    # Start viser server
    server = viser.ViserServer()
    print(f"Viser server started at: {server.get_host()}:{server.get_port()}")
    
    # Create multi-robot controller
    controller = MultiRobotDatasetController(server)
    
    # Load datasets first
    controller.load_dataset(hand_configs)
    
    # Load and setup each hand
    successful_hands = []
    for i, hand_config in enumerate(hand_configs):
        print(f"\nLoading hand {i+1}/{len(hand_configs)}: {hand_config.name}")
        print(f"  Position: {hand_config.position}")
        print(f"  Dataset: {hand_config.pkl_file}")
        
        try:
            # Load URDF
            robot_urdf = load_robot_urdf(
                urdf_path=hand_config.urdf_path,
                robot_type=hand_config.robot_type,
                load_meshes=load_meshes,
                load_collision_meshes=load_collision_meshes
            )
            
            # Check if URDF loaded successfully
            if robot_urdf is None:
                print(f"❌ Failed to load URDF for {hand_config.name}")
                continue
                
            # Generate robot ID
            robot_id = f"hand_{i}"
            
            # Add robot to controller
            viser_urdf = controller.add_robot(
                robot_id,
                robot_urdf,
                position=hand_config.position,
                rotation=hand_config.rotation,
                load_meshes=load_meshes,
                load_collision_meshes=load_collision_meshes,
            )
            
            # Check joint limits
            joint_limits = viser_urdf.get_actuated_joint_limits()
            print(f"  Found {len(joint_limits)} actuated joints")
            
            # Create control sliders
            controller.create_robot_control_sliders(robot_id, hand_config.name)
            
            successful_hands.append(hand_config.name)
            print(f"✅ Successfully loaded {hand_config.name}")
            
        except Exception as e:
            print(f"❌ Error loading {hand_config.name}: {e}")
            import traceback
            print(f"   Full error: {traceback.format_exc()}")
            continue
    
    if not controller.robots:
        print("No robots loaded successfully!")
        return
    
    # Add visibility controls
    with server.gui.add_folder("👁️ Global Visibility"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", load_meshes)
        show_collision_meshes_cb = server.gui.add_checkbox("Show collision meshes", load_collision_meshes)
        
        @show_meshes_cb.on_update
        def _(_):
            for viser_urdf in controller.robots.values():
                viser_urdf.show_visual = show_meshes_cb.value
        
        @show_collision_meshes_cb.on_update
        def _(_):
            for viser_urdf in controller.robots.values():
                viser_urdf.show_collision = show_collision_meshes_cb.value
        
        # Hide checkboxes if meshes are not loaded
        show_meshes_cb.visible = load_meshes
        show_collision_meshes_cb.visible = load_collision_meshes
    
    # Create dataset navigation controls
    controller.create_dataset_navigation_controls()
    
    # Create preset pose buttons
    controller.create_preset_buttons()
    
    # Create reset button
    controller.create_reset_button()
    
    # Load initial pose from dataset
    if controller.max_index > 0:
        controller.update_all_poses_from_dataset(0)
    
    # Auto-start play if requested
    if auto_start_play:
        controller.auto_play = True
    
    # Create grid
    if controller.robots:
        first_robot = next(iter(controller.robots.values()))
        trimesh_scene = first_robot._urdf.scene or first_robot._urdf.collision_scene
        
        # Calculate bounds for all robots to center the grid
        all_bounds = []
        for robot in controller.robots.values():
            scene = robot._urdf.scene or robot._urdf.collision_scene
            if scene is not None:
                all_bounds.append(scene.bounds)
        
        if all_bounds:
            # Combine all bounds
            combined_bounds = np.array(all_bounds)
            min_bounds = np.min(combined_bounds[:, 0, :], axis=0)
            max_bounds = np.max(combined_bounds[:, 1, :], axis=0)
            
            # Calculate grid size and position
            size_x = max_bounds[0] - min_bounds[0] + 1.0
            size_y = max_bounds[1] - min_bounds[1] + 1.0
            center_x = (max_bounds[0] + min_bounds[0]) / 2
            center_y = (max_bounds[1] + min_bounds[1]) / 2
            
            server.scene.add_grid(
                "/grid",
                width=max(2.0, size_x),
                height=max(2.0, size_y),
                position=(center_x, center_y, min_bounds[2]),
            )
        else:
            # Fallback grid
            server.scene.add_grid(
                "/grid",
                width=3,
                height=2,
                position=(0.0, 0.0, 0.0),
            )
    
    # Add information display
    with server.gui.add_folder("ℹ️ Scene Information"):
        server.gui.add_text("Total Hands", str(len(controller.robots)))
        server.gui.add_text("Loaded Hands", ", ".join(successful_hands))
        if controller.max_index > 0:
            server.gui.add_text("Dataset Size", f"{controller.max_index + 1} poses")
        
        for i, config in enumerate(hand_configs):
            if f"hand_{i}" in controller.robots:
                joint_count = len(controller.slider_handles.get(f"hand_{i}", []))
                server.gui.add_text(f"{config.name}", f"{joint_count} joints")
    
    print(f"\n✅ Successfully loaded {len(controller.robots)} hands with dataset!")
    print(f"📊 Dataset contains {controller.max_index + 1} pose sequences")
    print("🎮 Use the navigation controls to browse through different poses.")
    print("▶️ Use the play button to animate through all poses automatically.")
    print("🎯 Use preset buttons to override with manual poses.")
    
    # Start auto-play loop in background
    import threading
    play_thread = threading.Thread(target=controller.auto_play_loop, daemon=True)
    play_thread.start()
    
    # Sleep forever
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)