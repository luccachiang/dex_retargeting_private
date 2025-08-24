import gc
import sapien
import numpy as np
from typing import Optional, Tuple


def reset_sapien_scene(scene: Optional[sapien.Scene] = None) -> None:
    """
    Completely reset the sapien scene, cleaning up all objects and resources.

    Args:
        scene: The scene object to reset. If None, all scenes will be cleaned up.
    """
    if scene is not None:
        try:
            # Remove all actors from the scene
            for actor in scene.get_all_actors():
                scene.remove_actor(actor)
            for entity in scene.get_entities():
                scene.remove_entity(entity)
            for camera in scene.get_cameras():
                scene.remove_camera(camera)
        except Exception as e:
            print(f"Error occurred while cleaning up scene objects: {e}")

    # Force garbage collection
    gc.collect()


def destroy_sapien_objects(*objects) -> None:
    """
    Safely delete sapien objects.

    Args:
        *objects: The sapien objects to delete.
    """
    for obj in objects:
        if obj is not None:
            try:
                del obj
            except Exception as e:
                print(f"Error occurred while deleting object: {e}")

    # Force garbage collection
    gc.collect()


def create_clean_sapien_scene(
    width: int = 600,
    height: int = 600,
    use_ray_tracing: bool = True,
    ray_tracing_samples: int = 16,
    ray_tracing_depth: int = 8
):
    """
    Create a brand new clean sapien scene.

    Args:
        width: Render width.
        height: Render height.
        use_ray_tracing: Whether to use ray tracing.
        ray_tracing_samples: Number of ray tracing samples.
        ray_tracing_depth: Ray tracing depth.

    Returns:
        scene: The newly created scene.
        camera: The newly created camera.
    """
    # Set rendering parameters
    if use_ray_tracing:
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(ray_tracing_samples)
        sapien.render.set_ray_tracing_path_depth(ray_tracing_depth)
        sapien.render.set_ray_tracing_denoiser("oidn")
    else:
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")

    # Create new scene
    scene = sapien.Scene()

    # Add ground
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Add lights
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)

    # Add environment lighting
    from sapien.asset import create_dome_envmap
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )

    if use_ray_tracing:
        scene.add_area_light_for_ray_tracing(
            sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
        )

    # Add camera
    camera = scene.add_camera(
        name="render_cam", width=width, height=height, fovy=1, near=0.1, far=10
    )
    camera.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    return scene, camera


def safe_render_robot(
    scene: sapien.Scene,
    camera: sapien.render.RenderCameraComponent,
    robot: sapien.physx.PhysxArticulation,
    qpos: np.ndarray,
    retargeting_joint_names: list
) -> np.ndarray:
    """
    Safely render the robot, with error handling.

    Args:
        scene: The sapien scene.
        camera: The render camera.
        robot: The robot object.
        qpos: Joint positions.
        retargeting_joint_names: List of joint names.

    Returns:
        rgb: The rendered RGB image.
    """
    try:
        # Map joint order
        sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        retargeting_to_sapien = np.array(
            [retargeting_joint_names.index(name) for name in sapien_joint_names]
        ).astype(int)

        # Set robot pose
        robot.set_qpos(np.array(qpos)[retargeting_to_sapien])

        # Render
        scene.update_render()
        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        return rgb

    except Exception as e:
        print(f"Error occurred while rendering robot: {e}")
        # Return a black image as fallback
        return np.zeros((camera.get_height(), camera.get_width(), 3), dtype=np.uint8)


def cleanup_sapien_resources():
    """
    Clean up all sapien-related resources.
    """
    try:
        # Try to destroy all scenes
        sapien.destroy_all_scenes()
    except:
        pass

    # Force garbage collection
    gc.collect()

    # Wait a short time to allow resources to be released
    import time
    time.sleep(0.1)
