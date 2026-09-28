"""ASPIRE/LIBERO agent-generated-code example.

This file is intentionally shaped like a saved trial ``code.py``.  It is not a
standalone host-side script: ASPIRE injects the perception and robot-control
functions used below into the generated program's globals before execution.

Example task: pick up a cream-cheese box and place it in a bowl.
"""

import numpy as np


def best_text_mask(rgb, prompt):
    """Ask SAM3 for instances and keep its highest-confidence mask."""
    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        raise RuntimeError(f"SAM3 found no instance for {prompt!r}")
    return max(masks, key=lambda item: item["score"])


def camera_data(observation):
    camera = observation["agentview"]
    depth = camera["images"]["depth"]
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return (
        camera["images"]["rgb"],
        depth,
        camera["intrinsics"],
        camera["pose_mat"],
    )


def run():
    # 1. Observe and find the object in the external camera image.
    rgb, depth, intrinsics, camera_to_world = camera_data(get_observation())
    object_mask = best_text_mask(rgb, "cream cheese box")

    # 2. Convert masked depth pixels to 3-D points.  This is observation-based;
    #    generated programs must not read MuJoCo's ground-truth object position.
    object_points = mask_to_world_points(
        object_mask["mask"].astype(np.uint8),
        depth,
        intrinsics,
        camera_to_world,
    )
    object_center = object_points.mean(axis=0)
    print("observed object center:", object_center)

    # 3. Contact-GraspNet proposes end-effector transforms in camera coordinates.
    grasp_poses_camera, grasp_scores = plan_grasp(
        depth, intrinsics, object_mask["mask"]
    )
    if len(grasp_scores) == 0:
        raise RuntimeError("grasp planner returned no candidates")

    grasp_world, _ = select_top_down_grasp(
        grasp_poses_camera,
        grasp_scores,
        camera_to_world,
    )
    if grasp_world is None:
        best_index = int(np.argmax(grasp_scores))
        grasp_world = camera_to_world @ grasp_poses_camera[best_index]
    grasp_position, grasp_quaternion = decompose_transform(grasp_world)

    # 4. The API turns Cartesian poses into seven Franka joint angles, then the
    #    simulator controller repeatedly steps until those joints are reached.
    open_gripper()
    goto_pose(grasp_position, grasp_quaternion, z_approach=0.10)
    goto_pose(grasp_position, grasp_quaternion)
    close_gripper()

    lift_position = grasp_position + np.array([0.0, 0.0, 0.15])
    lift_joints = solve_ik(lift_position, grasp_quaternion)
    if lift_joints is None:
        raise RuntimeError("IK failed for lift pose")
    move_to_joints(lift_joints)

    # 5. Re-observe after moving: the camera view and scene may have changed.
    rgb2, depth2, intrinsics2, camera_to_world2 = camera_data(get_observation())
    target_mask = best_text_mask(rgb2, "bowl")
    target_points = mask_to_world_points(
        target_mask["mask"].astype(np.uint8),
        depth2,
        intrinsics2,
        camera_to_world2,
    )
    target_center = target_points.mean(axis=0)
    target_surface_z = target_points[:, 2].max()

    # 6. Move above the bowl, descend, and release.
    above_target = np.array([target_center[0], target_center[1], lift_position[2]])
    above_joints = solve_ik(above_target, grasp_quaternion)
    if above_joints is None:
        raise RuntimeError("IK failed above target")
    move_to_joints(above_joints)

    release_position = np.array(
        [target_center[0], target_center[1], target_surface_z + 0.03]
    )
    release_joints = solve_ik(release_position, grasp_quaternion)
    if release_joints is None:
        raise RuntimeError("IK failed for release pose")
    move_to_joints(release_joints)
    open_gripper()


run()
