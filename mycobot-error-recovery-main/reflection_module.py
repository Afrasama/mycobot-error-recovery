import os
import pybullet as p
import numpy as np
import datetime
from PIL import Image


# -------- IMAGE FOLDER --------
IMAGE_DIR = "reflection_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


# -------- WRIST CAMERA CAPTURE --------
def capture_image(robot_id, ee_index):

    width, height = 640, 480

    ee_state = p.getLinkState(robot_id, ee_index)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]

    rot_matrix = p.getMatrixFromQuaternion(ee_orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    camera_forward = rot_matrix[:, 2]
    camera_up = rot_matrix[:, 1]

    camera_pos = np.array(ee_pos) + 0.05 * camera_forward
    camera_target = camera_pos + 0.2 * camera_forward

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos.tolist(),
        cameraTargetPosition=camera_target.tolist(),
        cameraUpVector=camera_up.tolist()
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.01,
        farVal=2.0
    )

    _, _, rgb_img, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix
    )

    rgb_array = np.array(rgb_img, dtype=np.uint8)
    rgb_array = rgb_array.reshape((height, width, 4))
    rgb_array = rgb_array[:, :, :3]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(IMAGE_DIR, f"wrist_reflection_{timestamp}.png")

    Image.fromarray(rgb_array).save(filename)

    print("Wrist image saved:", filename)

    return filename


# -------- SIMPLE MOCK VLM --------
def mock_vlm_reason(attempt_id):

    explanations = [
        "x misalignment",
        "y misalignment",
        "grasp too high",
        "release too early"
    ]

    explanation = explanations[attempt_id % len(explanations)]
    print("VLM Explanation:", explanation)

    return explanation


# -------- POLICY UPDATE --------
def update_policy(policy, explanation):

    print("Updating policy...")

    if explanation == "x misalignment":
        policy["x_offset"] += 0.01
        print("→ shifting X offset")

    elif explanation == "y misalignment":
        policy["y_offset"] += 0.01
        print("→ shifting Y offset")

    elif explanation == "grasp too high":
        policy["grasp_height"] -= 0.005
        print("→ lowering grasp height")

    elif explanation == "release too early":
        policy["release_delay"] += 30
        print("→ increasing release delay")

    print("Updated policy:", policy)

    return policy


# -------- MASTER REFLECTION --------
def reflect_and_update(policy, retry_count, robot_id, ee_index):

    print("\n========== REFLECTION ==========")
    print("Attempt:", retry_count)

    # 1. capture wrist image
    capture_image(robot_id, ee_index)

    # 2. mock reasoning
    explanation = mock_vlm_reason(retry_count)

    # 3. update policy
    policy = update_policy(policy, explanation)

    print("================================\n")

    return policy