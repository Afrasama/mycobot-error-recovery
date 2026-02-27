import os
import pybullet as p
import numpy as np
import datetime
from PIL import Image


# -------- CREATE IMAGE FOLDER --------
IMAGE_DIR = "reflection_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


# -------- CAPTURE IMAGE --------
def capture_image():
    """
    Capture camera image from PyBullet and save it.
    """

    width, height = 640, 480

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.3, 0.1, 0.08],
        distance=1.2,
        yaw=45,
        pitch=-35,
        roll=0,
        upAxisIndex=2
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.1,
        farVal=2.0
    )

    _, _, rgb_img, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix
    )

    # Convert to numpy array properly
    rgb_array = np.array(rgb_img, dtype=np.uint8)
    rgb_array = rgb_array.reshape((height, width, 4))
    rgb_array = rgb_array[:, :, :3]  # remove alpha channel

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(IMAGE_DIR, f"reflection_{timestamp}.png")

    Image.fromarray(rgb_array).save(filename)

    print("Image saved:", filename)

    return filename


# -------- MOCK VLM REASONING --------
def mock_vlm_reason(attempt_id):

    explanations = [
        "Release timing too early.",
        "Grasp height too high; object slipped.",
        "Object not centered during descent.",
        "End effector misaligned in X direction."
    ]

    explanation = explanations[attempt_id % len(explanations)]
    print("VLM Explanation:", explanation)

    return explanation


# -------- POLICY UPDATE --------
def update_policy(policy, explanation):

    print("Updating policy...")

    if "Release timing" in explanation:
        policy["release_delay"] += 30
        print("Policy updated → increasing release delay")

    elif "Grasp height too high" in explanation:
        policy["grasp_height"] -= 0.005
        print("Policy updated → lowering grasp height")

    elif "not centered" in explanation:
        policy["grasp_height"] -= 0.002
        print("Policy updated → fine lowering grasp height")

    elif "misaligned" in explanation:
        policy["approach_height"] += 0.01
        print("Policy updated → increasing approach height")

    return policy


# -------- MASTER REFLECTION --------
def reflect_and_update(policy, attempt_id):

    print("Capturing camera image...")
    capture_image()

    print("Running VLM reasoning...")
    explanation = mock_vlm_reason(attempt_id)

    policy = update_policy(policy, explanation)

    return policy