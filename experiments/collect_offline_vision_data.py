import csv
import os
import random
from typing import Optional

import sys

import numpy as np
import pybullet as p
import pybullet_data

# Ensure project root is importable when running as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from perception.segmentation import (
    get_overhead_camera_image,
    compute_body_centroid,
)
from perception.offline_vision_classifier import discretize_from_pixel_error


def main(
    num_samples: int = 2000,
    out_dir: str = "data/offline_vlm",
    image_size: int = 128,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    labels_csv = os.path.join(out_dir, "labels.csv")

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    # load plane
    plane_id = p.loadURDF("plane.urdf")

    # load robot
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(base_dir, "urdf", "mycobot_320.urdf")
    robot = p.loadURDF(urdf_path, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    # find end effector
    ee_index = None
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[12].decode() == "link6":
            ee_index = i
            break
    if ee_index is None:
        raise RuntimeError("Could not find end-effector joint named link6")

    # create simple gripper visual/collision and attach
    gripper = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.01]
        ),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 0.02, 0.01],
            rgbaColor=[0.2, 0.2, 0.2, 1],
        ),
    )
    p.createConstraint(
        robot,
        ee_index,
        gripper,
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0.04],
        [0, 0, 0],
    )

    # cube
    cube = p.loadURDF("cube_small.urdf", [0.3, 0.1, 0.02])

    # collect
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "label",
                "pixel_error_x",
                "pixel_error_y",
                "centroid_x",
                "centroid_y",
            ],
        )
        writer.writeheader()

        for idx in range(num_samples):
            # randomize cube a little
            cube_x = 0.30 + np.random.uniform(-0.08, 0.08)
            cube_y = 0.10 + np.random.uniform(-0.08, 0.08)
            p.resetBasePositionAndOrientation(cube, [cube_x, cube_y, 0.02], [0, 0, 0, 1])

            # step once to update rendering
            p.stepSimulation()

            rgb, _, seg = get_overhead_camera_image(width=640, height=480)
            cube_centroid = compute_body_centroid(seg, cube)
            gripper_centroid = compute_body_centroid(seg, gripper)

            if cube_centroid is None or gripper_centroid is None:
                label = "NOT_VISIBLE"
                px, py = 0.0, 0.0
                cx, cy = "", ""
            else:
                # cube relative to gripper
                px = float(cube_centroid[0] - gripper_centroid[0])
                py = float(cube_centroid[1] - gripper_centroid[1])
                label = discretize_from_pixel_error(px, py, threshold=15.0)
                cx, cy = float(cube_centroid[0]), float(cube_centroid[1])

            # save resized image (small)
            from PIL import Image

            out_name = f"{idx:06d}.png"
            out_path = os.path.join(images_dir, out_name)
            img = Image.fromarray(rgb.astype(np.uint8), mode="RGB").resize(
                (image_size, image_size)
            )
            img.save(out_path)

            writer.writerow(
                {
                    "filename": f"images/{out_name}",
                    "label": label,
                    "pixel_error_x": float(px),
                    "pixel_error_y": float(py),
                    "centroid_x": cx,
                    "centroid_y": cy,
                }
            )

            if (idx + 1) % 200 == 0:
                print(f"Collected {idx+1}/{num_samples}")

    p.disconnect()
    print(f"Saved dataset to: {out_dir}")
    print(f"Labels: {labels_csv}")


if __name__ == "__main__":
    main()

