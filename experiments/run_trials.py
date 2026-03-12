import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

# Ensure project root is importable when running as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from perception.segmentation import get_relative_pixel_error_overhead  # cube vs gripper
from reflection.reflection_agent import ReflectionAgent


@dataclass
class TrialConfig:
    trials: int = 5
    perception_bias_scale: float = 0.06  # meters (fixed bias per episode)
    max_retries: int = 10
    enable_reflection: bool = True
    seed: int = 0
    out_csv: str = "data/plots/trials_results.csv"


def smooth_move(robot_id: int, ee_index: int, target_pos: np.ndarray, steps: int = 120):
    joint_indices = []
    current_positions = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_indices.append(j)
            current_positions.append(p.getJointState(robot_id, j)[0])

    target_positions = p.calculateInverseKinematics(
        robot_id, ee_index, target_pos.tolist(), maxNumIterations=200
    )

    for step in range(steps):
        alpha = step / steps
        for idx, j in enumerate(joint_indices):
            interpolated = (1 - alpha) * current_positions[idx] + alpha * target_positions[j]
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                interpolated,
                force=250,
                positionGain=0.5,
                velocityGain=1.0,
            )
        p.stepSimulation()


def run_episode(
    episode_seed: int,
    enable_reflection: bool,
    perception_bias_scale: float,
    max_retries: int,
) -> Dict[str, float]:
    random.seed(episode_seed)
    np.random.seed(episode_seed)

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setPhysicsEngineParameter(fixedTimeStep=1 / 240)

    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=1.5)

    base_dir = PROJECT_ROOT
    urdf_path = os.path.join(base_dir, "urdf", "mycobot_320.urdf")
    robot = p.loadURDF(urdf_path, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    ee_index = None
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[12].decode() == "link6":
            ee_index = i
            break
    if ee_index is None:
        raise RuntimeError("End effector link6 not found")

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
        robot, ee_index, gripper, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.04], [0, 0, 0]
    )

    cube = p.loadURDF("cube_small.urdf", [0.3, 0.1, 0.02])
    p.changeDynamics(
        cube,
        -1,
        lateralFriction=1.5,
        linearDamping=0.4,
        angularDamping=0.4,
        restitution=0.0,
    )

    goal_position = np.array([0.45, -0.15, 0.02], dtype=np.float32)

    policy = {
        "approach_height": 0.12,
        "grasp_height": 0.03,
        "lift_height": 0.20,
        "release_delay": 60,
        "x_offset": 0.0,
        "y_offset": 0.0,
    }

    # fixed perception bias per episode (what reflection tries to learn/cancel)
    bias_x = float(np.random.uniform(-perception_bias_scale, perception_bias_scale))
    bias_y = float(np.random.uniform(-perception_bias_scale, perception_bias_scale))

    agent = ReflectionAgent(scale=0.0002, max_step=0.02, swap_axes=False)

    state = "plan"
    timer = 0
    stable_counter = 0
    constraint_id = None
    retries = 0
    attempt_distances: List[float] = []

    # safety cap so episodes always terminate
    max_steps = 240 * 20  # 20 seconds sim time
    steps = 0

    while steps < max_steps:
        steps += 1
        timer += 1

        cube_pos, _ = p.getBasePositionAndOrientation(cube)
        cube_pos = np.array(cube_pos, dtype=np.float32)

        if state == "plan":
            perceived = cube_pos.copy()
            perceived[0] += policy["x_offset"] + bias_x
            perceived[1] += policy["y_offset"] + bias_y

            approach_target = perceived.copy()
            approach_target[2] += policy["approach_height"]
            grasp_target = perceived.copy()
            grasp_target[2] += policy["grasp_height"]

            state = "approach"
            timer = 0

        elif state == "approach":
            smooth_move(robot, ee_index, approach_target)
            state = "descend"
            timer = 0

        elif state == "descend":
            smooth_move(robot, ee_index, grasp_target)
            state = "grasp"
            timer = 0

        elif state == "grasp":
            grip_pos = p.getLinkState(robot, ee_index)[0]
            cube_pos_now, _ = p.getBasePositionAndOrientation(cube)
            dist = float(np.linalg.norm(np.array(grip_pos) - np.array(cube_pos_now)))

            if dist < 0.08:
                relative_offset = [
                    cube_pos_now[0] - grip_pos[0],
                    cube_pos_now[1] - grip_pos[1],
                    cube_pos_now[2] - grip_pos[2],
                ]
                constraint_id = p.createConstraint(
                    gripper, -1, cube, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], relative_offset
                )
                p.changeConstraint(constraint_id, maxForce=300)
                state = "lift"
                timer = 0
            else:
                state = "analyze"

        elif state == "lift":
            lift_target = cube_pos.copy()
            lift_target[2] += policy["lift_height"]
            smooth_move(robot, ee_index, lift_target)
            state = "place_above"
            timer = 0

        elif state == "place_above":
            above_goal = goal_position.copy()
            above_goal[2] += policy["approach_height"]
            smooth_move(robot, ee_index, above_goal)
            state = "lower_to_place"
            timer = 0

        elif state == "lower_to_place":
            place_target = goal_position.copy()
            place_target[2] += 0.025
            smooth_move(robot, ee_index, place_target)
            state = "release"
            timer = 0

        elif state == "release":
            if timer > policy["release_delay"]:
                if constraint_id is not None:
                    p.removeConstraint(constraint_id)
                    constraint_id = None
                state = "observe"
                timer = 0
                stable_counter = 0

        elif state == "observe":
            lin_vel, _ = p.getBaseVelocity(cube)
            speed = float(np.linalg.norm(lin_vel))
            distance_to_goal = float(np.linalg.norm(cube_pos - goal_position))

            if distance_to_goal < 0.10 and speed < 0.05:
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter > 30:
                attempt_distances.append(distance_to_goal)
                state = "done"
            elif timer > 180:
                attempt_distances.append(distance_to_goal)
                state = "analyze"

        elif state == "analyze":
            if retries >= max_retries:
                state = "done"
            else:
                # use overhead relative pixel error (cube relative to gripper)
                err = get_relative_pixel_error_overhead(
                    target_body_id=cube,
                    reference_body_id=gripper,
                    verbose=False,
                )
                if enable_reflection and err is not None:
                    px, py = err
                    result = agent.reflect(
                        {
                            "pixel_error_x": float(px),
                            "pixel_error_y": float(py),
                            "retry_count": int(retries),
                        }
                    )
                    policy["x_offset"] += float(result["action"]["adjust_x"])
                    policy["y_offset"] += float(result["action"]["adjust_y"])

                retries += 1
                state = "plan"
                timer = 0

        elif state == "done":
            break

        p.stepSimulation()

    p.disconnect()

    success = 1.0 if (len(attempt_distances) > 0 and attempt_distances[-1] < 0.10) else 0.0
    final_distance = float(attempt_distances[-1]) if attempt_distances else float("nan")
    attempts = float(len(attempt_distances)) if attempt_distances else float(retries + 1)

    return {
        "success": success,
        "final_distance_m": final_distance,
        "attempts_logged": float(len(attempt_distances)),
        "retries_used": float(retries),
        "bias_x": bias_x,
        "bias_y": bias_y,
        "enable_reflection": 1.0 if enable_reflection else 0.0,
    }


def run_suite(cfg: TrialConfig) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for i in range(cfg.trials):
        ep_seed = cfg.seed + i
        r = run_episode(
            episode_seed=ep_seed,
            enable_reflection=cfg.enable_reflection,
            perception_bias_scale=cfg.perception_bias_scale,
            max_retries=cfg.max_retries,
        )
        r["trial"] = float(i)
        r["seed"] = float(ep_seed)
        results.append(r)
        print(f"Trial {i+1}/{cfg.trials} success={int(r['success'])} final_d={r['final_distance_m']:.3f}")
    return results


def save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    succ = np.array([r["success"] for r in rows], dtype=np.float32)
    d = np.array([r["final_distance_m"] for r in rows], dtype=np.float32)
    d = d[np.isfinite(d)]
    return {
        "success_rate": float(succ.mean()) if len(succ) else 0.0,
        "mean_final_distance_m": float(d.mean()) if len(d) else float("nan"),
    }


def main():
    # reflection
    cfg_reflect = TrialConfig(
        enable_reflection=True,
        out_csv="data/plots/trials_reflection.csv",
    )
    reflect_rows = run_suite(cfg_reflect)
    save_csv(os.path.join(PROJECT_ROOT, cfg_reflect.out_csv), reflect_rows)
    reflect_summary = summarize(reflect_rows)

    # baseline
    cfg_base = TrialConfig(
        enable_reflection=False,
        out_csv="data/plots/trials_baseline.csv",
    )
    base_rows = run_suite(cfg_base)
    save_csv(os.path.join(PROJECT_ROOT, cfg_base.out_csv), base_rows)
    base_summary = summarize(base_rows)

    print("\n=== SUMMARY ===")
    print("Baseline:", base_summary)
    print("Reflection:", reflect_summary)
    print("\nSaved:")
    print(os.path.join(PROJECT_ROOT, cfg_base.out_csv))
    print(os.path.join(PROJECT_ROOT, cfg_reflect.out_csv))


if __name__ == "__main__":
    main()

