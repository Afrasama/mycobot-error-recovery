import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from perception.segmentation import get_relative_pixel_error_overhead_and_rgb
from reflection.reflection_agent import ReflectionAgent

# optional offline "VLM-like" vision classifier (runs fully offline after training)
USE_OFFLINE_VISION_CLASSIFIER = True
offline_classifier = None

# enable/disable reflection (baseline vs reflection)
ENABLE_REFLECTION = True


# ---------------- CONNECT ----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

p.setPhysicsEngineParameter(numSolverIterations=150)
p.setPhysicsEngineParameter(fixedTimeStep=1 / 240)


# ---------------- PLANE ----------------
plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=1.5)


# ---------------- LOAD ROBOT ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

# try loading offline classifier (does nothing if deps/weights missing)
if USE_OFFLINE_VISION_CLASSIFIER:
    try:
        from perception.offline_vision_classifier import OfflineVisionClassifier

        _offline_model_path = os.path.join(
            BASE_DIR, "models", "offline_vlm", "tinycnn_direction.pt"
        )
        offline_classifier = OfflineVisionClassifier(model_path=_offline_model_path)
        print("Offline vision classifier loaded:", _offline_model_path)
    except Exception as e:
        offline_classifier = None
        print("Offline vision classifier disabled:", e)

robot = p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)


# ---------------- FIND END EFFECTOR ----------------
ee_index = None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, i)[12].decode() == "link6":
        ee_index = i
        break

print("End effector index:", ee_index)


# ---------------- SIMPLE GRIPPER ----------------
gripper = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.01]),
    baseVisualShapeIndex=p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.01],
        rgbaColor=[0.2, 0.2, 0.2, 1])
)

p.createConstraint(
    robot, ee_index,
    gripper, -1,
    p.JOINT_FIXED,
    [0, 0, 0],
    [0, 0, 0.04],
    [0, 0, 0]
)


# ---------------- CUBE ----------------
cube = p.loadURDF("cube_small.urdf", [0.3, 0.1, 0.02])
p.changeDynamics(
    cube, -1,
    lateralFriction=1.5,
    linearDamping=0.4,
    angularDamping=0.4,
    restitution=0.0
)

goal_position = np.array([0.45, -0.15, 0.02])


# ---------------- CAMERA (DEBUG VIEW) ----------------
p.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[0.3, 0.1, 0.08]
)


# ---------------- POLICY ----------------
policy = {
    "approach_height": 0.12,
    "grasp_height": 0.03,
    "lift_height": 0.20,
    "release_delay": 60,
    "x_offset": 0.0,
    "y_offset": 0.0,
}

max_retries = 10
retry_count = 0

inject_failure = True
perception_noise_scale = 0.025


# ---------------- SMOOTH MOTION ----------------
def smooth_move(target_pos, steps=180):
    joint_indices = []
    current_positions = []

    for j in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, j)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_indices.append(j)
            current_positions.append(p.getJointState(robot, j)[0])

    target_positions = p.calculateInverseKinematics(
        robot,
        ee_index,
        target_pos.tolist(),
        maxNumIterations=200
    )

    for step in range(steps):
        alpha = step / steps

        for idx, j in enumerate(joint_indices):
            interpolated = (
                (1 - alpha) * current_positions[idx]
                + alpha * target_positions[j]
            )

            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                interpolated,
                force=250,
                positionGain=0.5,
                velocityGain=1.0
            )

        p.stepSimulation()
        time.sleep(1 / 240)


# ---------------- STATE MACHINE ----------------
state = "plan"
timer = 0
stable_counter = 0
constraint_id = None

# reflection agent (agentic reasoning layer)
# swap_axes=False matches the direct pixel_error_x/y mapping used earlier.
agent = ReflectionAgent(scale=0.0002, max_step=0.02, swap_axes=False)

# ---------------- LOGGING ----------------
attempt_distances = []

while p.isConnected():

    timer += 1
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    cube_pos = np.array(cube_pos)

    if state == "plan":

        print("\nPlanning attempt", retry_count + 1)

        perceived_cube_pos = cube_pos.copy()
        perceived_cube_pos[0] += policy["x_offset"]
        perceived_cube_pos[1] += policy["y_offset"]

        if inject_failure:
            print("Injecting perception error")
            perceived_cube_pos[0] += np.random.uniform(
                -perception_noise_scale, perception_noise_scale
            )
            perceived_cube_pos[1] += np.random.uniform(
                -perception_noise_scale, perception_noise_scale
            )

        approach_target = perceived_cube_pos.copy()
        approach_target[2] += policy["approach_height"]

        grasp_target = perceived_cube_pos.copy()
        grasp_target[2] += policy["grasp_height"]

        state = "approach"
        timer = 0

    elif state == "approach":
        smooth_move(approach_target)
        state = "descend"
        timer = 0

    elif state == "descend":
        smooth_move(grasp_target)
        state = "grasp"
        timer = 0

    elif state == "grasp":

        grip_pos = p.getLinkState(robot, ee_index)[0]
        cube_pos, cube_orn = p.getBasePositionAndOrientation(cube)

        dist = np.linalg.norm(np.array(grip_pos) - np.array(cube_pos))

        if dist < 0.08:
            relative_offset = [
                cube_pos[0] - grip_pos[0],
                cube_pos[1] - grip_pos[1],
                cube_pos[2] - grip_pos[2],
            ]

            constraint_id = p.createConstraint(
                gripper, -1,
                cube, -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0],
                relative_offset
            )

            p.changeConstraint(constraint_id, maxForce=300)

            print("Object grasped (stable)")
            state = "lift"
            timer = 0

        else:
            print("Grasp failed")
            state = "analyze"

    elif state == "lift":
        lift_target = cube_pos.copy()
        lift_target[2] += policy["lift_height"]
        smooth_move(lift_target)
        state = "place_above"
        timer = 0

    elif state == "place_above":
        above_goal = goal_position.copy()
        above_goal[2] += policy["approach_height"]
        smooth_move(above_goal)
        state = "lower_to_place"
        timer = 0

    elif state == "lower_to_place":
        place_target = goal_position.copy()
        place_target[2] += 0.025
        smooth_move(place_target)
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
        speed = np.linalg.norm(lin_vel)
        distance_to_goal = np.linalg.norm(cube_pos - goal_position)

        print("Distance:", round(distance_to_goal, 3),
              "Speed:", round(speed, 3))

        if distance_to_goal < 0.10 and speed < 0.05:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter > 30:
            print("SUCCESS → task completed")
            state = "done"
            # log final distance for this successful attempt
            attempt_distances.append(distance_to_goal)

        elif timer > 180:
            # log final distance for this failed attempt
            attempt_distances.append(distance_to_goal)
            state = "analyze"

    elif state == "analyze":

        if retry_count >= max_retries:
            print("Max retries reached")
            state = "done"
            continue

        print("\n========== VISUAL REFLECTION ==========")

        # compute cube-vs-gripper relative pixel error from overhead camera
        error, rgb = get_relative_pixel_error_overhead_and_rgb(
            target_body_id=cube,
            reference_body_id=gripper,
            verbose=False,
        )

        if error is None:
            print("Cube not visible → cannot correct")
        else:
            pixel_error_x, pixel_error_y = error

            # offline vision "explanation" from RGB (optional)
            if offline_classifier is not None:
                try:
                    pred = offline_classifier.predict(rgb)
                    print(
                        f"OfflineVLM: {pred.label} (conf={pred.confidence:.2f})"
                    )
                except Exception as e:
                    print("OfflineVLM prediction failed:", e)

            if ENABLE_REFLECTION:
                # use agentic reflection to decide correction
                result = agent.reflect(
                    {
                        "pixel_error_x": float(pixel_error_x),
                        "pixel_error_y": float(pixel_error_y),
                        "retry_count": int(retry_count),
                    }
                )

                explanation = result["explanation"]
                correction_x = result["action"]["adjust_x"]
                correction_y = result["action"]["adjust_y"]

                print("Agent explanation:", explanation)
                print("Applying correction (m):",
                      round(correction_x, 4),
                      round(correction_y, 4))

                policy["x_offset"] += correction_x
                policy["y_offset"] += correction_y

                print("Updated offsets:",
                      round(policy["x_offset"], 4),
                      round(policy["y_offset"], 4))
            else:
                print("Reflection disabled → baseline mode (no correction applied)")

        print("======================================\n")

        inject_failure = False
        retry_count += 1
        state = "plan"
        timer = 0

    elif state == "done":
        print("Terminating simulation.")
        break

    p.stepSimulation()
    time.sleep(1 / 240)

p.disconnect()

# ---------------- PLOT RESULTS ----------------
if attempt_distances:
    attempts = np.arange(1, len(attempt_distances) + 1)
    plt.figure()
    plt.plot(attempts, attempt_distances, marker="o")
    plt.xlabel("Attempt")
    plt.ylabel("Final distance to goal (m)")
    plt.title("Distance to goal vs. attempt with visual reflection")
    plt.grid(True)
    plt.tight_layout()

    # Save results so the script never blocks on a GUI plot window.
    out_dir = os.path.join(BASE_DIR, "data", "plots")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(out_dir, f"distance_vs_attempt_{timestamp}.png")
    csv_path = os.path.join(out_dir, f"distance_vs_attempt_{timestamp}.csv")

    plt.savefig(png_path, dpi=200)
    plt.close()

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("attempt,final_distance_m\n")
        for a, d in zip(attempts.tolist(), attempt_distances):
            f.write(f"{a},{float(d)}\n")

    print("Saved plot:", png_path)
    print("Saved data:", csv_path)