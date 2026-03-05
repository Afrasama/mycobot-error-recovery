import pybullet as p
import pybullet_data
import time
import os
import cv2
import numpy as np

# ---------------- CONNECT ----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Turbo physics
p.setPhysicsEngineParameter(numSolverIterations=80)
p.setPhysicsEngineParameter(fixedTimeStep=1/120)

p.loadURDF("plane.urdf")

# ---------------- LOAD ROBOT ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

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

p.createConstraint(robot, ee_index, gripper, -1,
                   p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.04], [0, 0, 0])

# ---------------- CUBE ----------------
cube = p.loadURDF("cube_small.urdf", [0.3, 0.1, 0.02])
goal_position = np.array([0.3, 0.1, 0.02])

goal_visual = p.createVisualShape(
    p.GEOM_SPHERE,
    radius=0.035,
    rgbaColor=[1, 0, 0, 1]
)

p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=goal_visual,
    basePosition=goal_position.tolist()
)

# ---------------- PARAMETERS ----------------
retry_offset = np.array([0.0, 0.0, 0.0])
retry_count = 0
max_retries = 8

constraint_id = None
state = "manual"
timer = 0
stable_counter = 0

# ---------------- MOCK REFLECTION ----------------
def mock_reflection(cube_pos, goal_pos):
    error = cube_pos - goal_pos
    gain = 0.6

    dx = -gain * error[0]
    dy = -gain * error[1]
    dz = 0.01

    max_step = 0.1
    dx = np.clip(dx, -max_step, max_step)
    dy = np.clip(dy, -max_step, max_step)

    print("Correction:", round(dx,4), round(dy,4), dz)

    return np.array([dx, dy, dz])

# ---------------- TARGET CONTROL ----------------
current_target = np.array([0.30, 0.10, 0.08])

x_slider = p.addUserDebugParameter("X", -0.4, 0.4, 0.30)
y_slider = p.addUserDebugParameter("Y", -0.4, 0.4, 0.10)
z_slider = p.addUserDebugParameter("Z", 0.02, 0.5, 0.08)

print("\nPRESS SPACE TO START\n")

# ---------------- MAIN LOOP ----------------
while p.isConnected():

    timer += 1

    target = np.array([
        p.readUserDebugParameter(x_slider),
        p.readUserDebugParameter(y_slider),
        p.readUserDebugParameter(z_slider)
    ]) + retry_offset

    # Instant movement (NO smoothing)
    current_target = target

    joint_angles = p.calculateInverseKinematics(
        robot,
        ee_index,
        current_target.tolist(),
        maxNumIterations=40
    )

    for j in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, j)
        joint_type = joint_info[2]

        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.setJointMotorControl2(
                robot, j, p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=600,
                positionGain=1.0,
                velocityGain=2.0
            )

    keys = p.getKeyboardEvents()

    # START TASK
    if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
        print("Starting task...")
        retry_count = 0
        retry_offset = np.array([0.0,0.0,0.0])
        state = "grab"

    # GRAB
    if state == "grab":
        grip_pos, _ = p.getBasePositionAndOrientation(gripper)
        cube_pos, _ = p.getBasePositionAndOrientation(cube)
        dist = np.linalg.norm(np.array(grip_pos) - np.array(cube_pos))

        if dist < 0.07:
            constraint_id = p.createConstraint(
                gripper, -1, cube, -1,
                p.JOINT_FIXED, [0, 0, 0],
                [0, 0, 0], [0, 0, 0]
            )
            state = "release_timer"
            timer = 0

    # RELEASE QUICKLY
    if state == "release_timer" and timer > 20:
        p.removeConstraint(constraint_id)
        constraint_id = None
        state = "checking"
        timer = 0
        stable_counter = 0

    # CHECK RESULT
    if state == "checking":

        cube_pos, _ = p.getBasePositionAndOrientation(cube)
        cube_pos = np.array(cube_pos)

        lin_vel, _ = p.getBaseVelocity(cube)
        speed = np.linalg.norm(lin_vel)

        distance_to_goal = np.linalg.norm(cube_pos - goal_position)

        if distance_to_goal < 0.05 and speed < 0.02:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter > 20:
            print("SUCCESS → Object placed correctly")
            state = "manual"

        elif timer > 40:

            if retry_count < max_retries:
                print("FAILED → Reflecting")
                correction = mock_reflection(cube_pos, goal_position)
                retry_offset += correction
                retry_count += 1
                state = "grab"
                timer = 0
            else:
                print("Max retries reached")
                state = "manual"

    p.stepSimulation()

p.disconnect()