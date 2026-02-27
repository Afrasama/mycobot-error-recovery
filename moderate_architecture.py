import pybullet as p
import pybullet_data
import numpy as np
import time
import os

# ---------------- CONNECT ----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/240)
p.setRealTimeSimulation(0)

p.loadURDF("plane.urdf")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

robot = p.loadURDF(URDF_PATH, useFixedBase=True)

# Stable damping
for j in range(p.getNumJoints(robot)):
    p.changeDynamics(robot, j, linearDamping=0.08, angularDamping=0.08)

# ---------------- END EFFECTOR ----------------
ee_index = None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, i)[12].decode() == "link6":
        ee_index = i
        break

if ee_index is None:
    raise ValueError("link6 not found")

# ---------------- SIMPLE GRIPPER ----------------
gripper = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02]*3)
)

p.createConstraint(robot, ee_index, gripper, -1,
                   p.JOINT_FIXED, [0,0,0], [0,0,0.04], [0,0,0])

# ---------------- CUBE ----------------
cube_start = [
    np.random.uniform(0.22, 0.34),
    np.random.uniform(-0.12, 0.12),
    0.02
]
cube = p.loadURDF("cube_small.urdf", cube_start)

# ---------------- GOAL ----------------
goal_position = np.array([0.30, 0.0, 0.02])

goal_visual = p.createVisualShape(
    p.GEOM_SPHERE,
    radius=0.04,
    rgbaColor=[1,0,0,1]
)

p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=goal_visual,
    basePosition=goal_position.tolist()
)

# ---------------- FSM VARIABLES ----------------
state = "idle"
retry_offset = np.array([0.0, 0.0, 0.0])
retry_count = 0
max_retries = 6
constraint_id = None
timer = 0

current_target = np.array([0.25, 0.0, 0.25])

# ---------------- REFLECTION ----------------
def mock_reflection(cube_pos, goal_pos):
    error = cube_pos - goal_pos
    gain = 0.45
    correction = -gain * error
    correction = np.clip(correction, -0.07, 0.07)
    print("Reflection correction:", round(correction[0],3),
          round(correction[1],3))
    return np.array([correction[0], correction[1], 0])

print("Press SPACE to start")

# ---------------- MAIN LOOP ----------------
while p.isConnected():

    keys = p.getKeyboardEvents()
    if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
        if state == "idle":
            print("Starting task")
            state = "approach_cube"
            timer = 0

    timer += 1
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    cube_pos = np.array(cube_pos)

    # ---------------- FSM ----------------

    if state == "approach_cube":
        target = cube_pos + np.array([0,0,0.12])
        if np.linalg.norm(current_target - target) < 0.03:
            state = "grasp"
            timer = 0

    elif state == "grasp":
        grip_pos,_ = p.getBasePositionAndOrientation(gripper)
        if np.linalg.norm(np.array(grip_pos) - cube_pos) < 0.05:
            constraint_id = p.createConstraint(
                gripper,-1,cube,-1,
                p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0]
            )
            state = "lift"
            timer = 0

    elif state == "lift":
        target = cube_pos + np.array([0,0,0.25])
        if timer > 40:
            state = "move_to_goal"
            timer = 0

    elif state == "move_to_goal":
        target = goal_position + retry_offset + np.array([0,0,0.25])
        if timer > 60:
            state = "lower"
            timer = 0

    elif state == "lower":
        target = goal_position + retry_offset + np.array([0,0,0.06])
        if timer > 40:
            state = "release"
            timer = 0

    elif state == "release":
        if constraint_id is not None:
            p.removeConstraint(constraint_id)
            constraint_id = None
        state = "checking"
        timer = 0

    elif state == "checking":
        dist = np.linalg.norm(cube_pos - goal_position)
        if dist < 0.05:
            print("SUCCESS")
            state = "done"
        elif timer > 70:
            print("FAILED → Reflecting")
            retry_offset += mock_reflection(cube_pos, goal_position)
            retry_count += 1
            if retry_count < max_retries:
                state = "move_to_goal"
                timer = 0
            else:
                print("Max retries reached")
                state = "done"

    # ---------------- CONTROL ----------------
    if state not in ["idle", "done"]:
        current_target += (target - current_target) * 0.7

        joint_angles = p.calculateInverseKinematics(
            robot,
            ee_index,
            current_target.tolist()
        )

        for j in range(p.getNumJoints(robot)):
            if p.getJointInfo(robot, j)[2] == p.JOINT_REVOLUTE:
                p.setJointMotorControl2(
                    robot, j,
                    p.POSITION_CONTROL,
                    targetPosition=joint_angles[j],
                    force=400,
                    positionGain=0.5,
                    velocityGain=1.1
                )

    p.stepSimulation()
    # Balanced fast → slight sleep for stability
    time.sleep(1/480)