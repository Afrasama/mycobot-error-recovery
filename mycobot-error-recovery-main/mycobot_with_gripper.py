import pybullet as p
import pybullet_data
import time
import os

# connect
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load robot
robot = p.loadURDF(
    os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf"),
    useFixedBase=True
)

# find end-effector (link6)
ee_index = None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, i)[12].decode() == "link6":
        ee_index = i
        break

print("end-effector index:", ee_index)

# load gripper
gripper = p.loadURDF(
    os.path.join(BASE_DIR, "urdf", "simple_gripper.urdf"),
    basePosition=[0, 0, 0]
)

# attach gripper to link6
p.createConstraint(
    parentBodyUniqueId=robot,
    parentLinkIndex=ee_index,
    childBodyUniqueId=gripper,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0.03],
    childFramePosition=[0, 0, 0]
)

# ---- IK target ----
target = [0.25, 0.0, 0.15]

# gripper state
grip_open = True

print("""
controls:
W/S/A/D/Q/E : move end-effector
G           : toggle gripper open/close
ESC         : quit
""")

while True:
    keys = p.getKeyboardEvents()

    step = 0.005
    if ord('a') in keys: target[0] -= step
    if ord('d') in keys: target[0] += step
    if ord('w') in keys: target[1] += step
    if ord('s') in keys: target[1] -= step
    if ord('q') in keys: target[2] += step
    if ord('e') in keys: target[2] -= step

    if ord('g') in keys:
        grip_open = not grip_open
        time.sleep(0.2)

    if 27 in keys:
        break

    # IK
    joints = p.calculateInverseKinematics(
        robot,
        ee_index,
        target
    )

    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, j)[2] == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=joints[j],
                force=800
            )

    # gripper control
    finger_pos = 0.03 if grip_open else 0.0
    p.setJointMotorControl2(gripper, 0, p.POSITION_CONTROL, finger_pos, force=20)
    p.setJointMotorControl2(gripper, 1, p.POSITION_CONTROL, finger_pos, force=20)

    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
