import pybullet as p
import pybullet_data
import time
import os

# connect (GUI)
physics_id = p.connect(p.GUI)
assert physics_id >= 0, "failed to connect to pybullet"

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

# load robot
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

robot = p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

# find end-effector
ee_index = None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, i)[12].decode() == "link6":
        ee_index = i
        break

print("end-effector index:", ee_index)

# target
target_pos = [0.25, 0.0, 0.15]
step = 0.01

print("""
keyboard controls:
W/S : +Y / -Y
A/D : -X / +X
Q/E : +Z / -Z
R   : reset
ESC : quit
""")

# ---------------- MAIN LOOP ----------------
while p.isConnected():

    p.stepSimulation()

    try:
        keys = p.getKeyboardEvents()
    except:
        # physics server closed unexpectedly
        print("physics server disconnected")
        break

    if ord('a') in keys:
        target_pos[0] -= step
    if ord('d') in keys:
        target_pos[0] += step
    if ord('w') in keys:
        target_pos[1] += step
    if ord('s') in keys:
        target_pos[1] -= step
    if ord('q') in keys:
        target_pos[2] += step
    if ord('e') in keys:
        target_pos[2] -= step

    if ord('r') in keys:
        target_pos = [0.25, 0.0, 0.15]

    # esc key = 27
    if 27 in keys:
        print("exiting...")
        break

    joint_positions = p.calculateInverseKinematics(
        robot,
        ee_index,
        target_pos,
        maxNumIterations=200
    )

    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, j)[2] == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[j],
                force=800
            )

    time.sleep(1 / 240)

p.disconnect()
