import pybullet as p
import pybullet_data
import time
import os

# absolute path handling (windows-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

print("urdf path:", URDF_PATH)

# start pybullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# load ground
p.loadURDF("plane.urdf")

# load robot
robot = p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

print("robot id:", robot)

# -----------------------------
# create joint sliders
# -----------------------------
joint_sliders = []

num_joints = p.getNumJoints(robot)
print("number of joints:", num_joints)

for j in range(num_joints):
    info = p.getJointInfo(robot, j)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]

    # only revolute joints
    if joint_type == p.JOINT_REVOLUTE:
        lower = info[8]
        upper = info[9]

        slider = p.addUserDebugParameter(
            joint_name,
            lower,
            upper,
            0.0
        )
        joint_sliders.append((j, slider))

        print(f"added slider for joint {j}: {joint_name}")

# -----------------------------
# control loop
# -----------------------------
while True:
    for joint_index, slider_id in joint_sliders:
        target_pos = p.readUserDebugParameter(slider_id)
        p.setJointMotorControl2(
            bodyIndex=robot,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_pos,
            force=500
        )

    p.stepSimulation()
    time.sleep(1.0 / 240.0)
