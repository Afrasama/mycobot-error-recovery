import pybullet as p
import pybullet_data
import time
import os

# ---------------- CONNECT ----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

p.loadURDF("plane.urdf")

BASE = os.path.dirname(os.path.abspath(__file__))

robot = p.loadURDF(
    os.path.join(BASE,"urdf","mycobot_320.urdf"),
    useFixedBase=True
)

# ---------- ADD OBJECT (VERY IMPORTANT) ----------
cube = p.loadURDF(
    "cube_small.urdf",
    basePosition=[0.28,0.0,0.02]
)

print("cube id:", cube)

# ---------- FIND END EFFECTOR ----------
ee = None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot,i)[12].decode()=="link6":
        ee=i

print("ee index:",ee)

target=[0.25,0.0,0.15]

print("""
WASDQE = move end effector
""")

# ---------------- LOOP ----------------
while True:

    keys=p.getKeyboardEvents()
    step=0.005

    if ord('a') in keys: target[0]-=step
    if ord('d') in keys: target[0]+=step
    if ord('w') in keys: target[1]+=step
    if ord('s') in keys: target[1]-=step
    if ord('q') in keys: target[2]+=step
    if ord('e') in keys: target[2]-=step

    joints=p.calculateInverseKinematics(robot,ee,target)

    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot,j)[2]==p.JOINT_REVOLUTE:
            p.setJointMotorControl2(robot,j,p.POSITION_CONTROL,
                                    targetPosition=joints[j],force=800)

    p.stepSimulation()
    time.sleep(1/240)
