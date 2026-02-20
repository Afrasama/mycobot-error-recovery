import pybullet as p
import pybullet_data
import time
import os

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

robot = p.loadURDF(
    "D:/Internship/mycobot_320/urdf/mycobot_320.urdf",
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

print("robot loaded, id =", robot)

while True:
    time.sleep(1)
