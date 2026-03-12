import pybullet as p
import pybullet_data
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ---------------- CONNECT ----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

p.loadURDF("plane.urdf")

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH=os.path.join(BASE_DIR,"urdf","mycobot_320.urdf")

robot=p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

# -------- JOINT DAMPING --------
for j in range(p.getNumJoints(robot)):
    p.changeDynamics(robot,j,linearDamping=0.04,angularDamping=0.04)

# ---------------- FIND END EFFECTOR ----------------
ee_index=None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot,i)[12].decode()=="link6":
        ee_index=i

print("end effector index:",ee_index)

# ---------------- SIMPLE GRIPPER ----------------
gripper=p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=p.createCollisionShape(
        p.GEOM_BOX,halfExtents=[0.02,0.02,0.01]),
    baseVisualShapeIndex=p.createVisualShape(
        p.GEOM_BOX,halfExtents=[0.02,0.02,0.01],
        rgbaColor=[0.2,0.2,0.2,1])
)

p.createConstraint(robot,ee_index,gripper,-1,p.JOINT_FIXED,[0,0,0],[0,0,0.04],[0,0,0])

# ---------------- CUBE + TARGET ----------------
cube=p.loadURDF("cube_small.urdf",[0.30,0.10,0.02])
p.changeDynamics(cube,-1,mass=0.2)

target_place=[0.15,-0.20,0.02]

# green target marker
vis=p.createVisualShape(p.GEOM_SPHERE,radius=0.03,rgbaColor=[0,1,0,1])
p.createMultiBody(baseVisualShapeIndex=vis,basePosition=target_place)

# ---------------- STATE MACHINE ----------------
state="approach"
constraint_id=None
release_timer=0

current_target=[0.30,0.10,0.20]
smooth_gain=0.05

# ---------------- MAIN LOOP ----------------
while p.isConnected():

    cube_pos,_=p.getBasePositionAndOrientation(cube)

    # ---------- STATE LOGIC ----------
    if state=="approach":
        current_target=[cube_pos[0],cube_pos[1],cube_pos[2]+0.10]

        if abs(current_target[0]-cube_pos[0])<0.01:
            state="pick"
            print("STATE → pick")

    elif state=="pick":
        current_target=[cube_pos[0],cube_pos[1],cube_pos[2]+0.02]

        contacts=p.getContactPoints(gripper,cube)
        if len(contacts)>0:
            constraint_id=p.createConstraint(gripper,-1,cube,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0])
            state="lift"
            print("STATE → lift")

    elif state=="lift":
        current_target=[cube_pos[0],cube_pos[1],0.25]

        if cube_pos[2]>0.15:
            state="move"
            print("STATE → move to place")

    elif state=="move":
        current_target=[target_place[0],target_place[1],0.25]

        if abs(cube_pos[0]-target_place[0])<0.02:
            state="lower"
            print("STATE → lower")

    elif state=="lower":
        current_target=[target_place[0],target_place[1],0.04]

        if cube_pos[2]<0.05:
            state="release_wait"
            release_timer=40
            print("STATE → prepare release")

    elif state=="release_wait":
        release_timer-=1
        p.resetBaseVelocity(cube,[0,0,0],[0,0,0])

        if release_timer<=0:
            p.removeConstraint(constraint_id)
            constraint_id=None
            p.resetBaseVelocity(cube,[0,0,0],[0,0,0])
            state="done"
            print("STATE → done")
            
            p.resetBaseVelocity(cube,[0,0,0],[0,0,0])

    # ---------- IK CONTROL ----------
    joint_angles=p.calculateInverseKinematics(
        robot,
        ee_index,
        current_target,
        maxNumIterations=120
    )

    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot,j)[2]==p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                robot,j,p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=120,
                positionGain=0.03,
                velocityGain=0.3
            )
 
    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
