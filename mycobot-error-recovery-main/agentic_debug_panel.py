import pybullet as p
import pybullet_data
import time
import os
import random

# ---------------- CONNECT ----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

p.loadURDF("plane.urdf")

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
URDF_PATH=os.path.join(BASE_DIR,"urdf","mycobot_320.urdf")

robot=p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

# damping (anti shake)
for j in range(p.getNumJoints(robot)):
    p.changeDynamics(robot,j,linearDamping=0.04,angularDamping=0.04)

# ---------------- FIND END EFFECTOR ----------------
ee_index=None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot,i)[12].decode()=="link6":
        ee_index=i
        break

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

p.createConstraint(robot,ee_index,gripper,-1,
                   p.JOINT_FIXED,[0,0,0],[0,0,0.04],[0,0,0])

# ---------------- CUBE ----------------
cube=p.loadURDF("cube_small.urdf",basePosition=[0.30,0.10,0.02])
p.changeDynamics(cube,-1,mass=0.2)

# ---------------- DEBUG PANEL ----------------
grasp_fail_count=0
placement_fail_count=0
reflection_active=False
debug_id=None

def update_debug_panel(state):
    global debug_id
    text=(f"STATE: {state}\n"
          f"GRASP FAILS: {grasp_fail_count}\n"
          f"PLACEMENT FAILS: {placement_fail_count}\n"
          f"REFLECTION ACTIVE: {reflection_active}")

    if debug_id is not None:
        p.removeUserDebugItem(debug_id)

    debug_id=p.addUserDebugText(
        text,[0.2,0,0.35],
        textColorRGB=[0,1,0],
        textSize=1.3
    )

# ---------------- TRAJECTORY VISUALIZATION ----------------
last_ee_pos=None

def draw_trajectory():
    global last_ee_pos
    link_state=p.getLinkState(robot,ee_index)
    ee_pos=link_state[0]

    if last_ee_pos is not None:
        p.addUserDebugLine(
            last_ee_pos,
            ee_pos,
            [1,0,0],     # red path
            lineWidth=2,
            lifeTime=5
        )

    last_ee_pos=ee_pos

# ---------------- TARGETS ----------------
base_pick=[0.30,0.10,0.05]
place_pos=[0.15,-0.20,0.10]

approach_offset=[0.0,0.0]
current_target=[0.30,0.10,0.20]
smooth_gain=0.05

constraint_id=None

# ---------------- STATE MACHINE ----------------
state="approach"
timer=0

print("agent debug + trajectory active")

# ---------------- MAIN LOOP ----------------
while p.isConnected():

    timer+=1

    pick_pos=[base_pick[0]+approach_offset[0],
              base_pick[1]+approach_offset[1],
              base_pick[2]]

    # ---------- STATE LOGIC ----------
    if state=="approach":
        target=pick_pos
        if timer>200:
            state="grab"
            timer=0

    elif state=="grab":
        contacts=p.getContactPoints(gripper,cube)

        if len(contacts)>0:
            constraint_id=p.createConstraint(
                gripper,-1,cube,-1,
                p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0]
            )
            reflection_active=False
            state="lift"
            timer=0
            print("cube grabbed")

        elif timer>200:
            grasp_fail_count+=1
            reflection_active=True
            print(f"❌ grasp failure #{grasp_fail_count}")

            approach_offset[0]+=random.uniform(-0.03,0.03)
            approach_offset[1]+=random.uniform(-0.03,0.03)

            state="approach"
            timer=0

    elif state=="lift":
        target=[pick_pos[0],pick_pos[1],0.25]
        if timer>240:
            state="move_to_place"
            timer=0

    elif state=="move_to_place":
        target=place_pos
        if timer>300:
            state="lower"
            timer=0

    elif state=="lower":
        target=[place_pos[0],place_pos[1],0.05]
        if timer>240:
            state="release"
            timer=0

    elif state=="release":
        if constraint_id is not None:
            p.removeConstraint(constraint_id)
            constraint_id=None
            p.resetBaseVelocity(cube,[0,0,0],[0,0,0])
            print("cube released")

        cube_pos,_=p.getBasePositionAndOrientation(cube)
        lin_vel,_=p.getBaseVelocity(cube)
        speed=abs(lin_vel[0])+abs(lin_vel[1])+abs(lin_vel[2])

        if cube_pos[2]<0.04 and speed<0.02:
            reflection_active=False
            print("✔ placement success")
            state="done"
        else:
            placement_fail_count+=1
            reflection_active=True
            print(f"❌ placement failure #{placement_fail_count}")
            state="retry"

        timer=0

    elif state=="retry":
        target=[place_pos[0],place_pos[1],0.20]
        if timer>200:
            state="lower"
            timer=0

    elif state=="done":
        target=[place_pos[0],place_pos[1],0.25]

    # ---------- SMOOTH MOTION ----------
    for i in range(3):
        current_target[i]+= (target[i]-current_target[i])*smooth_gain

    # ---------- IK ----------
    joint_angles=p.calculateInverseKinematics(
        robot,
        ee_index,
        current_target,
        maxNumIterations=120
    )

    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot,j)[2]==p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=120,
                positionGain=0.03,
                velocityGain=0.3
            )

    # ---------- UPDATE DEBUG ----------
    update_debug_panel(state)
    draw_trajectory()

    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
