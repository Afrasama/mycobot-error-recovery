import pybullet as p
import pybullet_data
import time
import os

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

# reduce shaking
for j in range(p.getNumJoints(robot)):
    p.changeDynamics(robot,j,linearDamping=0.04,angularDamping=0.04)

# ---------------- FIND END EFFECTOR ----------------
ee_index=None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot,i)[12].decode()=="link6":
        ee_index=i
        break

print("end effector index:",ee_index)
print("press SPACE to grab/release")

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
cube=p.loadURDF("cube_small.urdf",[0.3,0.1,0.02])
p.changeDynamics(cube,-1,mass=0.2,lateralFriction=1.2)

constraint_id=None

# ---------------- SLIDERS ----------------
x_slider=p.addUserDebugParameter("X",-0.4,0.4,0.30)
y_slider=p.addUserDebugParameter("Y",-0.4,0.4,0.10)
z_slider=p.addUserDebugParameter("Z",0.02,0.5,0.20)

current_target=[0.30,0.10,0.20]

# ---------------- STATE MACHINE ----------------
state="manual"
timer=0
stable_counter=0

place_pos=[0.15,-0.20,0.05]

mode_text=None

def update_mode():
    global mode_text
    if mode_text is not None:
        p.removeUserDebugItem(mode_text)
    mode_text=p.addUserDebugText(
        f"MODE: {state.upper()}",
        [0.2,0,0.35],
        textColorRGB=[0,1,0],
        textSize=1.5
    )

# ---------------- MAIN LOOP ----------------
while p.isConnected():

    timer+=1

    # -------- READ SLIDERS --------
    target=[
        p.readUserDebugParameter(x_slider),
        p.readUserDebugParameter(y_slider),
        p.readUserDebugParameter(z_slider)
    ]

    # smooth motion
    for i in range(3):
        current_target[i]+= (target[i]-current_target[i])*0.08

    # -------- IK --------
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

    # -------- SPACEBAR GRAB / RELEASE --------
    keys=p.getKeyboardEvents()
    if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:

        if constraint_id is None:
            grip_pos,_=p.getBasePositionAndOrientation(gripper)
            cube_pos,_=p.getBasePositionAndOrientation(cube)

            dx=grip_pos[0]-cube_pos[0]
            dy=grip_pos[1]-cube_pos[1]
            dz=grip_pos[2]-cube_pos[2]

            dist=(dx*dx+dy*dy+dz*dz)**0.5

            if dist<0.07:
                print("✔ grabbed cube")
                constraint_id=p.createConstraint(
                    gripper,-1,cube,-1,
                    p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0]
                )
                state="holding"

        else:
            print("released cube")
            p.removeConstraint(constraint_id)
            constraint_id=None
            state="checking"
            timer=0
            stable_counter=0

    # -------- PLACEMENT CHECK --------
    if state=="checking":

        cube_pos,_=p.getBasePositionAndOrientation(cube)
        lin_vel,_=p.getBaseVelocity(cube)
        speed=abs(lin_vel[0])+abs(lin_vel[1])+abs(lin_vel[2])

        if cube_pos[2]<0.04 and speed<0.01:
            stable_counter+=1
        else:
            stable_counter=0

        if stable_counter>120:
            print("✔ placement success")
            state="manual"

        elif timer>240:
            print("❌ placement failed → retry")
            state="retry"
            timer=0

    # -------- RETRY --------
    if state=="retry":

        retry_target=[place_pos[0],place_pos[1],0.25]

        for i in range(3):
            current_target[i]+= (retry_target[i]-current_target[i])*0.05

        if timer>200:
            state="manual"

    update_mode()

    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
