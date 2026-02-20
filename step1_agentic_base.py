import pybullet as p
import pybullet_data
import time
import os
import cv2
import numpy as np

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "data", "failures")
os.makedirs(SAVE_DIR, exist_ok=True)

URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

print("Images will be saved in:", SAVE_DIR)

# ---------------- CONNECT ----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

p.loadURDF("plane.urdf")

# ---------------- LOAD ROBOT ----------------
robot = p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

# reduce shaking
for j in range(p.getNumJoints(robot)):
    p.changeDynamics(robot,j,linearDamping=0.04,angularDamping=0.04)

# ---------------- FIND END EFFECTOR ----------------
ee_index = None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot,i)[12].decode()=="link6":
        ee_index = i
        break

if ee_index is None:
    raise ValueError("End effector 'link6' not found in URDF")

print("end effector index:", ee_index)

# ---------------- SIMPLE GRIPPER ----------------
gripper = p.createMultiBody(
    baseMass=0.0,
    baseCollisionShapeIndex=p.createCollisionShape(
        p.GEOM_BOX,halfExtents=[0.02,0.02,0.01]),
    baseVisualShapeIndex=p.createVisualShape(
        p.GEOM_BOX,halfExtents=[0.02,0.02,0.01],
        rgbaColor=[0.2,0.2,0.2,1])
)

p.createConstraint(robot,ee_index,gripper,-1,
                   p.JOINT_FIXED,[0,0,0],[0,0,0.04],[0,0,0])

# ---------------- CUBE ----------------
cube = p.loadURDF("cube_small.urdf",[0.3,0.1,0.02])
p.changeDynamics(cube,-1,mass=0.2,lateralFriction=1.2)

constraint_id = None

# ---------------- SLIDERS ----------------
x_slider = p.addUserDebugParameter("X",-0.4,0.4,0.30)
y_slider = p.addUserDebugParameter("Y",-0.4,0.4,0.10)
z_slider = p.addUserDebugParameter("Z",0.02,0.5,0.20)

current_target = [0.30,0.10,0.20]

# ---------------- STATE ----------------
state="manual"
timer=0
stable_counter=0

failure_data={
    "type":None,
    "retry_count":0,
    "ee_position":None
}

# ---------------- CAMERA CAPTURE ----------------
def capture_failure_image():

    width=640
    height=480

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[0.6,0.6,0.6],
        cameraTargetPosition=[0.3,0.1,0],
        cameraUpVector=[0,0,1]
    )

    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width/height,
        nearVal=0.01,
        farVal=2
    )

    _,_,rgb,_,_ = p.getCameraImage(
        width,
        height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb_array=np.reshape(rgb,(height,width,4))
    rgb_array=rgb_array[:,:,:3].astype(np.uint8)
    rgb_array=np.flip(rgb_array,axis=0)

    file_name=f"failure_{failure_data['retry_count']}.png"
    save_path=os.path.join(SAVE_DIR,file_name)

    cv2.imwrite(save_path,rgb_array)

    print(f"📸 failure image saved → {save_path}")

print("\nSPACE = grab / release cube\n")

# ---------------- MAIN LOOP ----------------
while p.isConnected():

    timer+=1

    # ----- read sliders -----
    target=[
        p.readUserDebugParameter(x_slider),
        p.readUserDebugParameter(y_slider),
        p.readUserDebugParameter(z_slider)
    ]

    # smooth movement
    for i in range(3):
        current_target[i]+= (target[i]-current_target[i])*0.08

    # ----- IK -----
    joint_angles=p.calculateInverseKinematics(
        robot,
        ee_index,
        current_target,
        maxNumIterations=120
    )

    # ⭐ FIXED IK MOTOR INDEXING
    idx=0
    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot,j)[2]==p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[idx],
                force=120,
                positionGain=0.03,
                velocityGain=0.3
            )
            idx+=1

    # ----- SPACEBAR GRAB/RELEASE -----
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

    # ----- FAILURE CHECK -----
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
            print("❌ placement failed")

            ee_pos=p.getLinkState(robot,ee_index)[0]
            failure_data["type"]="placement_fail"
            failure_data["retry_count"]+=1
            failure_data["ee_position"]=ee_pos

            print("failure data:",failure_data)

            capture_failure_image()

            state="manual"

    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()