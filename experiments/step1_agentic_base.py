import pybullet as p
import pybullet_data
import time
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "data", "failures")
os.makedirs(SAVE_DIR, exist_ok=True)

URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

print("images will be saved in:", SAVE_DIR)

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
ee_index=None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot,i)[12].decode()=="link6":
        ee_index=i
        break

if ee_index is None:
    raise ValueError("link6 not found")

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
cube=p.loadURDF("cube_small.urdf",[0.3,0.1,0.02])
p.changeDynamics(cube,-1,mass=0.2,lateralFriction=1.2)

constraint_id=None

# ============================================================
# ⭐ properly positioned calibrated camera (stable version)
# ============================================================
def get_ee_camera():

    width=320
    height=240

    # end-effector position
    link_state=p.getLinkState(robot,ee_index)
    ee_pos=link_state[0]

    # cube position (camera always aims here)
    cube_pos,_=p.getBasePositionAndOrientation(cube)

    # --- calibrated camera mount near gripper ---
    cam_pos=[
        ee_pos[0]-0.05,   # slightly behind gripper
        ee_pos[1],
        ee_pos[2]+0.06    # slightly above
    ]

    # camera always looks at cube
    target_pos=[
        cube_pos[0],
        cube_pos[1],
        cube_pos[2]+0.02
    ]

    up_vec=[0,0,1]

    viewMatrix=p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up_vec
    )

    projectionMatrix=p.computeProjectionMatrixFOV(
        fov=75,
        aspect=width/height,
        nearVal=0.02,
        farVal=2
    )

    _,_,rgb,_,_=p.getCameraImage(
        width,
        height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    img=np.reshape(rgb,(height,width,4))[:,:,:3]
    img=np.flip(img,0).astype(np.uint8)

    return img

# ---------------- SAVE FAILURE IMAGE ----------------
def capture_failure_image(tag):
    img=get_ee_camera()
    path=os.path.join(SAVE_DIR,f"{tag}_{int(time.time())}.png")
    cv2.imwrite(path,img)
    print("saved:",path)

# ---------------- SLIDERS ----------------
x_slider=p.addUserDebugParameter("X",-0.4,0.4,0.30)
y_slider=p.addUserDebugParameter("Y",-0.4,0.4,0.10)
z_slider=p.addUserDebugParameter("Z",0.02,0.5,0.20)

current_target=[0.30,0.10,0.20]

retry_offset=[0,0,0]
retry_count=0

state="manual"
timer=0
stable_counter=0

print("\nSPACE = grab / release cube\n")

# ---------------- MAIN LOOP ----------------
while p.isConnected():

    timer+=1

    # ⭐ live camera window
    cam_img=get_ee_camera()
    cv2.imshow("EE Camera View",cam_img)
    cv2.waitKey(1)

    target=[
        p.readUserDebugParameter(x_slider)+retry_offset[0],
        p.readUserDebugParameter(y_slider)+retry_offset[1],
        p.readUserDebugParameter(z_slider)+retry_offset[2]
    ]

    for i in range(3):
        current_target[i]+= (target[i]-current_target[i])*0.08

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

    keys=p.getKeyboardEvents()

    if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:

        if constraint_id is None:

            grip_pos,_=p.getBasePositionAndOrientation(gripper)
            cube_pos,_=p.getBasePositionAndOrientation(cube)

            dist=np.linalg.norm(np.array(grip_pos)-np.array(cube_pos))

            if dist<0.07:
                print("grabbed cube")
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

    if state=="checking":

        cube_pos,_=p.getBasePositionAndOrientation(cube)
        lin_vel,_=p.getBaseVelocity(cube)
        speed=sum(abs(v) for v in lin_vel)

        if cube_pos[2]<0.04 and speed<0.01:
            stable_counter+=1
        else:
            stable_counter=0

        if stable_counter>120:
            print("placement success")
            retry_offset=[0,0,0]
            state="manual"

        elif timer>240:
            print("placement failed")
            capture_failure_image("placement_fail")

            retry_count+=1
            retry_offset[0]+=np.random.uniform(-0.02,0.02)
            retry_offset[1]+=np.random.uniform(-0.02,0.02)
            retry_offset[2]+=0.01

            print("retry offset:",retry_offset)
            state="manual"

    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
cv2.destroyAllWindows()