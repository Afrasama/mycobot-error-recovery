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

# ---------- ADD JOINT DAMPING (ANTI-SHAKE) ----------
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

p.createConstraint(
    parentBodyUniqueId=robot,
    parentLinkIndex=ee_index,
    childBodyUniqueId=gripper,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0,0,0],
    parentFramePosition=[0,0,0.04],
    childFramePosition=[0,0,0]
)

# ---------------- ADD CUBE ----------------
cube=p.loadURDF(
    "cube_small.urdf",
    basePosition=[0.30,0.10,0.02]
)

# make cube slightly heavier (more stable grasp)
p.changeDynamics(cube,-1,mass=0.2)

# ---------------- SLIDERS ----------------
x_slider=p.addUserDebugParameter("X",0.1,0.4,0.30)
y_slider=p.addUserDebugParameter("Y",-0.3,0.3,0.10)
z_slider=p.addUserDebugParameter("Z",0.05,0.4,0.20)

grab_button=p.addUserDebugParameter("GRAB / RELEASE",1,0,0)

constraint_id=None
last_button_state=0

# ---------------- SMOOTH TARGET ----------------
current_target=[0.30,0.10,0.20]
smooth_gain=0.05

# industrial anti-shiver flag
holding_object=False

# ---------------- MAIN LOOP ----------------
while p.isConnected():

    slider_target=[
        p.readUserDebugParameter(x_slider),
        p.readUserDebugParameter(y_slider),
        p.readUserDebugParameter(z_slider)
    ]

    # ----- SMOOTH INTERPOLATION -----
    for i in range(3):
        current_target[i]+= (slider_target[i]-current_target[i]) * smooth_gain

    # ----- IK ONLY WHEN NEEDED -----
    joint_angles=p.calculateInverseKinematics(
        robot,
        ee_index,
        current_target,
        maxNumIterations=120,
        residualThreshold=1e-4
    )

    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot,j)[2]==p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=120,            # reduced stiffness
                positionGain=0.03,
                velocityGain=0.3
            )

    # ----- MANUAL GRAB CONTROL -----
    button_state=p.readUserDebugParameter(grab_button)

    if button_state!=last_button_state:
        last_button_state=button_state

        contacts=p.getContactPoints(gripper,cube)

        if constraint_id is None:
            if len(contacts)>0:
                print("manual grab activated")

                constraint_id=p.createConstraint(
                    parentBodyUniqueId=gripper,
                    parentLinkIndex=-1,
                    childBodyUniqueId=cube,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0,0,0],
                    parentFramePosition=[0,0,0],
                    childFramePosition=[0,0,0]
                )
                holding_object=True
            else:
                print("bring gripper closer before grabbing")

        else:
            p.removeConstraint(constraint_id)
            constraint_id=None
            holding_object=False
            print("cube released")

    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
