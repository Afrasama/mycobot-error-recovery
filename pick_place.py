import pybullet as p
import pybullet_data
import time
import os

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

p.loadURDF("plane.urdf")

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
URDF_PATH=os.path.join(BASE_DIR,"urdf","mycobot_320.urdf")

robot=p.loadURDF(URDF_PATH,useFixedBase=True,flags=p.URDF_USE_INERTIA_FROM_FILE)

for j in range(p.getNumJoints(robot)):
    p.changeDynamics(robot,j,linearDamping=0.04,angularDamping=0.04)

ee_index=None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot,i)[12].decode()=="link6":
        ee_index=i

print("end effector index:",ee_index)

gripper=p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.02,0.02,0.01]),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,halfExtents=[0.02,0.02,0.01],rgbaColor=[0.2,0.2,0.2,1])
)

p.createConstraint(robot,ee_index,gripper,-1,p.JOINT_FIXED,[0,0,0],[0,0,0.04],[0,0,0])

cube=p.loadURDF("cube_small.urdf",[0.30,0.10,0.02])
p.changeDynamics(cube,-1,mass=0.2)

x_slider=p.addUserDebugParameter("X",0.1,0.4,0.30)
y_slider=p.addUserDebugParameter("Y",-0.3,0.3,0.10)
z_slider=p.addUserDebugParameter("Z",0.05,0.4,0.20)
grab_button=p.addUserDebugParameter("GRAB / PLACE",1,0,0)

constraint_id=None
last_button_state=0
release_wait=0

current_target=[0.30,0.10,0.20]
smooth_gain=0.05

while p.isConnected():

    slider_target=[
        p.readUserDebugParameter(x_slider),
        p.readUserDebugParameter(y_slider),
        p.readUserDebugParameter(z_slider)
    ]

    for i in range(3):
        current_target[i]+= (slider_target[i]-current_target[i]) * smooth_gain

    joint_angles=p.calculateInverseKinematics(robot,ee_index,current_target,maxNumIterations=120)

    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot,j)[2]==p.JOINT_REVOLUTE:
            p.setJointMotorControl2(robot,j,p.POSITION_CONTROL,
                                    targetPosition=joint_angles[j],
                                    force=120,
                                    positionGain=0.03,
                                    velocityGain=0.3)

    button_state=p.readUserDebugParameter(grab_button)

    if button_state!=last_button_state:
        last_button_state=button_state
        contacts=p.getContactPoints(gripper,cube)

        if constraint_id is None:
            if len(contacts)>0:
                print("manual grab activated")
                constraint_id=p.createConstraint(gripper,-1,cube,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0])
        else:
            cube_pos,_=p.getBasePositionAndOrientation(cube)

            if cube_pos[2] < 0.045:
                print("preparing gentle placement")
                release_wait=40   # wait frames before release

    # ----- GENTLE RELEASE TIMER -----
    if release_wait>0:
        release_wait-=1

        # freeze robot slightly while placing
        p.resetBaseVelocity(cube,[0,0,0],[0,0,0])

        if release_wait==0 and constraint_id is not None:
            print("gentle release")
            p.removeConstraint(constraint_id)
            constraint_id=None
            p.resetBaseVelocity(cube,[0,0,0],[0,0,0])

    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
