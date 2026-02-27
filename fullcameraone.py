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
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/240)
p.loadURDF("plane.urdf")

# ---------------- LOAD ROBOT ----------------
robot = p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

for j in range(p.getNumJoints(robot)):
    p.changeDynamics(robot, j, linearDamping=0.04, angularDamping=0.04)

# ---------------- FIND END EFFECTOR ----------------
ee_index = None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, i)[12].decode() == "link6":
        ee_index = i
        break

if ee_index is None:
    raise ValueError("link6 not found")

print("End effector index:", ee_index)

# ---------------- SIMPLE GRIPPER ----------------
gripper = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.01]),
    baseVisualShapeIndex=p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.01],
        rgbaColor=[0.2, 0.2, 0.2, 1])
)

p.createConstraint(robot, ee_index, gripper, -1,
                   p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.04], [0, 0, 0])

# ---------------- CUBE ----------------
cube = p.loadURDF("cube_small.urdf", [0.3, 0.1, 0.02])
p.changeDynamics(cube, -1, mass=0.2, lateralFriction=1.2)

# ---------------- GOAL ----------------
goal_position = np.array([0.3, 0.1, 0.02])

goal_visual = p.createVisualShape(
    p.GEOM_SPHERE,
    radius=0.035,
    rgbaColor=[1, 0, 0, 1]
)

p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=goal_visual,
    basePosition=goal_position.tolist()
)

constraint_id = None
retry_offset = np.array([0.0, 0.0, 0.0])
retry_count = 0

state = "manual"
timer = 0
stable_counter = 0

# ==================================================
# PERFECT REASONING CAMERA
# ==================================================
def get_reasoning_camera():

    width = 520
    height = 400

    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    cube_pos = np.array(cube_pos)

    # Always center cube
    focus_point = cube_pos.copy()
    focus_point[2] += 0.03  # slight upward bias

    # Dynamic zoom based on cube height (keeps framing tight)
    camera_distance = 0.42 + 0.2 * cube_pos[2]

    yaw = 50
    pitch = -32
    roll = 0

    viewMatrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=focus_point.tolist(),
        distance=camera_distance,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        upAxisIndex=2
    )

    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=55,
        aspect=width/height,
        nearVal=0.05,
        farVal=2
    )

    _, _, rgb, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer=p.ER_TINY_RENDERER
    )

    img = np.reshape(rgb, (height, width, 4))[:, :, :3]
    img = np.flip(img, 0).astype(np.uint8)

    return img

# ---------------- SAVE FAILURE IMAGE ----------------
def capture_failure_image(tag):
    img = get_reasoning_camera()
    path = os.path.join(SAVE_DIR, f"{tag}_{int(time.time())}.png")
    cv2.imwrite(path, img)
    print("Saved:", path)

# ---------------- SLIDERS ----------------
x_slider = p.addUserDebugParameter("X", -0.4, 0.4, 0.30)
y_slider = p.addUserDebugParameter("Y", -0.4, 0.4, 0.10)
z_slider = p.addUserDebugParameter("Z", 0.02, 0.5, 0.20)

current_target = np.array([0.30, 0.10, 0.20])

print("\nSPACE = grab / release cube\n")

# ---------------- MAIN LOOP ----------------
while p.isConnected():

    timer += 1

    cam_img = get_reasoning_camera()
    cv2.imshow("Reasoning Camera View", cam_img)
    cv2.waitKey(1)

    target = np.array([
        p.readUserDebugParameter(x_slider),
        p.readUserDebugParameter(y_slider),
        p.readUserDebugParameter(z_slider)
    ]) + retry_offset

    # Faster but still smooth
    current_target += (target - current_target) * 0.22

    joint_angles = p.calculateInverseKinematics(
        robot,
        ee_index,
        current_target.tolist(),
        maxNumIterations=80
    )

    for j in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, j)
        joint_type = joint_info[2]

        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.setJointMotorControl2(
                robot, j, p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=160,
                positionGain=0.18,
                velocityGain=0.7
            )

    keys = p.getKeyboardEvents()

    if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:

        if constraint_id is None:

            grip_pos, _ = p.getBasePositionAndOrientation(gripper)
            cube_pos, _ = p.getBasePositionAndOrientation(cube)

            dist = np.linalg.norm(np.array(grip_pos) - np.array(cube_pos))

            if dist < 0.07:
                print("Grabbed cube")
                constraint_id = p.createConstraint(
                    gripper, -1, cube, -1,
                    p.JOINT_FIXED, [0, 0, 0],
                    [0, 0, 0], [0, 0, 0]
                )
                state = "holding"

        else:
            print("Released cube")
            p.removeConstraint(constraint_id)
            constraint_id = None
            state = "checking"
            timer = 0
            stable_counter = 0

    if state == "checking":

        cube_pos, _ = p.getBasePositionAndOrientation(cube)
        cube_pos = np.array(cube_pos)

        lin_vel, _ = p.getBaseVelocity(cube)
        speed = np.linalg.norm(lin_vel)

        distance_to_goal = np.linalg.norm(cube_pos - goal_position)

        if distance_to_goal < 0.05 and speed < 0.01:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter > 100:
            print("Placement SUCCESS")
            retry_offset = np.array([0.0, 0.0, 0.0])
            state = "manual"

        elif timer > 200:
            print("Placement FAILED")
            capture_failure_image("placement_fail")

            retry_count += 1

            retry_offset += np.array([
                np.random.uniform(-0.015, 0.015),
                np.random.uniform(-0.015, 0.015),
                0.01
            ])

            print("Retry offset:", retry_offset)
            state = "manual"

    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
cv2.destroyAllWindows()