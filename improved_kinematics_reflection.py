import pybullet as p
import pybullet_data
import time
import os
import numpy as np

from reflection_module import reflect_and_update


# ---------------- CONNECT ----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)
p.setPhysicsEngineParameter(numSolverIterations=120)
p.setPhysicsEngineParameter(fixedTimeStep=1/240)


# ---------------- PLANE ----------------
plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=1.5)


# ---------------- LOAD ROBOT ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

robot = p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)


# ---------------- FIND END EFFECTOR ----------------
ee_index = None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, i)[12].decode() == "link6":
        ee_index = i
        break

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
p.changeDynamics(
    cube,
    -1,
    lateralFriction=1.5,
    linearDamping=0.4,
    angularDamping=0.4,
    restitution=0.0
)

goal_position = np.array([0.45, -0.15, 0.02])


# ---------------- POLICY ----------------
policy = {
    "approach_height": 0.12,
    "grasp_height": 0.03,
    "lift_height": 0.20,
    "release_delay": 60
}

max_retries = 3
retry_count = 0

inject_failure = True
perception_noise_scale = 0.025


# ---------------- SMOOTH MOTION ----------------
def smooth_move(target_pos, steps=120):

    joint_indices = []
    current_positions = []

    for j in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, j)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_indices.append(j)
            current_positions.append(p.getJointState(robot, j)[0])

    target_positions = p.calculateInverseKinematics(
        robot,
        ee_index,
        target_pos.tolist(),
        maxNumIterations=200
    )

    for step in range(steps):
        alpha = step / steps

        for idx, j in enumerate(joint_indices):
            interpolated = (
                (1 - alpha) * current_positions[idx] +
                alpha * target_positions[j]
            )

            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                interpolated,
                force=400,
                positionGain=0.5,
                velocityGain=1.0
            )

        p.stepSimulation()
        time.sleep(1/240)


# ---------------- STATE MACHINE ----------------
state = "plan"
timer = 0
stable_counter = 0
constraint_id = None

while p.isConnected():

    timer += 1
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    cube_pos = np.array(cube_pos)

    # -------- PLAN --------
    if state == "plan":

        print("\nPlanning attempt", retry_count + 1)

        perceived_cube_pos = cube_pos.copy()

        if inject_failure:
            print("Injecting perception error")
            perceived_cube_pos[0] += np.random.uniform(-perception_noise_scale, perception_noise_scale)
            perceived_cube_pos[1] += np.random.uniform(-perception_noise_scale, perception_noise_scale)

        approach_target = perceived_cube_pos.copy()
        approach_target[2] += policy["approach_height"]

        grasp_target = perceived_cube_pos.copy()
        grasp_target[2] += policy["grasp_height"]

        state = "approach"
        timer = 0

    # -------- APPROACH --------
    elif state == "approach":
        smooth_move(approach_target)
        state = "descend"
        timer = 0

    # -------- DESCEND --------
    elif state == "descend":
        smooth_move(grasp_target)
        state = "grasp"
        timer = 0

    # -------- GRASP --------
    elif state == "grasp":

        grip_pos = p.getLinkState(robot, ee_index)[0]
        dist = np.linalg.norm(np.array(grip_pos) - cube_pos)

        if dist < 0.06:
            constraint_id = p.createConstraint(
                gripper, -1, cube, -1,
                p.JOINT_FIXED,
                [0,0,0],[0,0,0],[0,0,0]
            )
            print("Object grasped")
            state = "lift"
            timer = 0
        else:
            print("Grasp failed")
            state = "analyze"

    # -------- LIFT --------
    elif state == "lift":
        lift_target = cube_pos.copy()
        lift_target[2] += policy["lift_height"]
        smooth_move(lift_target)
        state = "place_above"
        timer = 0

    # -------- MOVE ABOVE GOAL --------
    elif state == "place_above":
        above_goal = goal_position.copy()
        above_goal[2] += policy["approach_height"]
        smooth_move(above_goal)
        state = "lower_to_place"
        timer = 0

    # -------- LOWER --------
    elif state == "lower_to_place":
        place_target = goal_position.copy()
        place_target[2] += 0.025
        smooth_move(place_target)
        state = "release"
        timer = 0

    # -------- RELEASE --------
    elif state == "release":
        if timer > policy["release_delay"]:
            p.removeConstraint(constraint_id)
            constraint_id = None
            state = "observe"
            timer = 0
            stable_counter = 0

    # -------- OBSERVE --------
    elif state == "observe":

        lin_vel, _ = p.getBaseVelocity(cube)
        speed = np.linalg.norm(lin_vel)
        distance_to_goal = np.linalg.norm(cube_pos - goal_position)

        print("Distance:", round(distance_to_goal,3),
              "Speed:", round(speed,3))

        if distance_to_goal < 0.08 and speed < 0.1:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter > 30:
            print("SUCCESS → task completed")
            state = "done"

        elif timer > 120:
            state = "analyze"

    # -------- ANALYZE --------
    elif state == "analyze":

        if retry_count >= max_retries:
            print("Max retries reached")
            state = "done"
            continue

        print("Reflection triggered")

        policy = reflect_and_update(policy, retry_count)

        inject_failure = False
        retry_count += 1
        state = "plan"
        timer = 0

    elif state == "done":
        pass

    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()