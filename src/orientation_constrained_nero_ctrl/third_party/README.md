# third_party

- `agx_arm_ros/` — official AgileX ROS2 driver stack, cloned from
  https://github.com/agilexrobotics/agx_arm_ros branch `ros2`,
  commit `cfd71ecd5bdff65bcd6134185013491e630814bd` (2026-08-14).
  Supports Piper and Nero on ROS2 Humble/Jazzy. Packages under `src/`:
  - `agx_arm_ctrl` — Python driver node (CAN, via pyAgxArm) + launch files
  - `agx_arm_description` — URDF/Xacro (contains `agx_arm_urdf` as a git
    submodule, same commit `f6642ce0` we vendored in `../assets/nero/`)
  - `agx_arm_moveit` — MoveIt2 config (SRDF, kinematics.yaml,
    joint_limits.yaml, `ros2_controllers_nero.yaml`)
  - `agx_arm_msgs` — custom message definitions

Kept as a plain clone (not vendored into this repo's git history — ~360 MB
with meshes). Re-fetch with:
`git clone --branch ros2 --recurse-submodules https://github.com/agilexrobotics/agx_arm_ros.git`

Other official AgileX repos relevant to IK/simulation (not cloned yet):
- `agx_arm_kinematics` — official kinematics library
- `robotic_arm_kinematics` — kinematics for AgileX arm series
- `agilex_arm_mujoco_sim`, `agx_arm_sim` — official simulation repos
