"""Nero REAL-arm bring-up: driver + MoveIt + RViz + the sideways mount frame.

Same view as sim.launch.py (RViz Fixed Frame "ground", world up along
ground +z) but on the real robot: includes agx_arm_ctrl's
start_single_agx_arm_moveit.launch.py (driver on can_port, follow:=true so
RViz tracks feedback/joint_states) plus the ground -> world static TF.

    sudo bash third_party/agx_arm_ros/scripts/can_activate.sh can0 1000000
    source /opt/ros/humble/setup.bash
    source third_party/agx_arm_ros/install/setup.bash
    ros2 launch ./real.launch.py            # [can_port:=can0]
    # then: python3 level_gui.py --real   (or move_level.py --real)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    can_port = DeclareLaunchArgument("can_port", default_value="can0")
    real = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("agx_arm_ctrl"), "launch",
            "start_single_agx_arm_moveit.launch.py"])),
        launch_arguments={
            "arm_type": "nero",
            "effector_type": "agx_gripper",
            "can_port": LaunchConfiguration("can_port"),
            # real arm keeps follow:=true (default): MoveIt/RViz track
            # feedback/joint_states, the live robot state
        }.items())
    mount_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="sideways_mount_tf",
        # x y z yaw pitch roll parent child — same as sim.launch.py
        arguments=["0", "0", "0.5", "0", "-1.5708", "0", "ground", "world"])
    return LaunchDescription([can_port, real, mount_tf])
