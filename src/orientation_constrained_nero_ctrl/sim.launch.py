"""Nero sim bring-up: MoveIt mock demo (nero + AGX gripper) + the sideways
mount visualization frame.

    source /opt/ros/humble/setup.bash
    source third_party/agx_arm_ros/install/setup.bash
    ros2 launch ./sim.launch.py

Publishes ground -> world (pitch -90 deg) so base_link +X — the real-world up
of the sideways mount — points along ground +z. Set the RViz Fixed Frame to
"ground" (and Ctrl+S once to persist it in the rviz config).
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("agx_arm_moveit"), "launch", "demo.launch.py"])),
        launch_arguments={
            "arm_type": "nero",
            "effector_type": "agx_gripper",
            # follow:=false (default) is required in sim: move_group reads the
            # mock state from control/joint_states; with follow:=true it waits
            # on feedback/joint_states and every execution aborts with -4
            "follow": "false",
        }.items())
    mount_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="sideways_mount_tf",
        # x y z yaw pitch roll parent child
        arguments=["0", "0", "0.5", "0", "-1.5708", "0", "ground", "world"])
    return LaunchDescription([demo, mount_tf])
