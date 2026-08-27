"""Show the plane IK-scan result in RViz.

Publishes the scanned work plane as tiles on /reachable_workspace (same topic
the workspace_view.rviz config already displays): reachable cells blue, and
animates the arm through the stored per-cell IK solutions in a serpentine
sweep, so the flange visibly traces the plane with its orientation locked.

    source /opt/ros/humble/setup.bash
    python3 view_plane_ik_rviz.py [--file out/plane_ik_-0.31.npz] [--static]
"""
import argparse
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 0.02
BLUE = (0x2A / 255, 0x78 / 255, 0xD6 / 255)
ORANGE = (0xEB / 255, 0x68 / 255, 0x34 / 255)


class PlaneIKViz(Node):
    def __init__(self, path: str, animate: bool, rate: float = 25.0):
        super().__init__("nero_plane_ik_viz")
        d = np.load(path)
        targets, solved, q, x0 = d["targets"], d["solved"], d["q"], float(d["x0"])
        self.get_logger().info(f"{path}: {solved.sum()}/{len(solved)} cells reachable")

        qos = QoSProfile(depth=4, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        pub = self.create_publisher(Marker, "reachable_workspace", qos)

        wipe = Marker()
        wipe.header.frame_id = "base_link"
        wipe.action = Marker.DELETEALL
        pub.publish(wipe)

        tiles = Marker()
        tiles.header.frame_id = "base_link"
        tiles.ns, tiles.id = "plane_ik", 0
        tiles.type, tiles.action = Marker.CUBE_LIST, Marker.ADD
        tiles.pose.orientation.w = 1.0
        tiles.scale.x = 0.004
        tiles.scale.y = tiles.scale.z = CELL * 0.9
        tiles.frame_locked = True
        for t in targets[solved]:
            tiles.points.append(Point(x=float(t[0]), y=float(t[1]), z=float(t[2])))
            tiles.colors.append(ColorRGBA(r=BLUE[0], g=BLUE[1], b=BLUE[2], a=0.75))
        pub.publish(tiles)
        self.marker_timer = self.create_timer(2.0, lambda: pub.publish(tiles))

        if not animate:
            return
        # serpentine sweep over the grid (y-major rows, alternating z direction)
        ny = len(np.unique(np.round(targets[:, 1] / CELL)))
        nz = len(targets) // ny
        order = []
        for i in range(ny):
            js = range(nz) if i % 2 == 0 else range(nz - 1, -1, -1)
            order += [i * nz + j for j in js if solved[i * nz + j]]
        self.path_q = q[order]
        self.path_p = targets[order]
        self.get_logger().info(f"sweeping {len(order)} IK solutions")

        js_pub = self.create_publisher(JointState, "control/joint_states", 10)
        dot_pub = pub
        self._i = 0

        def tick():
            k = self._i % len(self.path_q)
            self._i += 1
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = [f"joint{n}" for n in range(1, 8)]
            js.position = [float(v) for v in self.path_q[k]]
            js_pub.publish(js)
            dot = Marker()  # highlight the cell being visited
            dot.header.frame_id = "base_link"
            dot.ns, dot.id = "plane_ik_cursor", 1
            dot.type, dot.action = Marker.SPHERE, Marker.ADD
            t = self.path_p[k]
            dot.pose.position.x, dot.pose.position.y, dot.pose.position.z = map(float, t)
            dot.pose.orientation.w = 1.0
            dot.scale.x = dot.scale.y = dot.scale.z = 0.035
            dot.color = ColorRGBA(r=ORANGE[0], g=ORANGE[1], b=ORANGE[2], a=1.0)
            dot.frame_locked = True
            dot_pub.publish(dot)
        self.timer = self.create_timer(1.0 / rate, tick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(HERE, "out", "plane_ik_-0.31.npz"))
    ap.add_argument("--static", action="store_true", help="tiles only, no sweep")
    ap.add_argument("--rate", type=float, default=25.0,
                    help="sweep speed in cells per second")
    args = ap.parse_args()
    rclpy.init()
    node = PlaneIKViz(args.file, animate=not args.static, rate=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
