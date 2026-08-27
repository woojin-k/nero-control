"""Publish the reachable-workspace cloud as RViz markers over the Nero model.

Run reachable_workspace.py first (creates out/workspace_*.npz), then:

    source /opt/ros/humble/setup.bash
    python3 view_workspace_rviz.py                # interpretation B, full volume
    python3 view_workspace_rviz.py --interp A
    python3 view_workspace_rviz.py --plane -0.31  # single 2 cm slab only
    python3 view_workspace_rviz.py --pose         # also pose the arm at a valid config

In RViz: Add -> Marker, topic /reachable_workspace (frame base_link).
With --pose this also publishes control/joint_states, so launch the display with
control:=false or expect the sliders to fight it.
"""
import argparse
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

HERE = os.path.dirname(os.path.abspath(__file__))
VOXEL = 0.02

# light->dark blue ramp (dataviz sequential), indexed by depth |x|
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


class WorkspaceViz(Node):
    def __init__(self, interp: str, plane: float | None, pose: bool):
        super().__init__("nero_workspace_viz")
        import nero_kinematics as nk
        G = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)
        self.R7_target = G if interp == "B" else G @ nk.R_FLANGE.T
        name = "workspace_B_link7.npz" if interp == "B" else "workspace_A_flange.npz"
        d = np.load(os.path.join(HERE, "out", name))
        p = d["p"]
        if plane is not None:
            m = np.abs(p[:, 0] - plane) < VOXEL / 2
            p, self.q_all = p[m], d["q"][m]
        else:
            self.q_all = d["q"]
        vox = np.unique(np.floor(p / VOXEL).astype(np.int32), axis=0)
        self.get_logger().info(f"{interp}: {len(p)} pts -> {len(vox)} voxels")

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        pub = self.create_publisher(Marker, "reachable_workspace", qos)
        msg = Marker()
        msg.header.frame_id = "base_link"
        msg.ns, msg.id, msg.type, msg.action = "reach", 0, Marker.CUBE_LIST, Marker.ADD
        msg.pose.orientation.w = 1.0
        msg.scale.x = msg.scale.y = msg.scale.z = VOXEL * 0.9
        msg.frame_locked = True
        centers = (vox + 0.5) * VOXEL
        depth = np.clip((-centers[:, 0]) / 0.65, 0.0, 0.999)
        for (x, y, z), t in zip(centers, depth):
            msg.points.append(_pt(x, y, z))
            msg.colors.append(_rgba(*hex_rgb(RAMP[int(t * len(RAMP))]), 0.55))
        pub.publish(msg)
        # re-publish periodically so a late-started RViz still gets it
        self.marker_timer = self.create_timer(2.0, lambda: pub.publish(msg))

        if pose == "tour" and len(self.q_all):
            self._start_tour()
        elif pose and len(self.q_all):
            # config whose flange is nearest the centroid of the shown cloud
            import nero_kinematics as nk
            _, pf = nk.fk_flange(self.q_all)
            q = self.q_all[np.argmin(np.linalg.norm(pf - pf.mean(0), axis=1))]
            js_pub = self.create_publisher(JointState, "control/joint_states", qos)
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = [f"joint{i}" for i in range(1, 8)]
            js.position = [float(v) for v in q]
            self.timer = self.create_timer(0.1, lambda: (js_pub.publish(js)))
        self.get_logger().info("publishing /reachable_workspace (latched); Ctrl+C to quit")


    def _start_tour(self, n_anchors: int = 10, steps: int = 30):
        """Smoothly move through configs spread over the cloud, re-projecting
        every interpolated frame onto the orientation constraint so the EE
        orientation stays exactly at the target throughout."""
        import nero_kinematics as nk
        _, pf = nk.fk_flange(self.q_all)
        # farthest-point sampling over flange positions -> anchors spread wide
        idx = [int(np.argmin(pf[:, 0]))]
        d = np.linalg.norm(pf - pf[idx[0]], axis=1)
        for _ in range(n_anchors - 1):
            idx.append(int(np.argmax(d)))
            d = np.minimum(d, np.linalg.norm(pf - pf[idx[-1]], axis=1))
        anchors = self.q_all[idx].astype(float)
        anchors = np.vstack([anchors, anchors[:1]])
        # interpolate in joint space, then project back onto the manifold
        segs = []
        for a, b in zip(anchors[:-1], anchors[1:]):
            t = np.linspace(0, 1, steps, endpoint=False)[:, None]
            segs.append(a + t * (b - a))
        path = np.vstack(segs)
        R7t = self.R7_target
        path, ok = nk.project_orientation(path, R7t, iters=120)
        R7, _, _, _ = nk.fk_chain(path)
        err = np.linalg.norm(nk.orientation_error(R7, R7t), axis=1)
        path = path[ok & (err < 2e-4)]
        self.get_logger().info(f"tour: {len(path)} frames, "
                               f"max orient err {err.max():.1e} rad")
        js_pub = self.create_publisher(JointState, "control/joint_states", 10)
        self._ti = 0

        def tick():
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = [f"joint{i}" for i in range(1, 8)]
            js.position = [float(v) for v in path[self._ti % len(path)]]
            self._ti += 1
            js_pub.publish(js)
        self.timer = self.create_timer(0.05, tick)


def _pt(x, y, z):
    from geometry_msgs.msg import Point
    return Point(x=float(x), y=float(y), z=float(z))


def _rgba(r, g, b, a):
    from std_msgs.msg import ColorRGBA
    return ColorRGBA(r=r, g=g, b=b, a=a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interp", choices=["A", "B"], default="B")
    ap.add_argument("--plane", type=float, default=None,
                    help="show only the 2 cm slab at this x0 (e.g. -0.31)")
    ap.add_argument("--pose", action="store_true",
                    help="also publish a valid joint pose on control/joint_states")
    ap.add_argument("--tour", action="store_true",
                    help="animate the arm through the cloud with the EE "
                         "orientation locked to the target")
    args = ap.parse_args()
    rclpy.init()
    node = WorkspaceViz(args.interp, args.plane,
                        "tour" if args.tour else args.pose)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
