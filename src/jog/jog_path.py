"""Straight-line Cartesian jog planning for the Nero TCP (no ROS in here).

A jog is one small, fully-specified 6-DoF move: the TCP (link7 + the fixed
tcp_offset, default 0 = the flange) goes from where it is to a nearby pose
along a straight line, with the orientation turning at a constant rate about
one fixed axis. Every step is solved with the same damped-least-squares IK
as the level planner (level_path.solve_pose, full 6-DoF here), warm-started
from the previous waypoint so the joint solution stays on one branch.

Frames — the Nero is mounted sideways, so world up = base_link +X. The
"world" jog frame is base_link's axes (base_link == world in the URDF; only
the labels differ: +X is up). The "tool" frame is the TCP's own axes (tool
axis = link7 +x).

    R0, p0 = pose_now(q, R_t, t_t)                         # FK
    R1, p1 = jog_target(R0, p0, dpos=(0, 0, 0.01))         # +10 mm world z
    R1, p1 = jog_target(R0, p0, axis="x", dang=deg2rad(5)) # 5 deg about up
    plan = plan_line(q, R1, p1, R_t=R_t, t_t=t_t)          # waypoints
    t, q, qd, qdd = retime(plan["q"], plan["s"], v=0.05, w=15, ...)

Timing: the path parameter s is a combined arc length — position travel
plus the orientation angle times the lever v/w, so one speed cap `v` [m/s]
on s bounds the linear speed by v AND the angular speed by w [deg/s].
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEVEL_DIR = os.path.join(os.path.dirname(_HERE),
                          "orientation_constrained_nero_ctrl")
if _LEVEL_DIR not in sys.path:
    sys.path.insert(0, _LEVEL_DIR)
import level_path as lp  # noqa: E402
import nero_kinematics as nk  # noqa: E402

UP = lp.UP                                   # world up in base_link (+X)
AXES = {"x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0])}
FRAMES = ("world", "tool")

# Korean labels for the world axes (sideways mount: +X points up).
AXIS_LABEL_KO = {"x": "X (위/아래)", "y": "Y", "z": "Z"}

DQ_MAX = 0.35        # rad, largest joint change between two waypoints that
                     # still counts as the same IK branch (else: fail)
POS_TOL = 1e-3       # m, waypoint IK position tolerance
ANG_TOL = 2e-3       # rad, waypoint IK orientation tolerance


def pose_now(q, R_t=np.eye(3), t_t=np.zeros(3)):
    """TCP pose of one joint config: (R (3,3), p (3,))."""
    R, p, _, _ = lp.fk_tcp(np.asarray(q, float)[None], R_t, t_t)
    return R[0], p[0]


def rpy_of(R):
    """Rotation matrix -> (roll, pitch, yaw) [rad], URDF convention
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (inverse of nero_kinematics.rpy_matrix,
    same convention show_pose.py prints)."""
    R = np.asarray(R, float)
    pitch = float(np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])))
    if abs(np.cos(pitch)) < 1e-9:            # gimbal lock: fold yaw into roll
        return (float(np.arctan2(-R[1, 2], R[1, 1])), pitch, 0.0)
    return (float(np.arctan2(R[2, 1], R[2, 2])), pitch,
            float(np.arctan2(R[1, 0], R[0, 0])))


def jog_target(R0, p0, dpos=(0.0, 0.0, 0.0), axis=None, dang=0.0,
               frame="world"):
    """The pose one jog step away from (R0, p0).

    dpos: translation [m] along the frame's axes.
    axis: "x" | "y" | "z" | unit vector — rotation axis in the frame;
    dang: rotation angle [rad] about it. The rotation is about the TCP
    point itself (the position does not move because of it).
    frame: "world" (base_link axes, +X up) or "tool" (the TCP's own axes).
    """
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}")
    R0 = np.asarray(R0, float)
    p0 = np.asarray(p0, float)
    d = np.asarray(dpos, float)
    if axis is None or abs(dang) < 1e-12:
        R1 = R0.copy()
    else:
        ax = AXES[axis] if isinstance(axis, str) else np.asarray(axis, float)
        ax = ax / np.linalg.norm(ax)
        if frame == "world":
            R1 = lp.rot_axis(ax, float(dang)) @ R0
        else:
            R1 = R0 @ lp.rot_axis(ax, float(dang))
    p1 = p0 + d if frame == "world" else p0 + R0 @ d
    return R1, p1


def pose_from_rpy(xyz, rpy):
    """(x, y, z) [m] + (roll, pitch, yaw) [rad] in base_link -> (R, p)."""
    return nk.rpy_matrix(*rpy), np.asarray(xyz, float)


def plan_line(q0, R1, p1, *, step=0.005, ang_step=np.deg2rad(2.0),
              R_t=np.eye(3), t_t=np.zeros(3), dq_max=DQ_MAX):
    """Straight-line TCP path from the pose of q0 to (R1, p1), IK'd per
    waypoint. Returns a dict:
      ok      True when every waypoint converged on one joint branch
      q       (N, 7) joint waypoints, q[0] == q0
      s       (N,) Cartesian arc length [m] (position only)
      ang     (N,) orientation angle travelled [rad]
      dist    total translation [m];  angle  total rotation [rad]
      msg     why it failed (empty when ok)
    Waypoint spacing: at most `step` metres and `ang_step` radians apart,
    never fewer than 2 segments (so the retimer gets 3+ points).
    """
    q0 = np.asarray(q0, float)
    R0, p0 = pose_now(q0, R_t, t_t)
    R1 = np.asarray(R1, float)
    p1 = np.asarray(p1, float)
    axis, angle = lp._log_rot(R1 @ R0.T)
    dist = float(np.linalg.norm(p1 - p0))
    n = int(max(2, np.ceil(dist / step), np.ceil(angle / ang_step)))
    lp.CANCEL.clear()
    qs = [q0]
    s = [0.0]
    ang = [0.0]
    q_prev = q0
    for k in range(1, n + 1):
        f = k / n
        p = p0 + f * (p1 - p0)
        R = lp.rot_axis(axis, f * angle) @ R0
        q, pos_err, _, _ = lp.solve_pose(q_prev, p, R, w_yaw=1.0,
                                         iters=80, pos_tol=POS_TOL / 2,
                                         ang_tol=ANG_TOL / 2, R_t=R_t, t_t=t_t)
        q = q[0]
        R_sol, _ = pose_now(q, R_t, t_t)
        _, ang_err = lp._log_rot(R @ R_sol.T)
        if pos_err[0] > POS_TOL or ang_err > ANG_TOL:
            return dict(ok=False, q=np.array(qs), s=np.array(s),
                        ang=np.array(ang), dist=dist, angle=angle,
                        msg=(f"IK 실패: 경로 {f * 100:.0f}% 지점 (위치 오차 "
                             f"{pos_err[0] * 1000:.1f} mm, 자세 오차 "
                             f"{np.rad2deg(ang_err):.2f}°) — 도달 불가 또는 "
                             "관절 한계"))
        jump = float(np.abs(q - q_prev).max())
        if jump > dq_max:
            return dict(ok=False, q=np.array(qs), s=np.array(s),
                        ang=np.array(ang), dist=dist, angle=angle,
                        msg=(f"관절 급변 {np.rad2deg(jump):.0f}° (경로 "
                             f"{f * 100:.0f}% 지점) — 특이점 근처, 다른 방향/"
                             "작은 스텝으로 시도"))
        qs.append(q)
        s.append(f * dist)
        ang.append(f * angle)
        q_prev = q
    return dict(ok=True, q=np.array(qs), s=np.array(s), ang=np.array(ang),
                dist=dist, angle=angle, msg="")


def retime(q, s, ang=None, *, v=0.05, w=15.0, a=0.2, qdot=1.0, qddot=3.0,
           dt=0.05):
    """Time the waypoints: TOPP-RA under (v, a) on the combined path length
    and (qdot, qddot) per joint; trapezoidal fallback when toppra is
    unavailable. w [deg/s] caps the angular speed through the lever v/w.
    Returns (t, q, qd, qdd); qd/qdd are None on the fallback path."""
    q = np.asarray(q, float)
    s = np.asarray(s, float)
    if ang is not None:
        s = s + np.asarray(ang, float) * (v / np.deg2rad(w))
    out = lp.retime_toppra(q, s, v_max=v, a_max=a, qdot_max=qdot,
                           qddot_max=qddot, dt=dt)
    if out is not None:
        return out
    t = lp.time_parameterize(s, q, v_max=v, a_max=a, qdot_max=qdot)
    return t, q, None, None
