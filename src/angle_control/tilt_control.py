"""Carry the container at a fixed angle to gravity on the Nero (sim or real).

The tilt angle is given ONCE at launch (--tilt, degrees from level) and the
container is then carried at that angle: every move keeps it, and the
direction it leans toward is one of the container's own sides (--lean,
default front), so it turns with the commanded yaw like the container
itself. Moves use the move_level planner unchanged (goal x y z [yaw]).

    source /opt/ros/humble/setup.bash
    # sim:  ros2 launch ../orientation_constrained_nero_ctrl/sim.launch.py
    # real: ros2 launch ../orientation_constrained_nero_ctrl/real.launch.py
    python3 tilt_control.py --tilt 20 [--lean front] [--real]   # prompt
    python3 tilt_control.py --tilt 20 --apply --real --execute  # tilt in place
    python3 tilt_control.py --tilt 20 --goal x y z [--goal-yaw D] --preview

yaw: level_path's absolute yaw [deg] about world up, 0 = the arm stretched
straight ahead (+Z) with the gripper down. The tilt is an in-place rotation
about tcp_link (--tcp-xyz/--tcp-rpy, same as the MoveIt tcp_offset).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEVEL_DIR = os.path.join(os.path.dirname(_HERE),
                          "orientation_constrained_nero_ctrl")
for _d in (_HERE, _LEVEL_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import level_path as lp  # noqa: E402
import move_level as ml  # noqa: E402
import tilt_path as tp  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402

OUT = os.path.join(_HERE, "out")
TILT_TOL = np.deg2rad(0.5)    # the carried tilt must match --tilt within this


def describe(qj, R_t, t_t):
    R, p, _, _ = lp.fk_tcp(qj[None], R_t, t_t)
    theta, d = tp.tilt_state(R[0])
    lean = "-" if d is None else f"azimuth {tp.azimuth_of(d):+.1f} deg"
    return (f"tcp pos: {np.round(p[0], 4).tolist()}  "
            f"yaw {np.rad2deg(lp.abs_yaw(R[0])):+.1f} deg  "
            f"tilt {np.rad2deg(theta):.2f} deg toward {lean}")


def tilt_mismatch(qj, args, R_t, t_t):
    """Angle [rad] between the carried tilt and --tilt, for warnings."""
    R, _, _, _ = lp.fk_tcp(qj[None], R_t, t_t)
    theta, _ = tp.tilt_state(R[0])
    return abs(theta - np.deg2rad(args.tilt))


def make_tilt(node: Node, qj, theta_deg, args, verbose=True):
    """Plan the in-place rotation to tilt theta_deg (0 = level). Returns
    (q, t, None, None) or None."""
    try:
        plan = tp.plan_tilt(
            qj, np.deg2rad(theta_deg), args.lean,
            omega_max=np.deg2rad(args.omega), dt=args.dt, t_min=args.t_min,
            qdot_max=args.qdot, tcp_xyz=tuple(args.tcp_xyz),
            tcp_rpy=tuple(args.tcp_rpy), verbose=verbose)
    except lp.PlanCancelled:
        print("planning cancelled — no plan")
        return None
    if not plan["ok"]:
        node.get_logger().error("tilt planning FAILED — see messages above")
        return None
    if len(plan["q"]) < 2:
        print("already at that tilt — nothing to do")
        return None
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(args.save, q=plan["q"], t=plan["t"],
                        theta=plan["theta"])
    node.get_logger().info(
        f"tilt plan: {len(plan['q'])} samples, {plan['t'][-1]:.1f} s, "
        f"{np.rad2deg(plan['dtheta']):.1f} deg rotation, saved {args.save}")
    return plan["q"], plan["t"], None, None


def make_carry(node: Node, qj, args):
    """Setup move to the tilted carry attitude (joint interpolation, not
    attitude-constrained). Returns (q, t, None, None) or None."""
    carry = tp.plan_carry_tilted(qj, np.deg2rad(args.tilt), args.lean,
                                 qdot_max=args.qdot,
                                 tcp_xyz=tuple(args.tcp_xyz),
                                 tcp_rpy=tuple(args.tcp_rpy))
    if carry is None:
        print("no IK solution for the tilted carry pose here — try closer to "
              "the base")
        return None
    qs, ts, ang, dq = carry
    print(f"carry: reorient {np.rad2deg(ang):.1f} deg to tilt {args.tilt:g} deg "
          f"toward {args.lean}, largest joint move {np.rad2deg(dq):.1f} deg, "
          f"{ts[-1]:.1f} s (unconstrained joint move — don't do this holding "
          "liquid)")
    return qs, ts, None, None


def chain(first, second, dwell):
    """Concatenate two timed joint paths (second starts where first ends)
    with a `dwell` second hold in between; velocities are recomputed."""
    q1, t1 = first[0], first[1]
    q2, t2 = second[0], second[1]
    if np.abs(q1[-1] - q2[0]).max() > 1e-6:
        raise ValueError("paths do not join")
    return (np.vstack([q1, q2[1:]]),
            np.concatenate([t1, t1[-1] + dwell + t2[1:]]), None, None)


def make_move(node: Node, qj, p_goal, yaw_deg, args):
    """Move to p_goal at tilt --tilt. If the arm is not carrying that tilt
    yet, the in-place tilt to --tilt comes first and the move follows in the
    same trajectory, so every move ends (and travels) at --tilt.
    Returns (q, t, qd, qdd) or None."""
    R_t, t_t = lp.tcp_transform(args.tcp_xyz, args.tcp_rpy)
    pre = None
    if tilt_mismatch(qj, args, R_t, t_t) > TILT_TOL:
        R_cur, _, _, _ = lp.fk_tcp(qj[None], R_t, t_t)
        th, _ = tp.tilt_state(R_cur[0])
        print(f"carried tilt is {np.rad2deg(th):.1f} deg, not {args.tilt:g} — "
              f"applying the tilt in place first, then moving")
        pre = make_tilt(node, qj, args.tilt, args)
        if pre is None:
            return None
        qj = pre[0][-1]
    R_goal = None
    if yaw_deg is not None:
        R_cur, _, _, _ = lp.fk_tcp(qj[None], R_t, t_t)
        R_goal = lp.orientation_at_yaw(R_cur[0], np.deg2rad(yaw_deg))
    move = ml.make_plan(node, qj, np.asarray(p_goal, float), R_goal, args)
    if move is None or pre is None:
        return move
    plan = chain(pre, move, max(args.dwell, 0.0))
    np.savez_compressed(args.save, q=plan[0], t=plan[1], n_tilt=len(pre[0]))
    node.get_logger().info(f"tilt {pre[1][-1]:.1f} s + move {move[1][-1]:.1f} s "
                           f"= {plan[1][-1]:.1f} s, saved {args.save}")
    return plan


def confirm_and_execute(node: Node, plan, args, what="plan"):
    if not ml.exec_allowed(node, args):
        return False
    try:
        mode = "REAL arm" if args.real else "sim"
        sure = input(f"execute {what} ({mode})? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if sure not in ("y", "yes"):
        return False
    return ml.execute_moveit(node, ml.to_joint_trajectory(*plan))


INTERACTIVE_HELP = """\
tilt control — container carried at --tilt deg toward its --lean side
(all plans start from the CURRENT joint state):
  x y z [yaw]            move there keeping the attitude (yaw = absolute deg,
                         0 = straight ahead; omit = keep current yaw)
  tilt                   apply --tilt in place (from level or another tilt);
                         a move does this by itself when needed
  carry                  setup move to the tilted carry pose (gripper down)
                         — joint move, NOT attitude-constrained, no liquid
  where                  current tcp pose, yaw, tilt and lean direction
  preview  (p)           show the last plan in RViz (Planned Path display)
  exec     (go)          execute the last plan on the robot (asks first)
  set NAME VALUE         omega t_min dt qdot | lift step v a qddot dwell
  profile [lift|straight]
  help     (?) / quit (q)"""

SETTABLE = ("omega", "t_min", "dt", "qdot", "lift", "step", "v", "a",
            "qddot", "dwell")


def interactive(node: Node, args):
    R_t, t_t = lp.tcp_transform(args.tcp_xyz, args.tcp_rpy)
    last = None
    print(INTERACTIVE_HELP)
    while True:
        try:
            line = input(f"tilt{args.tilt:g}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        toks = line.split()
        cmd = toks[0].lower()

        if cmd in ("q", "quit", "exit"):
            return
        if cmd in ("h", "help", "?"):
            print(INTERACTIVE_HELP)
            continue
        if cmd == "set" and len(toks) == 3:
            name = toks[1].lower()
            if name not in SETTABLE:
                print(f"set takes: {' '.join(SETTABLE)}")
                continue
            try:
                setattr(args, name, float(toks[2]))
                print(f"{name} = {getattr(args, name)}")
            except ValueError:
                print(f"not a number: {toks[2]}")
            continue
        if cmd == "profile":
            if len(toks) == 2 and toks[1].lower() in lp.PROFILES:
                args.profile = toks[1].lower()
            print(f"profile = {args.profile}")
            continue
        if cmd in ("p", "preview"):
            if last is None:
                print("no plan yet")
                continue
            ml.preview(node, *last)
            continue
        if cmd in ("e", "exec", "go"):
            if last is None:
                print("no plan yet")
                continue
            if confirm_and_execute(node, last, args):
                last = None
            continue

        qj = ml.read_current_joints(node, args.joint_states)
        if qj is None:
            print(f"no JointState on '{args.joint_states}' — driver running?")
            continue
        if cmd == "where":
            print(describe(qj, R_t, t_t))
            continue
        if cmd == "tilt":
            last = make_tilt(node, qj, args.tilt, args)
            continue
        if cmd == "carry":
            last = make_carry(node, qj, args)
            continue
        try:
            vals = [float(x) for x in toks]
        except ValueError:
            print("unknown command (try 'help')")
            continue
        if len(vals) not in (3, 4):
            print("goal takes x y z [yaw_deg]")
            continue
        last = make_move(node, qj, vals[:3], vals[3] if len(vals) == 4 else None,
                         args)


def add_common_args(ap):
    ap.add_argument("--tilt", type=float, required=True, metavar="DEG",
                    help="carried tilt angle from level [deg] (required)")
    ap.add_argument("--lean", choices=tuple(tp.LEAN), default="front",
                    help="container side it leans toward (default front = "
                         "the side facing +Z at yaw 0)")
    ap.add_argument("--omega", type=float, default=10.0,
                    help="peak tilt rate [deg/s] for in-place tilts")
    ap.add_argument("--t-min", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--lift", type=float, default=0.08)
    ap.add_argument("--profile", choices=lp.PROFILES, default="lift")
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--v", type=float, default=0.06, help="max tcp speed [m/s]")
    ap.add_argument("--a", type=float, default=0.2, help="max tcp accel [m/s^2]")
    ap.add_argument("--qdot", type=float, default=1.0)
    ap.add_argument("--qddot", type=float, default=3.0)
    ap.add_argument("--dwell", type=float, default=0.1)
    ap.add_argument("--tcp-xyz", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--tcp-rpy", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--real", action="store_true",
                    help="work with the REAL arm (feedback/joint_states)")
    ap.add_argument("--joint-states", default=None)
    ap.add_argument("--save", default=os.path.join(OUT, "tilt.npz"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--apply", action="store_true",
                    help="plan the in-place tilt to --tilt and exit (with "
                         "--execute / --preview)")
    ap.add_argument("--goal", type=float, nargs=3, default=None,
                    metavar="V", help="x y z: plan an attitude-keeping move")
    ap.add_argument("--goal-yaw", type=float, default=None)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    if args.joint_states is None:
        args.joint_states = ("feedback/joint_states" if args.real
                             else "control/joint_states")

    rclpy.init()
    node = Node("nero_tilt_control")
    try:
        if not args.apply and args.goal is None:
            interactive(node, args)
            return
        qj = ml.read_current_joints(node, args.joint_states)
        if qj is None:
            node.get_logger().error(
                f"no JointState on '{args.joint_states}' — driver running?")
            return
        R_t, t_t = lp.tcp_transform(args.tcp_xyz, args.tcp_rpy)
        print(describe(qj, R_t, t_t))
        if args.apply:
            plan = make_tilt(node, qj, args.tilt, args)
        else:
            plan = make_move(node, qj, args.goal, args.goal_yaw, args)
        if plan is None:
            return
        if args.execute:
            confirm_and_execute(node, plan, args)
        elif args.preview:
            ml.preview(node, *plan)
        else:
            node.get_logger().info("plan-only mode (use --execute or --preview)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
