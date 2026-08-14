#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Move the Nero arm to a saved waypoint. ACTUATES.

!! THIS MOVES THE ARM. Loads a waypoint JSON from src/data/waypoints/ and drives
the arm to it. Clear the workspace and be ready to hit the e-stop. It asks for
y/N confirmation before enabling/moving unless --yes is given, and defaults to a
low speed. The arm must be in FOLLOWER / normal control mode (not LEADER).

Which method:
  * joints    (default when the waypoint has joints) — MOVE J, no IK. Reliably
    reproduces the exact captured posture, even where Cartesian IK would fail.
  * cartesian (default only if the waypoint has no joints) — MOVE P via IK.
  * linear    (only when forced with --linear) — MOVE L: the flange travels a
    straight line to the waypoint's Cartesian pose; the whole path must be
    reachable, so it can reject targets MOVE P would accept.
Force one with --joints, --cartesian, or --linear.

Usage:
    ./.venv/bin/python src/nero_ctrl/goto_waypoint.py --name home
    ./.venv/bin/python src/nero_ctrl/goto_waypoint.py --name home --cartesian --speed 15
    ./.venv/bin/python src/nero_ctrl/goto_waypoint.py --list

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse

# Sibling modules in src/nero_ctrl/.
import connect_check as cc
import waypoints as wp
import move_to_pose as mp
import move_to_joints as mj


def _choose_method(record, want):
    """Pick 'joints' / 'cartesian' / 'linear' for this waypoint.

    Returns (method, error). 'linear' is never chosen by auto — the straight-
    line path is a deliberate request (it can reject reachable targets).
    """
    has_j = record.get("joints_deg") is not None
    has_c = record.get("cartesian") is not None
    if want == "joints":
        return ("joints", None) if has_j else (None, "waypoint has no joints")
    if want in ("cartesian", "linear"):
        return (want, None) if has_c else (None, "waypoint has no cartesian pose")
    # auto: prefer joints (exact, IK-free), else cartesian
    if has_j:
        return "joints", None
    if has_c:
        return "cartesian", None
    return None, "waypoint has neither joints nor cartesian"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Move the Nero arm to a saved waypoint. ACTUATES — asks "
                    "before moving.")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="v112",
                   choices=["default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default: v112)")
    p.add_argument("--name", help="waypoint name to go to")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--joints", action="store_true",
                   help="force joint-space move (MOVE J)")
    g.add_argument("--cartesian", action="store_true",
                   help="force Cartesian move (MOVE P / IK)")
    g.add_argument("--linear", action="store_true",
                   help="force straight-line Cartesian move (MOVE L)")
    p.add_argument("--speed", type=int, default=20,
                   help="move speed percent 1-100 (default: 20, kept low)")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="seconds to wait for the motion to finish")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
    p.add_argument("--list", action="store_true",
                   help="list saved waypoints and exit (no connection)")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.list:
        names = wp.list_waypoints()
        if not names:
            print(f"[goto] no waypoints yet ({wp.WAYPOINTS_DIR})")
        else:
            print(f"[goto] {len(names)} waypoint(s):")
            for name in names:
                print(f"  {wp.format_waypoint(wp.load_waypoint(name))}")
        raise SystemExit(0)

    if not args.name:
        print("[goto] --name is required (or use --list).")
        raise SystemExit(2)
    try:
        record = wp.load_waypoint(args.name)
    except FileNotFoundError:
        print(f"[goto] no waypoint named '{args.name}'. use --list to see saved ones.")
        raise SystemExit(2)

    want = ("joints" if args.joints else "cartesian" if args.cartesian
            else "linear" if args.linear else "auto")
    method, err = _choose_method(record, want)
    if err:
        print(f"[goto] {err}: '{args.name}'.")
        raise SystemExit(2)

    print(f"[goto] waypoint '{args.name}': {wp.format_waypoint(record)}")
    print(f"[goto] method  : {method}   speed: {args.speed}%")

    if args.speed < 1 or args.speed > 100:
        print(f"[goto] --speed must be 1-100, got {args.speed}")
        raise SystemExit(2)

    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[goto] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    code = 0
    try:
        confirm = (lambda _: True) if args.yes else None
        if method == "joints":
            joints_rad = wp.to_joints_rad(record)
            outcome, status = mj.move_to_joints(robot, joints_rad, args.speed,
                                                args.timeout, confirm)
            code = mj._report(outcome, status, args.timeout, tag="goto")
        else:
            target = wp.to_cartesian_target(record)
            outcome, status = mp.move_to_pose(robot, target, args.speed,
                                              args.timeout, confirm,
                                              linear=(method == "linear"))
            code = _report_cartesian(outcome, status, args.timeout)

        # show where the arm ended up
        final, src = mp.sp.read_pose(robot)
        if final is not None:
            print(f"[goto] final   : {mp._fmt_pose(final)}   ({src})")
    finally:
        robot.disconnect()
    raise SystemExit(code)


def _report_cartesian(outcome, status, timeout):
    """Outcome message for a Cartesian goto. Returns exit code."""
    if outcome == "leader":
        print("[goto] arm is in LEADER mode — cannot move. Run "
              "set_follower_mode.py first.")
        return 2
    if outcome == "cancelled":
        print("[goto] cancelled. 취소되었습니다.")
        return 3
    if outcome == "enable_failed":
        print("[goto] failed to enable motors within timeout — aborting.")
        return 4
    if outcome == "reached":
        print("[goto] reached target.")
        return 0
    if outcome == "unreachable":
        print(f"[goto] NOT reachable (arm_status={status}). This waypoint's "
              "Cartesian pose has no IK solution from here — try --joints.")
        return 5
    print(f"[goto] motion not confirmed within {timeout}s "
          f"(arm_status={status or '?'}). check the arm.")
    return 6


if __name__ == "__main__":
    main()
