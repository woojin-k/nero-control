#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Move the Nero arm to target joint angles (MOVE J, no IK). ACTUATES.

!! THIS MOVES THE ARM. It enables the motors and commands the joints directly.
Because there is no IK, a joint move reliably reaches configurations that a
Cartesian MOVE P would reject (NO_SOLUTION) — this is the dependable way to
replay a captured posture. Clear the workspace and be ready to hit the e-stop.

Target is 7 joint angles. Degrees by default (converted to the SDK's native
radians internally); pass --rad to give radians directly. It always asks for
y/N confirmation before enabling/moving unless --yes is given, and defaults to
a low speed. The arm must be in FOLLOWER / normal control mode (not LEADER).
Motors are left enabled (holding) on completion.

Usage:
    ./.venv/bin/python src/nero_ctrl/move_to_joints.py --to 0 -100 20 120 0 0 -30
    ./.venv/bin/python src/nero_ctrl/move_to_joints.py --to 0 -1.74 0.35 2.09 0 0 -0.52 --rad --speed 15

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse
import math

# Sibling modules: reuse the connection + shared motion helpers.
import connect_check as cc
import move_to_pose as mp

# Nero reports/accepts 7 joints.
JOINT_NUMS = 7

WARN_MOVE_J = ("[movej] !! THE ARM WILL MOVE. Motors will be ENABLED and the "
               "joints driven directly to the target (MOVE J, no IK).\n"
               "[movej]   !! 팔이 움직입니다. 모터가 켜지고 관절이 목표값으로 "
               "직접 이동합니다. 주변을 비우고 비상정지 준비하세요.")


def move_to_joints(robot, joints_rad, speed=20, timeout=15.0, confirm=None):
    """Move to `joints_rad` (7 radians) via MOVE J on a connected `robot`.

    Reusable core for a CLI/GUI. `confirm(message) -> bool` gates the enable +
    move (default: console y/N; a GUI can inject a popup; ``lambda _: True``
    skips). Never disables motors. Returns ``(outcome, arm_status)`` where
    outcome is one of: "invalid", "leader", "cancelled", "enable_failed",
    "reached", "unreachable", "timeout".
    """
    if len(joints_rad) != JOINT_NUMS:
        return "invalid", None
    if cc.read_modes(robot).get("linkage_role") == "LEADER":
        return "leader", None
    confirm = confirm or mp._console_confirm
    if not confirm(WARN_MOVE_J):
        return "cancelled", None
    if not mp.enable_with_timeout(robot):
        return "enable_failed", None
    robot.set_speed_percent(speed)
    robot.move_j(list(joints_rad))     # sets MOVE J mode internally
    return mp.watch_motion(robot, timeout)


def _fmt_joints(joints_rad):
    return "[" + ", ".join(f"{math.degrees(j):+.1f}" for j in joints_rad) + "] deg"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Move the Nero arm to target joint angles (MOVE J). "
                    "ACTUATES — asks before moving.")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="v112",
                   choices=["default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default: v112)")
    p.add_argument("--to", nargs=JOINT_NUMS, type=float, required=True,
                   metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
                   help="target joint angles in degrees (7 values)")
    p.add_argument("--rad", action="store_true",
                   help="interpret --to as radians instead of degrees")
    p.add_argument("--speed", type=int, default=20,
                   help="move speed percent 1-100 (default: 20, kept low)")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="seconds to wait for the motion to finish")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
    return p.parse_args()


def main():
    args = _parse_args()
    if not 1 <= args.speed <= 100:
        print(f"[movej] --speed must be 1-100, got {args.speed}")
        raise SystemExit(2)

    joints_rad = list(args.to) if args.rad else [math.radians(v) for v in args.to]

    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[movej] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    code = 0
    try:
        print(f"[movej] target : {_fmt_joints(joints_rad)}")
        print(f"[movej] speed  : {args.speed}%   mode: MOVE J (no IK)")
        confirm = (lambda _: True) if args.yes else None
        outcome, status = move_to_joints(robot, joints_rad, args.speed,
                                         args.timeout, confirm)
        code = _report(outcome, status, args.timeout)
    finally:
        robot.disconnect()
    raise SystemExit(code)


def _report(outcome, status, timeout, tag="movej"):
    """Print an outcome message and return the process exit code."""
    if outcome == "leader":
        print(f"[{tag}] arm is in LEADER mode — cannot move. Run "
              "set_follower_mode.py first.")
        return 2
    if outcome == "invalid":
        print(f"[{tag}] need {JOINT_NUMS} joint values.")
        return 2
    if outcome == "cancelled":
        print(f"[{tag}] cancelled. 취소되었습니다.")
        return 3
    if outcome == "enable_failed":
        print(f"[{tag}] failed to enable motors within timeout — aborting.")
        return 4
    if outcome == "reached":
        print(f"[{tag}] reached target.")
        return 0
    if outcome == "unreachable":
        print(f"[{tag}] arm reports {status} (a joint likely exceeds its limit).")
        return 5
    print(f"[{tag}] motion not confirmed within {timeout}s "
          f"(arm_status={status or '?'}). check the arm.")
    return 6


if __name__ == "__main__":
    main()
