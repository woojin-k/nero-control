#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show the Nero arm's current joint angles (J1..Jn). Read-only.

Reads get_joint_angles() and prints each joint in native SDK units (radians)
plus a degrees convenience column. Nero reports 7 joint values (J1..J6 plus J7).
Does NOT enable motors or move the arm.

In normal/FOLLOWER mode the joints come from get_joint_angles(). In LEADER mode
the arm broadcasts leader-joint frames instead, so this falls back to
get_leader_joint_angles() and labels the source — meaning joint angles are shown
in both modes. (Cartesian pose, by contrast, is NOT available in leader mode;
see show_pose.py.)

Usage:
    ./.venv/bin/python src/nero_ctrl/show_joints.py --can can0
    ./.venv/bin/python src/nero_ctrl/show_joints.py --can can0 --watch 10   # 10 Hz

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse
import math
import time

# Sibling module in src/nero_ctrl/ (reuses its connect helper).
import connect_check as cc


def read_joints(robot, timeout=1.5):
    """Return (message, source), retrying briefly. (None, None) if unavailable.

    Tries normal joint feedback first (get_joint_angles, present in
    normal/FOLLOWER mode); if absent, falls back to the leader-joint broadcast
    (get_leader_joint_angles, present in LEADER mode).
    """
    end = time.time() + timeout
    while True:
        ja = robot.get_joint_angles()
        if ja is not None:
            return ja, "measured (get_joint_angles)"
        if time.time() >= end:
            break
        time.sleep(0.1)
    # Fallback: in LEADER mode the arm broadcasts leader-joint states instead.
    try:
        lja = robot.get_leader_joint_angles()
    except Exception:  # noqa: BLE001
        lja = None
    if lja is not None and getattr(lja, "hz", 0):
        return lja, "leader broadcast (get_leader_joint_angles)"
    return None, None


def print_joints(ja, source):
    """Print each joint angle (rad + deg), noting which feedback source it is."""
    joints = ja.msg
    print(f"[joints] joint angles  (n={len(joints)}, "
          f"hz={round(getattr(ja, 'hz', 0))}, source: {source})")
    for i, rad in enumerate(joints, start=1):
        print(f"  J{i} : {rad:+.4f} rad    ({math.degrees(rad):+8.2f} deg)")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Show the Nero joint angles (J1..Jn). "
                    "Read-only; does not enable motors.")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="auto",
                   choices=["auto", "default", "v111", "v112", "v120"],
                   help="firmware/parser; 'auto' detects it (default)")
    p.add_argument("--watch", type=float, default=None, metavar="HZ",
                   help="continuously refresh at HZ (0.5-50); Ctrl+C to stop")
    return p.parse_args()


def main():
    args = _parse_args()
    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[show_joints] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        print("[show_joints] is the interface up? bring it up with:")
        print("  sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/"
              "scripts/linux/can_activate.sh can0 1000000")
        raise SystemExit(1)

    try:
        if args.watch:
            hz = max(0.5, min(args.watch, 50.0))
            interval = 1.0 / hz
            try:
                while True:
                    ja, source = read_joints(robot, timeout=interval)
                    if ja is not None:
                        print_joints(ja, source)
                    else:
                        print("[joints] (no joint feedback — parser mismatch? "
                              "try --fw v112)")
                    print()
                    time.sleep(interval)
            except KeyboardInterrupt:
                print("\n[show_joints] stopped.")
            raise SystemExit(0)

        ja, source = read_joints(robot)
        if ja is None:
            print("[joints] no joint feedback available (measured or leader). "
                  "Likely a firmware-parser mismatch — pass the right --fw "
                  "(e.g. --fw v112).")
            raise SystemExit(2)
        print_joints(ja, source)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
