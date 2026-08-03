#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show the Nero arm's current flange pose: X Y Z + roll pitch yaw. Read-only.

Prints the six values in native SDK units (X/Y/Z in metres, roll/pitch/yaw in
radians) plus a mm / degrees convenience column. Does NOT enable motors or move
the arm.

Pose source, to work in every mode (like the Piper tooling):
  * FOLLOWER / normal: the measured flange pose (get_flange_pose).
  * LEADER: the arm broadcasts no cartesian pose, so we compute it by forward
    kinematics (pyAgxArm.utiles.mdh_kinematics) from the joint angles, which are
    available from the leader-joint broadcast. FK matches the measured pose to
    ~1e-4, so the value is consistent across modes.
Pass --measured-only to disable the FK fallback (then leader mode shows nothing).

Euler convention (per the SDK): extrinsic XYZ == intrinsic ZYX,
R = Rz(yaw) * Ry(pitch) * Rx(roll).

Usage:
    ./.venv/bin/python src/nero_ctrl/show_pose.py --can can0
    ./.venv/bin/python src/nero_ctrl/show_pose.py --can can0 --watch 10   # 10 Hz

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse
import math
import time

from pyAgxArm.utiles.mdh_kinematics import get_mdh, fk_from_mdh

# Sibling modules in src/nero_ctrl/ (reuse connect + joint-reading helpers).
import connect_check as cc
import show_joints as sj

# Nero MDH table (7 links); loaded once — pure computation, no hardware.
_NERO_MDH = get_mdh("nero")


def read_pose(robot, timeout=1.5, allow_fk=True):
    """Return (pose6, source). Measured flange pose if available; otherwise, if
    allow_fk, forward-kinematics from the joint angles (works in leader mode).
    (None, None) if neither is available.
    """
    end = time.time() + timeout
    while True:
        fp = robot.get_flange_pose()
        if fp is not None:
            return list(fp.msg), (
                f"measured (get_flange_pose, hz={round(getattr(fp, 'hz', 0))})")
        if time.time() >= end:
            break
        time.sleep(0.1)

    if allow_fk:
        ja, jsrc = sj.read_joints(robot, timeout=0.5)
        if ja is not None:
            return fk_from_mdh(_NERO_MDH, list(ja.msg)), f"FK from {jsrc}"
    return None, None


def print_pose(pose, source):
    """Print the six pose values (native units + mm/deg convenience)."""
    x, y, z, roll, pitch, yaw = pose
    print(f"[pose] flange pose  (extrinsic XYZ / intrinsic ZYX, source: {source})")
    print(f"  X    : {x:+.4f} m      ({x * 1000:+8.1f} mm)")
    print(f"  Y    : {y:+.4f} m      ({y * 1000:+8.1f} mm)")
    print(f"  Z    : {z:+.4f} m      ({z * 1000:+8.1f} mm)")
    print(f"  roll : {roll:+.4f} rad    ({math.degrees(roll):+8.2f} deg)")
    print(f"  pitch: {pitch:+.4f} rad    ({math.degrees(pitch):+8.2f} deg)")
    print(f"  yaw  : {yaw:+.4f} rad    ({math.degrees(yaw):+8.2f} deg)")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Show the Nero flange pose (X Y Z + roll pitch yaw). "
                    "Read-only; does not enable motors.")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="auto",
                   choices=["auto", "default", "v111", "v112", "v120"],
                   help="firmware/parser; 'auto' detects it (default)")
    p.add_argument("--watch", type=float, default=None, metavar="HZ",
                   help="continuously refresh at HZ (0.5-50); Ctrl+C to stop")
    p.add_argument("--measured-only", action="store_true",
                   help="disable the FK fallback (leader mode shows nothing)")
    return p.parse_args()


def main():
    args = _parse_args()
    allow_fk = not args.measured_only
    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[show_pose] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        print("[show_pose] is the interface up? bring it up with:")
        print("  sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/"
              "scripts/linux/can_activate.sh can0 1000000")
        raise SystemExit(1)

    try:
        if args.watch:
            hz = max(0.5, min(args.watch, 50.0))
            interval = 1.0 / hz
            try:
                while True:
                    pose, source = read_pose(robot, timeout=interval,
                                             allow_fk=allow_fk)
                    if pose is not None:
                        print_pose(pose, source)
                    else:
                        print("[pose] (no pose — parser mismatch, or leader mode "
                              "with --measured-only)")
                    print()
                    time.sleep(interval)
            except KeyboardInterrupt:
                print("\n[show_pose] stopped.")
            raise SystemExit(0)

        pose, source = read_pose(robot, allow_fk=allow_fk)
        if pose is None:
            print("[pose] no pose available. In leader mode the cartesian pose "
                  "is not broadcast; drop --measured-only to compute it by FK, "
                  "or pass the right --fw.")
            raise SystemExit(2)
        print_pose(pose, source)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
