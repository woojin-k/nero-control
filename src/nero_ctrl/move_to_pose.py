#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Move the Nero flange to a target Cartesian pose (MOVE P / IK). ACTUATES.

!! THIS MOVES THE ARM. It enables the motors and commands a Cartesian pose via
inverse kinematics. Clear the workspace, be ready to hit the e-stop, and keep
away from the arm. It always asks for y/N confirmation before enabling/moving
unless --yes is given, and defaults to a low speed.

Target pose is [x, y, z, roll, pitch, yaw] in mm + degrees by default; it is
converted to the SDK's native metres + radians internally. Pass --si to give
the target directly in metres + radians instead. Per the SDK, roll & yaw must
be in [-pi, pi] and pitch in [-pi/2, pi/2] (i.e. roll/yaw [-180, 180] deg,
pitch [-90, 90] deg).

Reachability is judged the way the arm reports it: after the command, arm_status
NO_SOLUTION / SINGULARITY_POINT / TARGET_POS_EXCEEDS_LIMIT means the target is
unreachable and the arm does not move. (If a Cartesian target is rejected, a
joint-space move can still reach poses IK refuses.)

With --linear the arm uses MOVE L instead: the flange travels in a straight
line to the same target. Every point on the line must be reachable, so MOVE L
can reject targets a plain MOVE P would accept.

The arm must be in FOLLOWER / normal control mode (not LEADER). This script does
NOT disable the motors afterwards, so the arm holds its pose; do not leave it
unattended enabled.

Usage:
    ./.venv/bin/python src/nero_ctrl/move_to_pose.py --to -400 0 400 -90 0 -180
    ./.venv/bin/python src/nero_ctrl/move_to_pose.py --to -0.4 0.0 0.4 -1.5708 0.0 -3.14159 --si

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse
import math
import time

# Sibling modules in src/nero_ctrl/ (reuse connect + pose reading).
import connect_check as cc
import show_pose as sp

# arm_status values (official enum names) that mean the target is unreachable.
UNREACHABLE = {"NO_SOLUTION", "SINGULARITY_POINT", "TARGET_POS_EXCEEDS_LIMIT"}

WARN_MOVE = ("[move] !! THE ARM WILL MOVE. Motors will be ENABLED and the "
             "flange driven to the target via IK.\n"
             "[move]   !! 팔이 움직입니다. 모터가 켜지고 IK로 목표 자세까지 "
             "이동합니다. 주변을 비우고 비상정지 준비하세요.")


def _enum_name(v):
    return getattr(v, "name", None) or (str(v) if v is not None else None)


def _console_confirm(message: str) -> bool:
    print(message)
    try:
        return input("proceed? 진행할까요? (y/N): ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def build_target(vals, si):
    """Convert the 6 CLI values to native units [x,y,z(m), r,p,y(rad)].

    CLI units are mm + degrees by default; pass si=True to interpret them as
    the native metres + radians instead (no conversion).
    """
    x, y, z, roll, pitch, yaw = vals
    if si:
        return [x, y, z, roll, pitch, yaw]
    return [x / 1000.0, y / 1000.0, z / 1000.0,
            math.radians(roll), math.radians(pitch), math.radians(yaw)]


def validate_target(pose):
    """Return a list of range-violation messages (empty if OK)."""
    _, _, _, roll, pitch, yaw = pose
    errs = []
    if not -math.pi <= roll <= math.pi:
        errs.append(f"roll {roll:+.4f} rad out of [-pi, pi]")
    if not -math.pi / 2 <= pitch <= math.pi / 2:
        errs.append(f"pitch {pitch:+.4f} rad out of [-pi/2, pi/2]")
    if not -math.pi <= yaw <= math.pi:
        errs.append(f"yaw {yaw:+.4f} rad out of [-pi, pi]")
    return errs


def enable_with_timeout(robot, timeout=5.0):
    """Enable all joints, bounded by `timeout` (avoids an unbounded busy-wait)."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            if robot.enable():
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.1)
    return False


def watch_motion(robot, timeout=15.0, poll=0.1):
    """Watch arm_status/motion_status after a move command.

    Returns (outcome, last_arm_status):
      "reached"     motion_status == REACH_TARGET_POS_SUCCESSFULLY
      "unreachable" arm_status in UNREACHABLE (IK gave no/again a bad solution)
      "timeout"     neither within `timeout`
    The unreachable status can appear a fraction of a second AFTER the command,
    so it is checked on every poll (not just once).
    """
    time.sleep(0.3)  # let the command register
    end = time.time() + timeout
    last = None
    while time.time() < end:
        st = robot.get_arm_status()
        if st is not None:
            last = _enum_name(st.msg.arm_status)
            if last in UNREACHABLE:
                return "unreachable", last
            if _enum_name(st.msg.motion_status) == "REACH_TARGET_POS_SUCCESSFULLY":
                return "reached", last
        time.sleep(poll)
    return "timeout", last


def _fmt_pose(pose):
    x, y, z, r, p, yaw = pose
    return (f"XYZ=({x * 1000:+.1f},{y * 1000:+.1f},{z * 1000:+.1f}) mm  "
            f"RPY=({math.degrees(r):+.1f},{math.degrees(p):+.1f},"
            f"{math.degrees(yaw):+.1f}) deg")


def move_to_pose(robot, target, speed=20, timeout=15.0, confirm=None,
                 linear=False):
    """Move the flange to `target` via MOVE P / IK on a connected `robot`.

    Reusable core for a CLI/GUI: takes an already-connected driver and the
    target as native units [x, y, z (m), roll, pitch, yaw (rad)]. It is
    read-only until confirmed.

    `confirm(message) -> bool` gates the enable + move (default: console y/N).
    A GUI can inject a popup-based confirm; pass ``lambda _: True`` to skip the
    prompt. Motors are NOT disabled afterwards (the arm holds its pose).

    ``linear=True`` uses MOVE L instead of MOVE P: the flange travels in a
    straight line to the target (same pose format). The whole line must be
    reachable, so MOVE L rejects targets MOVE P would accept.

    Returns ``(outcome, arm_status)`` where outcome is one of:
      "invalid" (out of range), "leader" (wrong mode), "cancelled",
      "enable_failed", "reached", "unreachable", "timeout".
    """
    if validate_target(target):
        return "invalid", None
    if cc.read_modes(robot).get("linkage_role") == "LEADER":
        return "leader", None
    confirm = confirm or _console_confirm
    if not confirm(WARN_MOVE):
        return "cancelled", None
    if not enable_with_timeout(robot):
        return "enable_failed", None
    robot.set_speed_percent(speed)
    if linear:
        robot.move_l(target)      # sets MOVE L mode internally (straight line)
    else:
        robot.move_p(target)      # sets MOVE P mode internally, sends the IK cmd
    return watch_motion(robot, timeout)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Move the Nero flange to a Cartesian pose (MOVE P/IK). "
                    "ACTUATES — asks before moving.")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="v112",
                   choices=["default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default: v112)")
    p.add_argument("--to", nargs=6, type=float, required=True,
                   metavar=("Xmm", "Ymm", "Zmm", "ROLLdeg", "PITCHdeg",
                            "YAWdeg"),
                   help="target pose in mm + degrees (x y z roll pitch yaw)")
    p.add_argument("--si", action="store_true",
                   help="interpret --to as native metres + radians instead of "
                        "mm + degrees")
    p.add_argument("--speed", type=int, default=20,
                   help="move speed percent 1-100 (default: 20, kept low)")
    p.add_argument("--linear", action="store_true",
                   help="use MOVE L (straight-line path) instead of MOVE P")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="seconds to wait for the motion to finish")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
    return p.parse_args()


def main():
    args = _parse_args()
    if not 1 <= args.speed <= 100:
        print(f"[move] --speed must be 1-100, got {args.speed}")
        raise SystemExit(2)

    target = build_target(args.to, args.si)
    errs = validate_target(target)
    if errs:
        print("[move] target out of range — not moving:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(2)

    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[move] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    code = 0
    try:
        cur, cur_src = sp.read_pose(robot)
        if cur is not None:
            print(f"[move] current : {_fmt_pose(cur)}   ({cur_src})")
        print(f"[move] target  : {_fmt_pose(target)}")
        mode = ("MOVE L (linear path)" if args.linear
                else "MOVE P (Cartesian IK)")
        print(f"[move] speed   : {args.speed}%   mode: {mode}")

        confirm = (lambda _: True) if args.yes else None
        outcome, status = move_to_pose(robot, target, args.speed, args.timeout,
                                       confirm, linear=args.linear)

        if outcome == "leader":
            print("[move] arm is in LEADER (zero-force drag) mode — cannot do a "
                  "CAN move. Run set_follower_mode.py first.")
            code = 2
        elif outcome == "cancelled":
            print("[move] cancelled. 취소되었습니다.")
            code = 3
        elif outcome == "enable_failed":
            print("[move] failed to enable motors within timeout — aborting.")
            code = 4
        elif outcome == "reached":
            print("[move] reached target.")
        elif outcome == "unreachable":
            print(f"[move] NOT reachable: arm_status={status}. The arm did not "
                  "move to this Cartesian target (IK found no solution). Try a "
                  "target further inside the workspace, or a joint-space move "
                  "(move_j) — it can reach configurations IK rejects.")
            code = 5
        else:  # timeout
            print(f"[move] motion not confirmed within {args.timeout}s "
                  f"(arm_status={status or '?'}). check the arm.")
            code = 6

        if outcome in ("reached", "unreachable", "timeout"):
            final, final_src = sp.read_pose(robot)
            if final is not None:
                print(f"[move] final   : {_fmt_pose(final)}   ({final_src})")
            print("[move] note: motors left ENABLED (holding pose). Do not leave "
                  "the arm unattended enabled.")
    finally:
        robot.disconnect()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
