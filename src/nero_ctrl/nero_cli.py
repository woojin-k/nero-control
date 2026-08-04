#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified Nero console: status, pose, joints, mode, moves, and waypoints in ONE
prompt over a single shared CAN connection.

This is the CLI presentation layer. It reuses the same robot-taking core
functions the scripts use (connect_check, show_pose, show_joints, mode_control,
move_to_pose, move_to_joints, waypoints), injecting a console y/N confirm for
the actuating commands. A future GUI can reuse the very same cores with a popup
confirm — nothing here reimplements arm logic.

Running this does NOT enable motors or move anything on its own; actuation only
happens when you type a moving command and confirm it.

Usage:
    ./.venv/bin/python src/nero_ctrl/nero_cli.py --can can0 --fw v112
"""
import argparse
import math

# Reusable cores (each takes an already-connected robot).
import connect_check as cc
import show_pose as sp
import show_joints as sj
import mode_control as mc
import move_to_pose as mp
import move_to_joints as mj
import waypoints as wp
import goto_waypoint as gw
import check_gripper as cg
import move_gripper as mg
import gripper_info as gi
import ctrl_mode as cm


HELP = """\
commands (one shared connection, one prompt):
  status               connection + mode (linkage / ctrl_mode) + fps
  pose                 current flange pose (X Y Z + roll pitch yaw)
  joints               current joint angles (J1..J7)
  mode                 show current mode
  mode follower|leader switch linkage mode (asks y/N)
  teach                enter drag-teach: arm becomes hand-movable (asks y/N)
  canmode              exit drag-teach, back to CAN control     (asks y/N)

  movep X Y Z R P Y    Cartesian move, mm + deg (MOVE P / IK)   (asks y/N)
  movej J1 .. J7       joint move, 7 values in deg (MOVE J)      (asks y/N)
  speed [pct]          show / set move speed % for this session (default 20)

  save <name> [desc]   capture current posture -> waypoint (both parts)
  points               list saved waypoints
  goto <name> [joints|cartesian]   move to a waypoint (asks y/N)

  gripper              check if an AGX gripper is connected (read-only)
  gripper info                  arm fw + gripper params/state (read-only)
  gripper calib [70|100]        zero-point calibration (close jaws by hand)
  gripper open|close [forceN]   open fully / close into a grasp
  gripper to <val> [forceN]     move to width mm / angle deg
                                (moves enable ALL motors — arm holds pose —
                                 and ask y/N)
  gripper reset                 disable + clear gripper errors

  help                 show this help
  quit                 exit (motors are left as-is; the arm holds its pose)
"""


def _move_msg(outcome, status):
    return {
        "invalid": "invalid target",
        "leader": "arm is in LEADER mode — run 'mode follower' first",
        "cancelled": "cancelled",
        "enable_failed": "failed to enable motors",
        "reached": "reached target",
        "unreachable": f"NOT reachable (arm_status={status})",
        "timeout": "motion not confirmed (timeout) — check the arm",
    }.get(outcome, str(outcome))


def cmd_status(robot):
    m = cc.read_modes(robot)
    print(f"[status] linkage : {m['linkage_role'] or '?'}   "
          f"ctrl_mode: {m['ctrl_mode'] or '(no status feedback)'}")
    print(f"[status] comm ok : {robot.is_ok()}   fps: {robot.get_fps()}")


def cmd_pose(robot):
    pose, src = sp.read_pose(robot)
    if pose is None:
        print("[pose] unavailable (parser mismatch?)")
    else:
        sp.print_pose(pose, src)


def cmd_joints(robot):
    ja, src = sj.read_joints(robot)
    if ja is None:
        print("[joints] unavailable (parser mismatch?)")
    else:
        sj.print_joints(ja, src)


def cmd_mode(robot, rest):
    if not rest:
        m = cc.read_modes(robot)
        print(f"[mode] {cc.format_mode_oneline(m)}")
        return
    target = rest[0].lower()
    if target not in ("follower", "leader"):
        print("usage: mode [follower|leader]")
        return
    ok, _before, after = mc.set_mode(robot, target)   # console y/N confirm
    if not ok:
        print("[mode] cancelled.")
        return
    print(f"[mode] now: {cc.format_mode_oneline(after)}")


def cmd_speed(state, rest):
    if not rest:
        print(f"[speed] {state['speed']}%")
        return
    try:
        v = int(rest[0])
    except ValueError:
        print(f"[speed] invalid: {rest[0]}")
        return
    if not 1 <= v <= 100:
        print("[speed] must be 1-100")
        return
    state["speed"] = v
    print(f"[speed] set to {v}%")


def cmd_movep(robot, state, rest):
    if len(rest) != 6:
        print("usage: movep X Y Z ROLL PITCH YAW  (mm, deg)")
        return
    try:
        vals = [float(x) for x in rest]
    except ValueError:
        print(f"[movep] invalid numbers: {' '.join(rest)}")
        return
    target = mp.build_target(vals, si=False)      # mm/deg -> native m/rad
    errs = mp.validate_target(target)
    if errs:
        print("[movep] out of range: " + "; ".join(errs))
        return
    outcome, status = mp.move_to_pose(robot, target, state["speed"],
                                      state["timeout"])
    print(f"[movep] {_move_msg(outcome, status)}")


def cmd_movej(robot, state, rest):
    if len(rest) != mj.JOINT_NUMS:
        print(f"usage: movej J1 .. J{mj.JOINT_NUMS}  (deg)")
        return
    try:
        joints = [math.radians(float(x)) for x in rest]
    except ValueError:
        print(f"[movej] invalid numbers: {' '.join(rest)}")
        return
    outcome, status = mj.move_to_joints(robot, joints, state["speed"],
                                        state["timeout"])
    print(f"[movej] {_move_msg(outcome, status)}")


def cmd_save(robot, rest):
    if not rest:
        print("usage: save <name> [description]")
        return
    name, desc = rest[0], " ".join(rest[1:])
    try:
        _path, existed, rec = wp.capture(robot, name, desc)
    except (ValueError, FileExistsError) as exc:
        print(f"[save] not saved: {exc}")
        return
    verb = "overwrote" if existed else "saved"
    print(f"[save] {verb} '{name}': {wp.format_waypoint(rec)}")


def cmd_points():
    names = wp.list_waypoints()
    if not names:
        print(f"[points] none yet ({wp.WAYPOINTS_DIR})")
        return
    print(f"[points] {len(names)} waypoint(s):")
    for name in names:
        print(f"  {wp.format_waypoint(wp.load_waypoint(name))}")


def cmd_gripper(robot, state, rest):
    # init_effector is one-shot per robot instance, so init lazily and cache it.
    if state.get("effector") is None:
        try:
            state["effector"] = robot.init_effector(
                robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        except Exception as exc:  # noqa: BLE001
            print(f"[gripper] init failed: {type(exc).__name__}: {exc}")
            return
    effector = state["effector"]

    if not rest:                      # bare 'gripper' = read-only check
        for line in cg.format_report(cg.check_gripper(effector)):
            print(line)
        return

    sub = rest[0].lower()
    if sub == "info":
        for line in gi.format_info(gi.gripper_info(robot, effector)):
            print(line)
        return
    if sub == "reset":
        effector.reset_gripper()      # disable + clear errors
        print("[gripper] reset sent (disabled + errors cleared).")
        return
    if sub == "calib":
        rng = None
        if len(rest) > 1:
            if rest[1] not in ("70", "100"):
                print("usage: gripper calib [70|100]  (stroke in mm)")
                return
            rng = int(rest[1]) / 1000.0
        outcome, param_ok = mg.calibrate(robot, effector, rng)  # y/N + Enter
        mg.report_calib(outcome, param_ok, tag="gripper")
        return
    if sub in ("open", "close"):
        spec, force_arg = sub, rest[1:]
    elif sub == "to":
        if len(rest) < 2:
            print("usage: gripper to <val> [forceN]")
            return
        spec, force_arg = rest[1], rest[2:]
    else:
        print("usage: gripper [info|calib|open|close|to <val>|reset] [forceN]")
        return
    force = 1.0
    if force_arg:
        try:
            force = float(force_arg[0])
        except ValueError:
            print(f"[gripper] invalid force: {force_arg[0]}")
            return

    mode, chk = mg.read_mode(effector)
    if mode is None:
        for line in cg.format_report(chk):
            print(line)
        print("[gripper] refusing to command a gripper that shows no "
              "telemetry.")
        return
    target, err = mg.resolve_target(effector, mode, spec)
    if err:
        print(f"[gripper] {err}")
        return
    cur = chk["status"].msg
    print(f"[gripper] mode {mode}: {mg.fmt_value(cur.value, mode)} -> "
          f"{mg.fmt_value(target, mode)} at {force:.2f} N")
    outcome, msg = mg.move_gripper(robot, effector, mode, target, force,
                                   state["timeout"])   # console y/N confirm
    mg.report(outcome, msg, mode, target, state["timeout"], tag="gripper")


def cmd_goto(robot, state, rest):
    if not rest:
        print("usage: goto <name> [joints|cartesian]")
        return
    name = rest[0]
    try:
        rec = wp.load_waypoint(name)
    except FileNotFoundError:
        print(f"[goto] no waypoint '{name}' (try 'points')")
        return
    want = rest[1].lower() if len(rest) > 1 else "auto"
    if want not in ("auto", "joints", "cartesian"):
        print("usage: goto <name> [joints|cartesian]")
        return
    method, err = gw._choose_method(rec, want)
    if err:
        print(f"[goto] {err}")
        return
    print(f"[goto] '{name}' via {method}: {wp.format_waypoint(rec)}")
    if method == "joints":
        outcome, status = mj.move_to_joints(robot, wp.to_joints_rad(rec),
                                            state["speed"], state["timeout"])
    else:
        outcome, status = mp.move_to_pose(robot, wp.to_cartesian_target(rec),
                                          state["speed"], state["timeout"])
    print(f"[goto] {_move_msg(outcome, status)}")


def repl(robot, state):
    print(HELP)
    cmd_status(robot)
    while True:
        try:
            line = input("nero> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        tokens = line.split()
        head, rest = tokens[0].lower(), tokens[1:]

        if head in ("quit", "exit", "q"):
            break
        elif head == "help":
            print(HELP)
        elif head == "status":
            cmd_status(robot)
        elif head == "pose":
            cmd_pose(robot)
        elif head == "joints":
            cmd_joints(robot)
        elif head == "mode":
            cmd_mode(robot, rest)
        elif head == "teach":
            ok, _b, after = cm.set_ctrl_mode(robot, "teach")   # console y/N
            cm.report("teach", ok, after)
        elif head == "canmode":
            ok, _b, after = cm.set_ctrl_mode(robot, "can")     # console y/N
            cm.report("can", ok, after)
        elif head == "speed":
            cmd_speed(state, rest)
        elif head == "movep":
            cmd_movep(robot, state, rest)
        elif head == "movej":
            cmd_movej(robot, state, rest)
        elif head == "save":
            cmd_save(robot, rest)
        elif head == "points":
            cmd_points()
        elif head == "goto":
            cmd_goto(robot, state, rest)
        elif head == "gripper":
            cmd_gripper(robot, state, rest)
        else:
            print(f"unknown command: {head}  (try 'help')")

    print("[nero_cli] bye. (motors left as-is — the arm holds its pose.)")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Unified Nero console: status / pose / joints / mode / "
                    "moves / waypoints in one prompt.")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="v112",
                   choices=["auto", "default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default: v112)")
    p.add_argument("--speed", type=int, default=20,
                   help="initial move speed percent (default: 20)")
    return p.parse_args()


def main():
    args = _parse_args()
    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[nero_cli] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(f"[nero_cli] connected on {args.can}.")
    state = {"speed": max(1, min(args.speed, 100)), "timeout": 15.0,
             "effector": None}
    try:
        repl(robot, state)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
