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
import time

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
import sequences as sq


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
  movep rel DX DY DZ [DR DP DY]   relative Cartesian move, mm + deg;
                       R P Y omitted = keep current orientation  (asks y/N)
  movej J1 .. J7       joint move, 7 values in deg (MOVE J)      (asks y/N)
  movej rel D1 .. D7   relative joint move, 7 deltas in deg      (asks y/N)
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

  sleep S              wait S seconds (mainly useful inside sequences)

  seq record <name>    start capturing typed move commands into a sequence
  seq stop | cancel    save / discard the sequence being captured
  seq add <name> <command...>   append a step WITHOUT executing it (robot
                       does not move; creates the sequence if new)
  seq list             list saved sequences (src/data/sequences/, git-ignored)
  seq show <name>      print a sequence's steps
  seq run <name>       run a sequence step by step (ONE y/N up front, then
                       every step is auto-confirmed; stops on first failure)

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


def _auto(state):
    """Confirm override for the cores: auto-yes during 'seq run', else None
    (= the core's own console y/N prompt)."""
    return (lambda _: True) if state.get("auto_yes") else None


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
        return False
    ok, _before, after = mc.set_mode(robot, target)   # console y/N confirm
    if not ok:
        print("[mode] cancelled.")
        return False
    print(f"[mode] now: {cc.format_mode_oneline(after)}")
    return True


def cmd_speed(state, rest):
    if not rest:
        print(f"[speed] {state['speed']}%")
        return True
    try:
        v = int(rest[0])
    except ValueError:
        print(f"[speed] invalid: {rest[0]}")
        return False
    if not 1 <= v <= 100:
        print("[speed] must be 1-100")
        return False
    state["speed"] = v
    print(f"[speed] set to {v}%")
    return True


def cmd_sleep(rest):
    if len(rest) != 1:
        print("usage: sleep <seconds>")
        return False
    try:
        s = float(rest[0])
    except ValueError:
        print(f"[sleep] invalid: {rest[0]}")
        return False
    if not 0 <= s <= 300:
        print("[sleep] must be 0-300 seconds")
        return False
    time.sleep(s)
    return True


def _wrap_pi(a):
    """Wrap an angle to [-pi, pi) — roll/yaw are full-circle, pitch is NOT."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def cmd_movep_rel(robot, state, rest):
    if len(rest) not in (3, 6):
        print("usage: movep rel DX DY DZ [DR DP DY]  (mm, deg; "
              "R P Y omitted = keep current orientation)")
        return False
    try:
        deltas = [float(x) for x in rest]
    except ValueError:
        print(f"[movep] invalid numbers: {' '.join(rest)}")
        return False
    cur, src = sp.read_pose(robot)
    if cur is None:
        print("[movep] cannot read the current pose — a relative move needs "
              "it (parser mismatch?)")
        return False
    target = list(cur)
    for i in range(3):
        target[i] += deltas[i] / 1000.0            # CLI mm -> native m
    if len(deltas) == 6:
        # Roll/yaw wrap across +-180 deg so e.g. -175 + -10 stays valid;
        # pitch must not wrap (its range is [-90, 90] deg).
        target[3] = _wrap_pi(target[3] + math.radians(deltas[3]))
        target[4] += math.radians(deltas[4])
        target[5] = _wrap_pi(target[5] + math.radians(deltas[5]))
    errs = mp.validate_target(target)
    if errs:
        print("[movep] resulting target out of range: " + "; ".join(errs))
        return False
    print(f"[movep] current: {mp._fmt_pose(cur)}   ({src})")
    print(f"[movep] target : {mp._fmt_pose(target)}")
    outcome, status = mp.move_to_pose(robot, target, state["speed"],
                                      state["timeout"], _auto(state))
    print(f"[movep] {_move_msg(outcome, status)}")
    return outcome == "reached"


def cmd_movej_rel(robot, state, rest):
    if len(rest) != mj.JOINT_NUMS:
        print(f"usage: movej rel D1 .. D{mj.JOINT_NUMS}  (deg deltas, "
              f"all {mj.JOINT_NUMS} required; 0 = keep that joint)")
        return False
    try:
        deltas = [math.radians(float(x)) for x in rest]
    except ValueError:
        print(f"[movej] invalid numbers: {' '.join(rest)}")
        return False
    ja, src = sj.read_joints(robot)
    if ja is None:
        print("[movej] cannot read the current joints — a relative move "
              "needs them (parser mismatch?)")
        return False
    cur = list(ja.msg)
    if len(cur) != mj.JOINT_NUMS:
        print(f"[movej] expected {mj.JOINT_NUMS} joints, feedback has "
              f"{len(cur)}")
        return False
    joints = [c + d for c, d in zip(cur, deltas)]
    fmt = lambda vals: "[" + ", ".join(  # noqa: E731
        f"{math.degrees(v):+.1f}" for v in vals) + "] deg"
    print(f"[movej] current: {fmt(cur)}   ({src})")
    print(f"[movej] target : {fmt(joints)}")
    outcome, status = mj.move_to_joints(robot, joints, state["speed"],
                                        state["timeout"], _auto(state))
    print(f"[movej] {_move_msg(outcome, status)}")
    return outcome == "reached"


def cmd_movep(robot, state, rest):
    if rest and rest[0].lower() == "rel":
        return cmd_movep_rel(robot, state, rest[1:])
    if len(rest) != 6:
        print("usage: movep X Y Z ROLL PITCH YAW  (mm, deg)")
        return False
    try:
        vals = [float(x) for x in rest]
    except ValueError:
        print(f"[movep] invalid numbers: {' '.join(rest)}")
        return False
    target = mp.build_target(vals, si=False)      # mm/deg -> native m/rad
    errs = mp.validate_target(target)
    if errs:
        print("[movep] out of range: " + "; ".join(errs))
        return False
    outcome, status = mp.move_to_pose(robot, target, state["speed"],
                                      state["timeout"], _auto(state))
    print(f"[movep] {_move_msg(outcome, status)}")
    return outcome == "reached"


def cmd_movej(robot, state, rest):
    if rest and rest[0].lower() == "rel":
        return cmd_movej_rel(robot, state, rest[1:])
    if len(rest) != mj.JOINT_NUMS:
        print(f"usage: movej J1 .. J{mj.JOINT_NUMS}  (deg)")
        return False
    try:
        joints = [math.radians(float(x)) for x in rest]
    except ValueError:
        print(f"[movej] invalid numbers: {' '.join(rest)}")
        return False
    outcome, status = mj.move_to_joints(robot, joints, state["speed"],
                                        state["timeout"], _auto(state))
    print(f"[movej] {_move_msg(outcome, status)}")
    return outcome == "reached"


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
            return False
    effector = state["effector"]

    if not rest:                      # bare 'gripper' = read-only check
        for line in cg.format_report(cg.check_gripper(effector)):
            print(line)
        return True

    sub = rest[0].lower()
    if sub == "info":
        for line in gi.format_info(gi.gripper_info(robot, effector)):
            print(line)
        return True
    if sub == "reset":
        effector.reset_gripper()      # disable + clear errors
        print("[gripper] reset sent (disabled + errors cleared).")
        return True
    if sub == "calib":
        rng = None
        if len(rest) > 1:
            if rest[1] not in ("70", "100"):
                print("usage: gripper calib [70|100]  (stroke in mm)")
                return False
            rng = int(rest[1]) / 1000.0
        outcome, param_ok = mg.calibrate(robot, effector, rng)  # y/N + Enter
        mg.report_calib(outcome, param_ok, tag="gripper")
        return outcome == "calibrated"
    if sub in ("open", "close"):
        spec, force_arg = sub, rest[1:]
    elif sub == "to":
        if len(rest) < 2:
            print("usage: gripper to <val> [forceN]")
            return False
        spec, force_arg = rest[1], rest[2:]
    else:
        print("usage: gripper [info|calib|open|close|to <val>|reset] [forceN]")
        return False
    force = 1.0
    if force_arg:
        try:
            force = float(force_arg[0])
        except ValueError:
            print(f"[gripper] invalid force: {force_arg[0]}")
            return False

    mode, chk = mg.read_mode(effector)
    if mode is None:
        for line in cg.format_report(chk):
            print(line)
        print("[gripper] refusing to command a gripper that shows no "
              "telemetry.")
        return False
    target, err = mg.resolve_target(effector, mode, spec)
    if err:
        print(f"[gripper] {err}")
        return False
    cur = chk["status"].msg
    print(f"[gripper] mode {mode}: {mg.fmt_value(cur.value, mode)} -> "
          f"{mg.fmt_value(target, mode)} at {force:.2f} N")
    outcome, msg = mg.move_gripper(robot, effector, mode, target, force,
                                   state["timeout"], _auto(state))
    mg.report(outcome, msg, mode, target, state["timeout"], tag="gripper")
    # "holding" = stalled into a grasp — that is a successful gripper action.
    return outcome in ("reached", "holding")


def cmd_goto(robot, state, rest):
    if not rest:
        print("usage: goto <name> [joints|cartesian]")
        return False
    name = rest[0]
    try:
        rec = wp.load_waypoint(name)
    except FileNotFoundError:
        print(f"[goto] no waypoint '{name}' (try 'points')")
        return False
    want = rest[1].lower() if len(rest) > 1 else "auto"
    if want not in ("auto", "joints", "cartesian"):
        print("usage: goto <name> [joints|cartesian]")
        return False
    method, err = gw._choose_method(rec, want)
    if err:
        print(f"[goto] {err}")
        return False
    print(f"[goto] '{name}' via {method}: {wp.format_waypoint(rec)}")
    if method == "joints":
        outcome, status = mj.move_to_joints(robot, wp.to_joints_rad(rec),
                                            state["speed"], state["timeout"],
                                            _auto(state))
    else:
        outcome, status = mp.move_to_pose(robot, wp.to_cartesian_target(rec),
                                          state["speed"], state["timeout"],
                                          _auto(state))
    print(f"[goto] {_move_msg(outcome, status)}")
    return outcome == "reached"


def cmd_seq(robot, state, rest):
    usage = ("usage: seq record <name> | stop | cancel | add <n> <cmd...> | "
             "list | show <n> | run <n>")
    if not rest:
        print(usage)
        return False
    sub = rest[0].lower()

    if sub == "add":                  # append a step WITHOUT executing it
        if len(rest) < 3:
            print("usage: seq add <name> <command...>   e.g. "
                  "seq add pick movep rel 0 0 -80")
            return False
        name, line = rest[1], " ".join(rest[2:])
        err = wp.check_name(name) or sq.check_step(line)
        if err:
            print(f"[seq] {err}")
            return False
        try:
            seq = sq.load_sequence(name)
        except FileNotFoundError:
            seq = sq.make_sequence(name, [line])
            seq["steps"] = []         # appended below; keeps one code path
        seq["steps"].append(line)
        sq.save_sequence(seq)
        print(f"[seq] '{name}' step {len(seq['steps'])}: {line}   "
              "(not executed)")
        return True

    if sub == "record":
        if len(rest) != 2:
            print("usage: seq record <name>")
            return False
        if state.get("seq_rec") is not None:
            print(f"[seq] already capturing '{state['seq_rec']['name']}' — "
                  "'seq stop' or 'seq cancel' first.")
            return False
        err = wp.check_name(rest[1])
        if err:
            print(f"[seq] {err}")
            return False
        state["seq_rec"] = {"name": rest[1], "steps": []}
        print(f"[seq] capturing commands into '{rest[1]}'. Every successful "
              "move/gripper/sleep/speed command you type is appended; "
              "'seq stop' saves, 'seq cancel' discards.")
        return True

    if sub in ("stop", "cancel"):
        cap = state.get("seq_rec")
        if cap is None:
            print("[seq] not capturing (start with 'seq record <name>').")
            return False
        state["seq_rec"] = None
        if sub == "cancel" or not cap["steps"]:
            print(f"[seq] discarded '{cap['name']}'"
                  + ("" if cap["steps"] else " (no steps captured)") + ".")
            return sub == "cancel"
        seq = sq.make_sequence(cap["name"], cap["steps"])
        path, existed = sq.save_sequence(seq)
        verb = "overwrote" if existed else "saved"
        print(f"[seq] {verb} '{cap['name']}' ({len(cap['steps'])} steps): "
              f"{path}")
        return True

    if sub == "list":
        names = sq.list_sequences()
        if not names:
            print(f"[seq] none yet ({sq.SEQUENCES_DIR})")
            return True
        print(f"[seq] {len(names)} sequence(s):")
        for name in names:
            print(f"  {sq.format_sequence(sq.load_sequence(name))}")
        return True

    if sub == "show":
        if len(rest) != 2:
            print("usage: seq show <name>")
            return False
        try:
            seq = sq.load_sequence(rest[1])
        except FileNotFoundError:
            print(f"[seq] no sequence '{rest[1]}' (try 'seq list')")
            return False
        print(f"[seq] {sq.format_sequence(seq)}")
        for i, step in enumerate(seq.get("steps") or [], start=1):
            print(f"  {i:2d}. {step}")
        return True

    if sub == "run":
        if len(rest) != 2:
            print("usage: seq run <name>")
            return False
        if state.get("seq_rec") is not None:
            print("[seq] cannot run a sequence while capturing one — "
                  "'seq stop' or 'seq cancel' first.")
            return False
        try:
            seq = sq.load_sequence(rest[1])
        except FileNotFoundError:
            print(f"[seq] no sequence '{rest[1]}' (try 'seq list')")
            return False
        steps = seq.get("steps") or []
        bad = [f"  step {i}: '{s}' — {sq.check_step(s)}"
               for i, s in enumerate(steps, start=1) if sq.check_step(s)]
        if not steps or bad:
            print(f"[seq] '{rest[1]}' is not runnable:")
            for line in bad or ["  no steps"]:
                print(line)
            return False
        print(f"[seq] '{rest[1]}' — {len(steps)} steps:")
        for i, step in enumerate(steps, start=1):
            print(f"  {i:2d}. {step}")
        if not mp._console_confirm(
                "[seq] !! ALL steps will run SEQUENTIALLY with every move "
                "AUTO-CONFIRMED (the arm and gripper move without further "
                "prompts). It stops at the first failed step.\n"
                "[seq]   !! 모든 스텝이 추가 확인 없이 순차 실행됩니다. "
                "실패한 스텝에서 멈춥니다. 주변을 비우고 비상정지 준비하세요."):
            print("[seq] cancelled. 취소되었습니다.")
            return False
        state["auto_yes"] = True
        try:
            for i, step in enumerate(steps, start=1):
                print(f"[seq] --- step {i}/{len(steps)}: {step}")
                if not dispatch(robot, state, step):
                    print(f"[seq] step {i} FAILED — sequence stopped. "
                          "(arm holds its pose)")
                    return False
        finally:
            state["auto_yes"] = False
        print(f"[seq] sequence '{rest[1]}' complete ({len(steps)} steps).")
        return True

    print(usage)
    return False


def dispatch(robot, state, line):
    """Execute one command line (from the prompt or a sequence step).

    Returns True on success, False on failure — 'seq run' stops on False.
    When a sequence capture is active, successful replayable commands are
    appended to it.
    """
    tokens = line.split()
    head, rest = tokens[0].lower(), tokens[1:]
    ok = True

    if head == "help":
        print(HELP)
    elif head == "status":
        cmd_status(robot)
    elif head == "pose":
        cmd_pose(robot)
    elif head == "joints":
        if rest and rest[0].lower() == "rel":   # alias for 'movej rel'
            ok = cmd_movej_rel(robot, state, rest[1:])
        else:
            cmd_joints(robot)
    elif head == "mode":
        ok = cmd_mode(robot, rest)
    elif head == "teach":
        done, _b, after = cm.set_ctrl_mode(robot, "teach")     # console y/N
        ok = cm.report("teach", done, after) == 0
    elif head == "canmode":
        done, _b, after = cm.set_ctrl_mode(robot, "can")       # console y/N
        ok = cm.report("can", done, after) == 0
    elif head == "speed":
        ok = cmd_speed(state, rest)
    elif head == "sleep":
        ok = cmd_sleep(rest)
    elif head == "movep":
        ok = cmd_movep(robot, state, rest)
    elif head == "movej":
        ok = cmd_movej(robot, state, rest)
    elif head == "save":
        cmd_save(robot, rest)
    elif head == "points":
        cmd_points()
    elif head == "goto":
        ok = cmd_goto(robot, state, rest)
    elif head == "gripper":
        ok = cmd_gripper(robot, state, rest)
    elif head == "seq":
        ok = cmd_seq(robot, state, rest)
    else:
        print(f"unknown command: {head}  (try 'help')")
        ok = False

    cap = state.get("seq_rec")
    if ok and cap is not None and sq.is_recordable(line):
        cap["steps"].append(line)
        print(f"[seq] + captured as step {len(cap['steps'])} of "
              f"'{cap['name']}'")
    return ok


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
        if line.split()[0].lower() in ("quit", "exit", "q"):
            break
        dispatch(robot, state, line)

    if state.get("seq_rec") is not None:
        print(f"[seq] capture '{state['seq_rec']['name']}' discarded (not "
              "saved before quit).")
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
             "effector": None, "auto_yes": False, "seq_rec": None}
    try:
        repl(robot, state)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
