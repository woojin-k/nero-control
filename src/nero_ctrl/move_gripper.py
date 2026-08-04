#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Open / close / move the AGX gripper. ACTUATES.

!! THE GRIPPER MOVES, and the ARM MOTORS ARE ENABLED first. The gripper motor
is powered through the arm's 0x471 motor-enable (verified on the real Nero:
0x159 alone flips the gripper's logical enable bit but the jaws never drive
until the motors are enabled — and this firmware's enable is all-or-nothing,
arm joints included). Enabling makes the arm LOCK and HOLD its current pose;
it does not move it. Keep fingers and objects clear of the jaws. It always
asks for y/N confirmation unless --yes is given.

The gripper runs in one of two firmware modes, visible in its 0x2A8 telemetry:
  width mode : value is the jaw opening in metres  (CLI takes mm)
  angle mode : value is the jaw angle in degrees   (CLI takes deg)
The mode is detected from live telemetry first (same signal check_gripper.py
uses), so the target units always match what the gripper actually is in. If no
real telemetry is seen, nothing is sent — a disconnected/unpowered gripper
silently ignores commands, so we refuse instead of pretending.

Closing on an object stalls before the target on purpose (force-limited grasp);
that is reported as "holding", not as an error.

A freshly mounted / power-cycled gripper is NOT homed (0x2A8 bit7 = 0): it
enables fine but ignores position commands. Run --calib once — the jaws are
closed by hand and that position is stored as zero (0xAE + 0x476 ack).

Usage:
    ./.venv/bin/python src/nero_ctrl/move_gripper.py --calib           # first!
    ./.venv/bin/python src/nero_ctrl/move_gripper.py --open
    ./.venv/bin/python src/nero_ctrl/move_gripper.py --close
    ./.venv/bin/python src/nero_ctrl/move_gripper.py --to 30 --force 2
    ./.venv/bin/python src/nero_ctrl/move_gripper.py --close --force 5 --yes

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse
import time

# Sibling modules: shared connection, console confirm, gripper detection.
import connect_check as cc
import move_to_pose as mp
import check_gripper as cg

# Convergence tolerances in the gripper's native units.
TOL_WIDTH_M = 0.002     # 2 mm on a 70/100 mm stroke
TOL_ANGLE_DEG = 2.0
# Fallback full-open width if the teaching-pendant param is unavailable.
DEFAULT_MAX_WIDTH_M = 0.07
# 0x159 force is an int16 in mN, so ~32 N is the wire-format ceiling.
MAX_FORCE_N = 30.0

WARN_GRIP = ("[grip] !! THE GRIPPER WILL MOVE, and the ARM MOTORS are ENABLED "
             "first (the arm locks and holds its pose — it does not move). "
             "Keep fingers and objects clear of the jaws.\n"
             "[grip]   !! 그리퍼가 움직입니다. 팔 모터도 먼저 켜져 현재 자세로 "
             "고정됩니다(팔은 움직이지 않음). 손가락과 물건을 그리퍼에서 "
             "치우세요.")

WARN_CALIB = ("[grip] Zero-point calibration: the gripper will be DISABLED, "
              "then you close the jaws fully BY HAND, and that position "
              "becomes zero.\n"
              "[grip]   영점 캘리브레이션: 그리퍼가 비활성화되면 손으로 조를 "
              "완전히 닫은 뒤 그 위치가 영점이 됩니다.")


def read_mode(effector, window=0.6):
    """Detect the gripper and its mode from live 0x2A8 telemetry. Read-only.

    Returns ``(mode, check_result)`` where mode is "width"/"angle", or
    ``(None, check_result)`` when no real gripper telemetry is present (same
    criterion as check_gripper.py: all-zero frames are the arm's placeholder).
    """
    res = cg.check_gripper(effector, window)
    if not res["detected"]:
        return None, res
    return res["status"].msg.mode, res


def max_width_m(effector):
    """Full-open width in metres from the teaching-pendant param (or default)."""
    try:
        param = effector.get_gripper_teaching_pendant_param()
    except Exception:  # noqa: BLE001
        param = None
    if param is not None and param.msg.max_range_config:
        return param.msg.max_range_config
    return DEFAULT_MAX_WIDTH_M


def resolve_target(effector, mode, spec):
    """Turn a CLI target spec into native units for `mode`.

    `spec` is "open", "close", or a number string — mm in width mode, deg in
    angle mode. Returns ``(target, err)``; on error target is None.
    """
    if spec == "close":
        return 0.0, None
    if spec == "open":
        if mode == "angle":
            return None, ("gripper is in ANGLE mode — full-open angle is not "
                          "known; give an explicit angle (deg)")
        return max_width_m(effector), None
    try:
        v = float(spec)
    except (TypeError, ValueError):
        return None, f"invalid target: {spec}"
    if mode == "angle":
        return v, None
    t = v / 1000.0                      # CLI mm -> native m
    limit = max_width_m(effector)
    if not 0.0 <= t <= limit + 0.005:
        return None, (f"width {v:.1f} mm out of range "
                      f"[0, {limit * 1000:.0f}] mm")
    return t, None


def fmt_value(value, mode):
    """Human units for a native gripper value: mm (width) / deg (angle)."""
    if mode == "angle":
        return f"{value:+.1f} deg"
    return f"{value * 1000:.1f} mm"


def _watch_gripper(effector, mode, target, timeout=10.0, poll=0.05,
                   resend=None):
    """Watch 0x2A8 until the jaws reach `target`, stall, or `timeout`.

    `resend()` re-issues the 0x159 command every poll — the vendor demo
    streams it continuously, and this gripper ignores a single frame (the
    first one only flips the enable bit; movement needs repeats).

    Returns (outcome, last_status_msg):
      "reached"  within tolerance of the target
      "holding"  jaws stopped short of the target (force-limited grasp or an
                 obstruction) — the commanded force keeps being applied
      "timeout"  still moving / no telemetry when time ran out
    """
    tol = TOL_ANGLE_DEG if mode == "angle" else TOL_WIDTH_M
    start = time.time()
    end = start + timeout
    last_msg = None
    last_val = None
    still = 0
    while time.time() < end:
        if resend is not None:
            resend()
        gs = effector.get_gripper_status()
        if gs is not None:
            m = gs.msg
            last_msg = m
            if abs(m.value - target) <= tol:
                return "reached", m
            # Stalled = value stopped changing well short of the target.
            if last_val is not None and abs(m.value - last_val) < tol / 20:
                still += 1
                if still >= 12 and time.time() - start > 1.0:
                    return "holding", m
            else:
                still = 0
            last_val = m.value
        time.sleep(poll)
    return "timeout", last_msg


def move_gripper(robot, effector, mode, target, force=1.0, timeout=10.0,
                 confirm=None):
    """Drive the gripper to `target` (native units for `mode`) on a connected
    robot. Reusable core for a CLI/GUI.

    `confirm(message) -> bool` gates the command (default: console y/N; a GUI
    can inject a popup; ``lambda _: True`` skips). Motor enable includes the
    ARM joints (all-or-nothing on this firmware): the arm locks/holds its
    pose. Motors are left enabled afterwards. Returns
    ``(outcome, last_status_msg)`` with outcome one of: "invalid", "leader",
    "cancelled", "enable_failed", "reached", "holding", "timeout".
    """
    if not 0.0 < force <= MAX_FORCE_N:
        return "invalid", None
    # In LEADER mode the teach pendant owns the gripper — don't fight it.
    if cc.read_modes(robot).get("linkage_role") == "LEADER":
        return "leader", None
    confirm = confirm or mp._console_confirm
    if not confirm(WARN_GRIP):
        return "cancelled", None
    ensure_can_ctrl(robot)     # teaching-input mode can swallow gripper cmds
    # The gripper motor is powered via the arm's 0x471 enable (piper demos'
    # EnablePiper step) — without it the jaws never drive.
    if not mp.enable_with_timeout(robot):
        return "enable_failed", None
    if mode == "angle":
        send = lambda: effector.move_gripper_deg(target, force)  # noqa: E731
    else:
        send = lambda: effector.move_gripper_m(target, force)    # noqa: E731
    send()
    return _watch_gripper(effector, mode, target, timeout, resend=send)


def ensure_can_ctrl(robot, tag="grip"):
    """Pull the arm out of LINKAGE_TEACHING_INPUT_MODE into CAN command mode.

    In linkage-teaching-input mode the controller ignores gripper config/zero
    frames (0x47D/0x476 get no ACK). set_motion_mode() sends the 0x151 mode
    frame with ctrl_mode=CAN_CTRL — it does NOT enable motors and nothing
    moves. No-op when the arm is already out of that mode.
    """
    if cc.read_modes(robot).get("ctrl_mode") != "LINKAGE_TEACHING_INPUT_MODE":
        return
    print(f"[{tag}] arm is in LINKAGE_TEACHING_INPUT_MODE — switching "
          "ctrl_mode to CAN_CTRL (no motion, motors untouched).")
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
    time.sleep(0.5)
    now = cc.read_modes(robot).get("ctrl_mode")
    print(f"[{tag}] ctrl_mode now: {now or '?'}")


def _console_pause(message):
    """Default `interact` for calibrate(): wait for Enter, False on Ctrl-C."""
    try:
        input(message)
        return True
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def calibrate(robot, effector, max_range_m=None, confirm=None, interact=None):
    """Zero-point calibration: disable, let the user close the jaws by hand,
    then set that position as zero (0xAE). Reusable core for a CLI/GUI.

    A gripper that is not homed (0x2A8 bit7 = 0) ignores position commands, so
    this must run once after mounting/power-cycling the gripper. `max_range_m`
    (0.07 or 0.1) optionally stores the gripper's physical stroke first.
    `interact(message) -> bool` pauses while the user closes the jaws.

    Returns ``(outcome, param_ok)``. outcome is one of "leader", "cancelled",
    "calibrated", "failed". param_ok is None when no max_range_m was given,
    else whether the param store was ACKed — a failed store is NOT fatal (it
    only affects the full-open target), so calibration continues without it.
    """
    if cc.read_modes(robot).get("linkage_role") == "LEADER":
        return "leader", None
    confirm = confirm or mp._console_confirm
    if not confirm(WARN_CALIB):
        return "cancelled", None
    ensure_can_ctrl(robot)     # teaching-input mode swallows config/zero acks
    effector.disable_gripper()
    param_ok = None
    if max_range_m:
        try:
            param_ok = effector.set_gripper_teaching_pendant_param(
                max_range_config=max_range_m)
        except Exception:  # noqa: BLE001
            param_ok = False
    interact = interact or _console_pause
    if not interact("[grip] close the jaws fully by hand, then press Enter. "
                    "손으로 조를 완전히 닫고 Enter: "):
        return "cancelled", param_ok
    ok = effector.calibrate_gripper(timeout=2.0)
    # This firmware never ACKs (0x476) nor raises the homed bit, but the zero
    # does apply: the value telemetry re-references. The jaws are closed by
    # hand right now, so a value near 0 is the functional proof.
    zeroed = False
    for _ in range(10):
        time.sleep(0.15)
        gs = effector.get_gripper_status()
        if gs is None:
            continue
        m = gs.msg
        if m.foc_status.homing_status:
            zeroed = True
            break
        tol = TOL_ANGLE_DEG if m.mode == "angle" else TOL_WIDTH_M
        if abs(m.value) <= tol * 2:
            zeroed = True
            break
    return ("calibrated" if (ok or zeroed) else "failed"), param_ok


def report_calib(outcome, param_ok=None, tag="grip"):
    """Print a calibrate() outcome and return the process exit code."""
    if outcome == "leader":
        print(f"[{tag}] arm is in LEADER mode — run set_follower_mode.py "
              "first.")
        return 2
    if outcome == "cancelled":
        print(f"[{tag}] cancelled. 취소되었습니다.")
        return 3
    if param_ok is False:
        print(f"[{tag}] warning: max-range param not stored (no ACK) — "
              "continuing anyway; only the full-open target is affected. "
              "(The arm's ctrl_mode may be blocking config frames.)")
    if outcome == "calibrated":
        print(f"[{tag}] zero point set — gripper is homed. Position commands "
              "will work now (jaws start from closed=0).")
        return 0
    print(f"[{tag}] zero-set sent but NOT confirmed (no ACK, homed bit still "
          "0) — this firmware may store it silently. Verify by hand: move the "
          "jaws and check that the value telemetry reads ~0 when closed.")
    print(f"[{tag}] then test a move; if the jaws still don't drive, "
          "power-cycle the gripper (전원 재투입) and calibrate again.")
    return 6


def _not_homed_hint(msg, tag):
    if msg is not None and not msg.foc_status.homing_status:
        print(f"[{tag}] note: homed bit is 0. If the value telemetry does not "
              "read ~0 with the jaws closed, the zero is not set — run "
              "'--calib' / 'gripper calib'. (Some firmware never raises the "
              "bit even when zeroed.)")


def report(outcome, msg, mode, target, timeout, tag="grip"):
    """Print an outcome message and return the process exit code."""
    if outcome == "invalid":
        print(f"[{tag}] invalid force (must be 0 < N <= {MAX_FORCE_N:.0f}).")
        return 2
    if outcome == "leader":
        print(f"[{tag}] arm is in LEADER mode — the teach pendant owns the "
              "gripper. Run set_follower_mode.py first.")
        return 2
    if outcome == "cancelled":
        print(f"[{tag}] cancelled. 취소되었습니다.")
        return 3
    if outcome == "enable_failed":
        print(f"[{tag}] failed to enable the motors (incl. gripper) within "
              "timeout — aborting.")
        return 4
    if outcome == "reached":
        print(f"[{tag}] reached {fmt_value(msg.value, mode)} "
              f"(force={msg.force:.2f} N).")
        return 0
    if outcome == "holding":
        print(f"[{tag}] stopped at {fmt_value(msg.value, mode)} before the "
              f"target {fmt_value(target, mode)} — holding with "
              f"force={msg.force:.2f} N (object grasped or jaws obstructed).")
        _not_homed_hint(msg, tag)
        return 0
    got = fmt_value(msg.value, mode) if msg is not None else "no telemetry"
    print(f"[{tag}] not settled within {timeout}s (at {got}). Check the "
          "gripper; it may need reset/calibration.")
    _not_homed_hint(msg, tag)
    return 6


def _parse_args():
    p = argparse.ArgumentParser(
        description="Open / close / move the AGX gripper. ACTUATES the "
                    "gripper (only) — asks before moving.")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="v112",
                   choices=["auto", "default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default: v112)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--open", action="store_true",
                   help="open fully (teaching-pendant max range)")
    g.add_argument("--close", action="store_true",
                   help="close fully (0) — stalls into a grasp on an object")
    g.add_argument("--to", type=float, metavar="VALUE",
                   help="target: mm in width mode / deg in angle mode")
    g.add_argument("--calib", action="store_true",
                   help="zero-point calibration (close jaws by hand when "
                        "asked) — required once before position commands work")
    g.add_argument("--reset", action="store_true",
                   help="disable + clear gripper errors")
    p.add_argument("--range", type=int, choices=(70, 100), default=None,
                   help="with --calib: store the gripper's physical stroke "
                        "in mm (70 or 100)")
    p.add_argument("--force", type=float, default=1.0,
                   help=f"grip force in N, 0-{MAX_FORCE_N:.0f} (default: 1.0)")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="seconds to wait for the jaws to settle (default: 10)")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
    return p.parse_args()


def main():
    args = _parse_args()
    spec = "open" if args.open else "close" if args.close else str(args.to)

    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[grip] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    code = 0
    try:
        effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        mode, chk = read_mode(effector)
        if mode is None:
            for line in cg.format_report(chk):
                print(line)
            print("[grip] refusing to command a gripper that shows no "
                  "telemetry.")
            raise SystemExit(2)

        if args.reset:
            effector.reset_gripper()      # disable + clear errors
            time.sleep(0.3)
            gs = effector.get_gripper_status()
            if gs is not None:
                m = gs.msg
                print(f"[grip] reset sent — value={fmt_value(m.value, mode)}  "
                      f"enabled={m.foc_status.driver_enable_status}  "
                      f"homed={m.foc_status.homing_status}")
            raise SystemExit(0)

        if args.calib:
            confirm = (lambda _: True) if args.yes else None
            rng = args.range / 1000.0 if args.range else None
            outcome, param_ok = calibrate(robot, effector, rng, confirm)
            code = report_calib(outcome, param_ok)
            raise SystemExit(code)

        target, err = resolve_target(effector, mode, spec)
        if err:
            print(f"[grip] {err}")
            raise SystemExit(2)

        cur = chk["status"].msg
        print(f"[grip] mode   : {mode}   current: {fmt_value(cur.value, mode)}"
              f"   force: {cur.force:.2f} N")
        print(f"[grip] target : {fmt_value(target, mode)}   "
              f"force: {args.force:.2f} N")

        confirm = (lambda _: True) if args.yes else None
        outcome, msg = move_gripper(robot, effector, mode, target,
                                    args.force, args.timeout, confirm)
        code = report(outcome, msg, mode, target, args.timeout)
    finally:
        robot.disconnect()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
