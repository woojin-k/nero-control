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

--calib fixes only the CLOSED end. The OPEN end comes from the teaching-pendant
param, i.e. the gripper's FACTORY stroke, which is wrong for a gripper with
modified jaw tips: thicker tips meet early, so the new zero sits part-way up the
factory scale, 'open' aims past the physical end stop, and the stalled resend
trips the driver (the gripper then drops off the CAN bus). --teach solves that
without measuring anything: with the motor disabled you set BOTH ends by hand and
the encoder readings are stored (see gripper_range.py). Prefer --teach.

Usage:
    ./.venv/bin/python src/nero_ctrl/move_gripper.py --teach           # first!
    ./.venv/bin/python src/nero_ctrl/move_gripper.py --calib     # zero end only
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
import gripper_range as gr

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


def read_mode(effector, window=0.6, latch=None):
    """Detect the gripper and its mode from live 0x2A8 telemetry. Read-only.

    Returns ``(mode, check_result)`` where mode is "width"/"angle", or
    ``(None, check_result)`` when no real gripper telemetry is present (same
    criterion as check_gripper.py: all-zero frames are the arm's placeholder).
    `latch` is passed straight through to check_gripper(), so a gripper parked
    on its zero point still resolves to its remembered mode instead of reading
    as disconnected.
    """
    res = cg.check_gripper(effector, window, latch=latch)
    if not res["detected"]:
        return None, res
    # On a latched (all-zero) window the placeholder frame's decoded mode is
    # meaningless — check_gripper hands back the remembered one.
    return res.get("mode") or res["status"].msg.mode, res


def max_width_m(effector):
    """Full-open width in metres from the teaching-pendant param (or default)."""
    try:
        param = effector.get_gripper_teaching_pendant_param()
    except Exception:  # noqa: BLE001
        param = None
    if param is not None and param.msg.max_range_config:
        return param.msg.max_range_config
    return DEFAULT_MAX_WIDTH_M


def resolve_target(effector, mode, spec, rec=None):
    """Turn a CLI target spec into native units for `mode`.

    `spec` is "open", "close", or a number string — mm in width mode, deg in
    angle mode. `rec` is a hand-taught range record (gripper_range); when None
    it is loaded for `mode`, and when there is none the factory teaching-pendant
    stroke is used as before. A taught record is authoritative: it defines both
    the 'open'/'close' targets and the accepted numeric range, which is what
    keeps a modified-tip gripper off its end stops.

    Returns ``(target, err)``; on error target is None.
    """
    if rec is None:
        rec = gr.load_range(mode=mode)

    if spec == "close":
        return (gr.close_target(rec) if rec else 0.0), None
    if spec == "open":
        if rec:
            return gr.open_target(rec), None
        if mode == "angle":
            return None, ("gripper is in ANGLE mode and no range is taught — "
                          "full-open angle is not known; run 'gripper calib' "
                          "or give an explicit angle (deg)")
        return max_width_m(effector), None

    try:
        v = float(spec)
    except (TypeError, ValueError):
        return None, f"invalid target: {spec}"
    t = v if mode == "angle" else v / 1000.0    # CLI mm -> native m
    if rec:
        lo, hi = gr.limits(rec)
        tol = TOL_ANGLE_DEG if mode == "angle" else TOL_WIDTH_M
        if not lo - tol <= t <= hi + tol:
            return None, (f"target {gr.fmt(t, mode)} outside the taught range "
                          f"[{gr.fmt(lo, mode)}, {gr.fmt(hi, mode)}] — "
                          "re-teach with 'gripper calib' if the jaws changed")
        return t, None
    if mode == "angle":
        return t, None
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

    This fixes the CLOSED end only; the open end stays whatever the pendant
    param claims. For anything but stock jaws prefer teach_range(), which sets
    both ends and cannot aim past the real travel.

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


WARN_TEACH = ("[grip] Range teach: the gripper will be DISABLED, then you set "
              "BOTH ends by hand — fully closed first (that becomes zero), "
              "then fully open. Nothing is driven by the motor.\n"
              "[grip]   범위 티칭: 그리퍼가 비활성화되면 손으로 (1) 완전히 닫은 "
              "위치, (2) 완전히 열린 위치를 차례로 잡아 주세요. 모터로 "
              "움직이지 않습니다.")

PROMPT_CLOSED = ("[grip] 1/2 — close the jaws FULLY by hand, then press Enter. "
                 "손으로 조를 완전히 닫고 Enter: ")
PROMPT_OPEN = ("[grip] 2/2 — open the jaws FULLY by hand, then press Enter. "
               "손으로 조를 완전히 열고 Enter: ")


def _settle_value(effector, mode, tries=15, poll=0.08):
    """Read the jaw value once telemetry stops changing. Read-only.

    Returns the settled native value, or None if no frame ever arrived. Waiting
    for it to stop moving matters here: the user has just let go of the jaws.
    """
    tol = TOL_ANGLE_DEG if mode == "angle" else TOL_WIDTH_M
    last = None
    stable = 0
    for _ in range(tries):
        time.sleep(poll)
        gs = effector.get_gripper_status()
        if gs is None:
            continue
        v = gs.msg.value
        if last is not None and abs(v - last) < tol / 20:
            stable += 1
            if stable >= 3:
                return v
        else:
            stable = 0
        last = v
    return last


def teach_range(robot, effector, mode, confirm=None, interact=None, note=""):
    """Teach the gripper's real travel by hand: closed end, then open end.

    Built for jaws that cannot reach the factory scale — modified/thicker tips.
    The motor stays DISABLED throughout, so nothing can be driven into an end
    stop; the human defines both limits and only the encoder is read.

    Step 1 doubles as the zero-point calibration (0xAE) the firmware needs
    before it will accept position commands at all, so this replaces
    calibrate(). The closed reading is recorded as measured rather than assumed
    to be 0 — on firmware that silently ignores the zero-set, the taught pair is
    still self-consistent and the targets still land in the right place.

    `confirm(msg) -> bool` gates the whole thing; `interact(msg) -> bool` pauses
    for each end. Returns ``(outcome, rec, param_ok)`` with outcome one of
    "leader", "cancelled", "no_telemetry", "no_span", "taught". `rec` is the
    gripper_range record (returned even for "no_span", unsaved, so the caller
    can show what was read). `param_ok` is whether the measured stroke was
    stored to the teaching-pendant param (None when not attempted) — a failed
    store is not fatal, the local record is what the CLI actually uses.
    """
    if cc.read_modes(robot).get("linkage_role") == "LEADER":
        return "leader", None, None
    confirm = confirm or mp._console_confirm
    if not confirm(WARN_TEACH):
        return "cancelled", None, None
    ensure_can_ctrl(robot)     # teaching-input mode swallows config/zero acks
    effector.disable_gripper()
    interact = interact or _console_pause

    if not interact(PROMPT_CLOSED):
        return "cancelled", None, None
    effector.calibrate_gripper(timeout=2.0)   # this position becomes zero
    # calibrate_gripper may leave the driver enabled; disable again so the jaws
    # stay free for step 2 instead of fighting the hand that opens them.
    effector.disable_gripper()
    closed = _settle_value(effector, mode)
    if closed is None:
        return "no_telemetry", None, None

    if not interact(PROMPT_OPEN):
        return "cancelled", None, None
    opened = _settle_value(effector, mode)
    if opened is None:
        return "no_telemetry", None, None

    rec = gr.make_range(mode, closed, opened, note=note)
    if gr.span(rec) < gr.min_span(mode):
        return "no_span", rec, None
    gr.save_range(rec)

    param_ok = None
    if mode == "width":
        # Keep the pendant's stroke in step with reality too, so any code path
        # still falling back to max_width_m() cannot aim past the end stop.
        try:
            param_ok = effector.set_gripper_teaching_pendant_param(
                max_range_config=round(gr.span(rec), 4))
        except Exception:  # noqa: BLE001
            param_ok = False
    return "taught", rec, param_ok


def report_teach(outcome, rec, param_ok=None, tag="grip"):
    """Print a teach_range() outcome and return the process exit code."""
    if outcome == "leader":
        print(f"[{tag}] arm is in LEADER mode — run set_follower_mode.py "
              "first.")
        return 2
    if outcome == "cancelled":
        print(f"[{tag}] cancelled — nothing stored. 취소되었습니다.")
        return 3
    if outcome == "no_telemetry":
        print(f"[{tag}] no gripper telemetry while teaching — check the "
              "gripper's power/CAN link and retry.")
        return 6
    if outcome == "no_span":
        mode = rec.get("mode", "width")
        print(f"[{tag}] the two ends read almost the same "
              f"({gr.fmt(rec['closed'], mode)} -> {gr.fmt(rec['open'], mode)}) "
              f"— nothing stored. Did the jaws actually move between the two "
              "prompts? 두 위치가 거의 같습니다 — 저장하지 않았습니다.")
        return 6
    if param_ok is False:
        print(f"[{tag}] warning: measured stroke not stored to the pendant "
              "param (no ACK) — harmless, the taught range file is what is "
              "used.")
    for line in gr.format_range(rec, tag=tag):
        print(line)
    print(f"[{tag}] taught and saved to {gr.RANGE_PATH}")
    print(f"[{tag}] 'open'/'close' now aim at the taught ends, and targets "
          "outside them are refused — no more driving into an end stop.")
    return 0


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
                   help="open fully (taught open end, else pendant max range)")
    g.add_argument("--close", action="store_true",
                   help="close fully (taught closed end, else 0) — stalls into "
                        "a grasp on an object")
    g.add_argument("--to", type=float, metavar="VALUE",
                   help="target: mm in width mode / deg in angle mode")
    g.add_argument("--teach", action="store_true",
                   help="teach the real travel by hand: closed end (also sets "
                        "zero), then open end — use this for modified jaws")
    g.add_argument("--calib", action="store_true",
                   help="zero-point calibration only (close jaws by hand when "
                        "asked); --teach does this and the open end too")
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

        if args.teach:
            confirm = (lambda _: True) if args.yes else None
            outcome, rec, param_ok = teach_range(robot, effector, mode, confirm)
            raise SystemExit(report_teach(outcome, rec, param_ok))

        if args.calib:
            confirm = (lambda _: True) if args.yes else None
            rng = args.range / 1000.0 if args.range else None
            outcome, param_ok = calibrate(robot, effector, rng, confirm)
            code = report_calib(outcome, param_ok)
            raise SystemExit(code)

        rec = gr.load_range(mode=mode)
        target, err = resolve_target(effector, mode, spec, rec)
        if err:
            print(f"[grip] {err}")
            raise SystemExit(2)

        cur = chk["status"].msg
        print(f"[grip] mode   : {mode}   current: {fmt_value(cur.value, mode)}"
              f"   force: {cur.force:.2f} N")
        if rec:
            print(f"[grip] range  : taught closed={fmt_value(rec['closed'], mode)}"
                  f"  open={fmt_value(rec['open'], mode)}")
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
