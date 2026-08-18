#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check whether an AGX gripper is actually connected/recognized. Read-only.

Why this is not trivial: the arm broadcasts the gripper-status frame (CAN 0x2A8)
at ~140 Hz with an ALL-ZERO payload even when NO gripper is attached. So the
naive signals lie:
  * effector.is_ok() / get_fps() are True/>0 just because those zero frames
    arrive — they do NOT prove a gripper is present.
  * get_gripper_teaching_pendant_param() returns SDK default values (e.g.
    100 / 0.1 / 1) even with no gripper — also useless for detection.

The reliable signal is real telemetry: a connected + powered gripper reports
NON-ZERO data on 0x2A8 (position, force, or status_code bits), whereas the
no-gripper placeholder stays all-zero. So: detected == any non-zero 0x2A8
telemetry seen within the sample window.

Caveat: an all-zero result means "no gripper OR a gripper that is attached but
unpowered/off" — an unpowered gripper reports nothing either way. A powered,
functional gripper always reports non-zero telemetry.

One real gripper DOES read all-zero, though: a disabled gripper parked exactly
at its zero point (value 0, no force, no fault bits) is bit-for-bit identical to
the placeholder frame. That is the normal state right after a zero-point
calibration, so `detected` would flip to False on a perfectly healthy gripper.
For that case a caller may pass a `latch` dict (e.g. the CLI's session state):
once real telemetry has been seen through that latch, a later all-zero window is
reported as detected-by-latch instead of as a disconnect.

Reads only; never enables the gripper or moves anything.

Usage:
    ./.venv/bin/python src/nero_ctrl/check_gripper.py --can can0

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse
import time

import connect_check as cc

GRIPPER_STATUS_ID = 0x2A8  # arm broadcasts an all-zero frame here when no gripper


def _status_is_real(msg):
    """True if a decoded gripper-status msg carries non-zero telemetry."""
    sc = getattr(msg, "status_code", 0) or 0
    return bool(msg.value) or bool(msg.force) or bool(sc)


def check_gripper(effector, window=1.2, poll=0.05, latch=None):
    """Detect the AGX gripper from its 0x2A8 telemetry, over `window` seconds.

    Takes an already-initialized gripper `effector` (robot.init_effector(...)).
    Read-only. `latch`, if given, is a mutable dict that remembers real
    telemetry across calls within a session (see the module docstring): keys
    "gripper_seen" and "gripper_mode" are written on a real detection and read
    back when a window comes up all-zero.

    Returns a result dict:
      detected  : bool  -- real telemetry seen, or vouched for by the latch
      latched   : bool  -- detected only because the latch had seen it before
      samples   : int   -- gripper-status reads that returned a frame
      nonzero   : int   -- of those, how many carried non-zero data
      status    : the last non-zero status message (or the last frame), or None
      mode      : "width"/"angle" from real telemetry (latched when all-zero)
      fps       : effector.get_fps()
    """
    samples = nonzero = 0
    last_any = last_real = None
    end = time.time() + window
    while time.time() < end:
        gs = effector.get_gripper_status()
        if gs is not None:
            samples += 1
            last_any = gs
            if _status_is_real(gs.msg):
                nonzero += 1
                last_real = gs
        time.sleep(poll)

    mode = last_real.msg.mode if last_real is not None else None
    if nonzero > 0 and latch is not None:
        latch["gripper_seen"] = True
        latch["gripper_mode"] = mode
    # All-zero, but this session already proved a gripper is there: a disabled
    # gripper sitting on its zero point looks exactly like the placeholder.
    latched = bool(nonzero == 0 and samples > 0 and latch is not None
                   and latch.get("gripper_seen"))
    if latched:
        mode = latch.get("gripper_mode") or mode

    return {
        "detected": nonzero > 0 or latched,
        "latched": latched,
        "samples": samples,
        "nonzero": nonzero,
        "status": last_real or last_any,
        "mode": mode,
        "fps": effector.get_fps(),
    }


def format_report(res):
    """Human-readable report lines for a check_gripper() result."""
    lines = []
    if res["samples"] == 0:
        lines.append("[gripper] NO status frames on 0x2A8 "
                     f"(fps={res['fps']}). No gripper-channel activity — check "
                     "the CAN link / firmware.")
        return lines

    if res.get("latched"):
        lines.append(f"[gripper] DETECTED (latched) — 0x2A8 is all-zero right "
                     f"now across {res['samples']} samples, but real telemetry "
                     "was seen earlier this session.")
        lines.append("[gripper]   That is what a DISABLED gripper parked on its "
                     "zero point looks like (value 0, no force, no faults) — "
                     "normal right after a calibration.")
        lines.append(f"[gripper]   mode={res.get('mode') or '?'}  "
                     f"fps={res['fps']}. Move the jaws by hand to see live "
                     "values again.")
        return lines

    if res["detected"]:
        m = res["status"].msg
        sc = getattr(m, "status_code", 0) or 0
        lines.append(f"[gripper] DETECTED — real telemetry present "
                     f"(fps={res['fps']}, {res['nonzero']}/{res['samples']} "
                     "non-zero).")
        lines.append(f"[gripper]   value={m.value}  force={m.force}  "
                     f"mode={m.mode}  status_code=0x{sc:02X}")
        fs = getattr(m, "foc_status", None)
        if fs is not None:
            faults = [n for n in ("voltage_too_low", "motor_overheating",
                                  "driver_overcurrent", "driver_overheating",
                                  "sensor_status", "driver_error_status")
                      if getattr(fs, n, False)]
            lines.append(f"[gripper]   enabled={getattr(fs, 'driver_enable_status', '?')}"
                         f"  homed={getattr(fs, 'homing_status', '?')}"
                         f"  faults={faults or 'none'}")
    else:
        lines.append(f"[gripper] NOT DETECTED — 0x2A8 is all-zero across "
                     f"{res['samples']} samples (fps={res['fps']}).")
        lines.append("[gripper]   That is the arm's placeholder frame; no real "
                     "gripper telemetry.")
        lines.append("[gripper]   Connect AND power the gripper, then retry. "
                     "(All-zero can also mean a gripper that is attached but "
                     "unpowered.)")
    return lines


def _parse_args():
    p = argparse.ArgumentParser(
        description="Check whether an AGX gripper is connected/recognized. "
                    "Read-only; never enables or moves the gripper.")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="v112",
                   choices=["auto", "default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default: v112)")
    p.add_argument("--window", type=float, default=1.2,
                   help="seconds to sample gripper telemetry (default: 1.2)")
    return p.parse_args()


def main():
    args = _parse_args()
    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[gripper] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)
    try:
        effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        res = check_gripper(effector, args.window)
        for line in format_report(res):
            print(line)
    finally:
        robot.disconnect()
    raise SystemExit(0 if res["detected"] else 2)


if __name__ == "__main__":
    main()
