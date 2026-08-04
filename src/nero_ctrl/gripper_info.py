#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show everything knowable about the attached AGX gripper. Read-only.

The AGX gripper does NOT expose its own firmware version over CAN — neither
piper_sdk nor pyAgxArm has such a query; the 0x4AF firmware request answers
for the ARM controller only. So this reports what CAN be read:

  * arm controller firmware (it owns the gripper protocol, so its version is
    what determines gripper behavior),
  * live gripper telemetry: width/angle mode, value, force, status bits,
  * the teaching-pendant params (range %, max stroke, friction) via 0x477/47E,
  * protocol fingerprints seen on this unit: whether the param query answers,
    and whether the homed bit / 0x476 set-ACKs are implemented (on the lab
    Nero fw 1.12 + gripper they are NOT — zero-set applies silently).

Usage:
    ./.venv/bin/python src/nero_ctrl/gripper_info.py --can can0

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse

import connect_check as cc
import check_gripper as cg


def gripper_info(robot, effector, window=1.0):
    """Collect arm-firmware + gripper telemetry/params. Read-only.

    Returns a dict: arm_fw (str|None), check (check_gripper result),
    param (dict|None: range_per/max_range_m/friction), param_answers (bool).
    """
    arm_fw = None
    try:
        fw = robot.get_firmware(timeout=2.0)
        if isinstance(fw, dict):
            arm_fw = fw.get("software_version") or str(fw)
        elif fw is not None:
            arm_fw = str(fw)
    except Exception:  # noqa: BLE001
        pass

    chk = cg.check_gripper(effector, window)

    param = None
    try:
        p = effector.get_gripper_teaching_pendant_param(timeout=1.5)
    except Exception:  # noqa: BLE001
        p = None
    if p is not None:
        param = {
            "range_per": p.msg.teaching_range_per,
            "max_range_m": p.msg.max_range_config,
            "friction": p.msg.teaching_friction,
        }

    return {"arm_fw": arm_fw, "check": chk, "param": param,
            "param_answers": param is not None}


def format_info(info):
    """Human-readable report lines for a gripper_info() result."""
    lines = []
    lines.append(f"[info] arm firmware   : {info['arm_fw'] or '(no answer)'}"
                 "   (the gripper protocol lives in the arm controller)")
    lines.append("[info] gripper own fw : not exposed over CAN — the AGX "
                 "protocol has no gripper-firmware query")

    chk = info["check"]
    if not chk["detected"]:
        lines.append("[info] gripper        : NOT DETECTED (0x2A8 all-zero — "
                     "not connected or unpowered)")
        return lines

    m = chk["status"].msg
    sc = getattr(m, "status_code", 0) or 0
    fs = m.foc_status
    lines.append(f"[info] gripper        : DETECTED  mode={m.mode}  "
                 f"fps={chk['fps']}")
    if m.mode == "angle":
        lines.append(f"[info] state          : value={m.value:+.2f} deg  "
                     f"force={m.force:.2f} N  status=0x{sc:02X}")
    else:
        lines.append(f"[info] state          : value={m.value * 1000:.1f} mm  "
                     f"force={m.force:.2f} N  status=0x{sc:02X}")
    faults = [n for n in ("voltage_too_low", "motor_overheating",
                          "driver_overcurrent", "driver_overheating",
                          "sensor_status", "driver_error_status")
              if getattr(fs, n, False)]
    lines.append(f"[info] bits           : enabled={fs.driver_enable_status}  "
                 f"homed={fs.homing_status}  faults={faults or 'none'}")

    p = info["param"]
    if p is not None:
        lines.append(f"[info] pendant params : teaching_range={p['range_per']}%"
                     f"  max_stroke={p['max_range_m'] * 1000:.0f} mm"
                     f"  friction={p['friction']}   (0x477->0x47E answers: OK)")
    else:
        lines.append("[info] pendant params : NO answer to the 0x477 enquiry")
    lines.append("[info] fingerprints   : homed bit + 0x476 set-ACKs are not "
                 "implemented on this unit — zero-set/config apply silently")
    return lines


def _parse_args():
    p = argparse.ArgumentParser(
        description="Show arm firmware + all readable AGX gripper info. "
                    "Read-only; never enables or moves anything.")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="v112",
                   choices=["auto", "default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default: v112)")
    return p.parse_args()


def main():
    args = _parse_args()
    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[info] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)
    try:
        effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        info = gripper_info(robot, effector)
        for line in format_info(info):
            print(line)
    finally:
        robot.disconnect()
    raise SystemExit(0 if info["check"]["detected"] else 2)


if __name__ == "__main__":
    main()
