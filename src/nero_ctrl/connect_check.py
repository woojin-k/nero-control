#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only connection / communication check for the Nero arm.

Connects to the CAN interface, verifies the SDK is receiving feedback frames,
and prints link status, FPS, firmware version, and arm mode. It does NOT
enable motors or move the arm, so it is safe to run freely.

Two subtleties this handles (learned from a real Nero on can0):

  * Firmware / parser match. The feedback CAN IDs differ per firmware
    generation, so the parser must match the arm's firmware or it decodes
    nothing (fps stays 0). We detect the firmware first, then reconnect with
    the matching config -- the same approach as demos/nero/detect_nero_series.py.
    Use --fw to skip detection and force a version.

  * Leader mode. If the arm is in LEADER (teach / teleop-master) mode it
    broadcasts leader-joint frames (0x155/156/157/170) at ~200 Hz instead of
    normal status feedback, and it will not answer firmware requests. We fall
    back to a firmware-agnostic raw-frame liveness check so "communicating" is
    still reported correctly, and we flag that the arm is in leader mode.

Call patterns are based on the pyAgxArm demos (demos/nero/detect_nero_series.py,
demos/nero/test1.py). Nothing here calls enable() or any move_* method.

Usage:
    ./.venv/bin/python src/nero_ctrl/connect_check.py --can can0
    ./.venv/bin/python src/nero_ctrl/connect_check.py --can can0 --fw v112

The CAN interface must be up first (real arm):
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse
import time
from platform import system

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

# --fw choice -> NeroFW constant. "auto" detects from firmware version.
FW_CHOICES = {
    "auto": None,
    "default": NeroFW.DEFAULT,
    "v111": NeroFW.V111,
    "v112": NeroFW.V112,
    "v120": NeroFW.V120,
}

def _enum_name(v):
    """Official SDK enum member name for a status field value (or None)."""
    if v is None:
        return None
    return getattr(v, "name", None) or str(v)


def read_modes(robot):
    """Read the arm's mode along its two independent SDK axes. Read-only.

    Returns a dict with official SDK terms:

      Axis 1 -- linkage role (set_leader_mode / set_follower_mode):
        linkage_role: "LEADER" | "FOLLOWER" | None
          The SDK exposes no direct getter. LEADER is inferred from the live
          ArmMsgFeedbackLeaderJointStates broadcast (the arm emits no normal
          status feedback while a leader); otherwise, if status feedback is
          present, the arm is a FOLLOWER (the v1.12+ single-arm default).

      Axis 2 -- control mode:
        ctrl_mode: the official CtrlMode enum name (e.g. "CAN_CTRL",
          "LINKAGE_TEACHING_INPUT_MODE"), or None when no status feedback is
          available (leader mode, or a mismatched firmware parser).

      Supplementary ArmMsgFeedbackStatus (CAN ID 0x2A1) fields -- NOT mode axes,
      just context: arm_status / mode_feedback / teach_status / motion_status,
      each the official enum name or None.
    """
    out = {"linkage_role": None, "ctrl_mode": None, "arm_status": None,
           "mode_feedback": None, "teach_status": None, "motion_status": None}

    # Axis 1 signal: is the arm broadcasting leader-joint states?
    try:
        lja = robot.get_leader_joint_angles()
    except Exception:  # noqa: BLE001 - method may be absent on some firmwares
        lja = None
    leader_active = lja is not None and getattr(lja, "hz", 0)

    # Axis 2: control-state feedback (absent while a leader / on parser mismatch).
    try:
        st = robot.get_arm_status()
    except Exception:  # noqa: BLE001
        st = None
    if st is not None:
        m = st.msg
        out["ctrl_mode"] = _enum_name(m.ctrl_mode)
        out["arm_status"] = _enum_name(m.arm_status)
        out["mode_feedback"] = _enum_name(m.mode_feedback)
        out["teach_status"] = _enum_name(m.teach_status)
        out["motion_status"] = _enum_name(m.motion_status)

    if leader_active:
        out["linkage_role"] = "LEADER"
    elif st is not None:
        out["linkage_role"] = "FOLLOWER"
    return out


def format_mode_oneline(modes):
    """One-line summary of both axes for compact before/after logging."""
    role = modes.get("linkage_role") or "(undetermined)"
    cm = modes.get("ctrl_mode")
    return f"{role} / ctrl_mode={cm}" if cm else role


def resolve_can_backend(channel=None):
    """Pick the python-can backend for this OS (mirrors the SDK demos).

    Returns (interface, channel). On Linux the real arm uses socketcan/can0.
    """
    plat = system()
    if plat == "Linux":
        return "socketcan", channel or "can0"
    if plat == "Windows":
        return "agx_cando", channel or "0"
    if plat == "Darwin":
        return "slcan", channel or "/dev/ttyACM0"
    raise RuntimeError(
        "Unsupported OS. pyAgxArm supports Linux socketcan, Windows agx_cando, "
        "and macOS slcan.")


def fw_from_version(software_version):
    """Map an arm software-version string (e.g. '1.12') to a NeroFW constant.

    Same thresholds as demos/nero/detect_nero_series.py.
    """
    sv = software_version
    if sv >= "1.20":
        return NeroFW.V120
    if sv >= "1.12":
        return NeroFW.V112
    if sv >= "1.11":
        return NeroFW.V111
    return NeroFW.DEFAULT


def _build(can_name, firmware):
    """Create + connect a Nero driver for the given firmware. Read-only."""
    interface, channel = resolve_can_backend(can_name)
    kwargs = dict(robot=ArmModel.NERO, interface=interface, channel=channel)
    if firmware is not None:
        kwargs["firmeware_version"] = firmware  # SDK spelling
    robot = AgxArmFactory.create_arm(create_agx_arm_config(**kwargs))
    robot.connect()   # starts the CAN read thread; does not actuate
    return robot


def detect_firmware(can_name, deadline=6.0):
    """Connect with the default parser and read the firmware version (read-only).

    Returns the software-version string, or None if the arm never answered
    (e.g. it is in leader mode, which does not service firmware requests).
    Does NOT call enable() -- unlike the SDK demo, which enables while polling.
    """
    robot = _build(can_name, NeroFW.DEFAULT)
    try:
        time.sleep(0.5)
        end = time.time() + deadline
        while time.time() < end:
            fw = robot.get_firmware(timeout=1.0)  # sends a request frame only
            if fw is not None:
                return fw.get("software_version")
            time.sleep(0.3)
        return None
    finally:
        robot.disconnect()


def raw_frame_rate(can_name, seconds=1.0):
    """Firmware-agnostic liveness: count raw CAN frames/sec via python-can.

    Independent of the SDK parser, so it answers "is anything on the bus?"
    even when the firmware config does not match. Read-only.
    """
    import can  # local import: only needed for the fallback check
    interface, channel = resolve_can_backend(can_name)
    bus = can.interface.Bus(channel=channel, interface=interface,
                            receive_own_messages=False)
    try:
        n = 0
        end = time.time() + seconds
        while time.time() < end:
            if bus.recv(timeout=max(0.0, end - time.time())) is not None:
                n += 1
        return n / seconds
    finally:
        bus.shutdown()


def probe_parser(can_name, warmup=1.2):
    """When firmware can't be read, guess the parser by trying a v1.12+ one.

    A leader-mode arm never answers firmware requests, so detect_firmware()
    returns None even though the arm is fine. Here we connect with the V112
    parser and check whether it actually decodes the live frames (is_ok). If it
    does, this is a v1.12+ arm (very likely in leader mode); otherwise fall back
    to DEFAULT. Read-only. Returns a NeroFW constant.
    """
    robot = _build(can_name, NeroFW.V112)
    try:
        time.sleep(warmup)
        return NeroFW.V112 if robot.is_ok() else NeroFW.DEFAULT
    finally:
        robot.disconnect()


def connect(can_name="can0", fw="auto", warmup=0.7):
    """Connect (read-only) and return (robot, firmware_version_or_None).

    fw: one of FW_CHOICES. "auto" reads the firmware and connects with the
    matching parser; if the firmware can't be read (e.g. leader mode), it probes
    a v1.12+ parser instead. Does NOT enable motors.
    """
    firmware_const = FW_CHOICES[fw]
    detected = None
    if fw == "auto":
        detected = detect_firmware(can_name)
        if detected:
            firmware_const = fw_from_version(detected)
        else:
            # Firmware unanswered — probe instead of blindly using DEFAULT.
            firmware_const = probe_parser(can_name)
    robot = _build(can_name, firmware_const)
    time.sleep(warmup)  # let the read thread + fps window fill
    return robot, detected


def report(robot, can_name, detected_version):
    """Print a read-only connection report. Returns True if the arm is talking.

    "Talking" = the SDK parser is receiving recognized frames (is_ok), OR raw
    CAN frames are present on the bus (parser mismatch / leader mode).
    """
    channel = robot.get_channel()
    connected = robot.is_connected()
    ok = robot.is_ok()
    fps = robot.get_fps()

    print(f"[connection] interface : {channel}")
    print(f"[connection] connected : {connected}")
    print(f"[connection] comm ok   : {ok}")
    print(f"[connection] sdk fps   : {fps}")
    print(f"[connection] firmware  : {detected_version or '(not detected)'}")

    # The two independent mode axes, in official SDK terms (read-only).
    m = read_modes(robot)
    print(f"[mode] axis 1  linkage role : {m['linkage_role'] or '(undetermined)'}"
          "   (leader / follower)")
    print(f"[mode] axis 2  ctrl_mode    : {m['ctrl_mode'] or '(no status feedback)'}"
          "   (CtrlMode)")
    # Other ArmMsgFeedbackStatus (0x2A1) fields -- context, not mode axes.
    print(f"[status] arm_status    : {m['arm_status'] or '-'}")
    print(f"[status] mode_feedback : {m['mode_feedback'] or '-'}")
    print(f"[status] teach_status  : {m['teach_status'] or '-'}")
    print(f"[status] motion_status : {m['motion_status'] or '-'}")

    if ok:
        print("[connection] OK: Nero is connected and communicating "
              f"({fps} fps).")
        return True

    # SDK saw nothing. Distinguish "bus is dead" from "parser mismatch".
    rate = raw_frame_rate(can_name)
    if rate > 0:
        print(f"[connection] OK(raw): CAN bus is alive ({rate:.0f} frames/s) "
              "but the SDK parser matched no feedback.")
        print("[connection]   -> wrong --fw for this arm, or the arm is in "
              "leader mode. Try --fw v112 / v120, or set the arm to normal "
              "mode.")
        return True

    print("[connection] WARNING: no CAN traffic at all. Check that the "
          "interface is up (can_activate.sh) and the arm is powered.")
    return False


def _parse_args():
    p = argparse.ArgumentParser(
        description="Read-only Nero connection / communication check "
                    "(does not enable motors).")
    p.add_argument("--can", default="can0",
                   help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="auto", choices=list(FW_CHOICES),
                   help="firmware/parser to use; 'auto' detects it (default)")
    p.add_argument("--warmup", type=float, default=0.7,
                   help="seconds to wait after connect for feedback frames")
    return p.parse_args()


def main():
    args = _parse_args()
    try:
        robot, detected = connect(args.can, args.fw, args.warmup)
    except Exception as exc:  # noqa: BLE001 - surface any connect failure clearly
        print(f"[connect_check] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        print("[connect_check] is the interface up? bring it up with:")
        print("  sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/"
              "scripts/linux/can_activate.sh can0 1000000")
        raise SystemExit(1)

    try:
        alive = report(robot, args.can, detected)
    finally:
        robot.disconnect()  # idempotent; stops the read thread, frees CAN
    raise SystemExit(0 if alive else 2)


if __name__ == "__main__":
    main()
