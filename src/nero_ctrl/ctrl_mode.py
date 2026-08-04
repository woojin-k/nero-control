#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Switch the Nero arm between drag-teach mode and CAN control mode. ACTUATES.

This is the ctrl_mode axis (NOT the leader/follower linkage axis — that is
mode_control.py). Two targets:

  teach : enter drag-teach (拖动示教). Motors are ENABLED first (brakes
          release), then the 0x150 frame with grag_teach_ctrl=0x01 is sent.
          !! The arm becomes hand-movable / gravity-compensated — support it!
          ctrl_mode feedback reads TEACHING_MODE (0x02).
  can   : exit drag-teach (grag_teach_ctrl=0x02) and assert CAN command
          control via the 0x151 mode frame (ctrl_mode=CAN_CTRL). This also
          pulls the arm out of LINKAGE_TEACHING_INPUT_MODE. The arm returns
          to controlled/hold — support it before confirming if it was limp.

pyAgxArm has no public drag-teach API (it only uses 0x150 for e-stop/reset),
so this sends the SDK's own ArmMsgMotionCtrl message through the driver —
the same frame piper_sdk's MotionCtrl_1(0, 0, 0x01/0x02) sends on Piper.

Usage:
    ./.venv/bin/python src/nero_ctrl/ctrl_mode.py teach
    ./.venv/bin/python src/nero_ctrl/ctrl_mode.py can
    ./.venv/bin/python src/nero_ctrl/ctrl_mode.py can --yes

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse
import time

from pyAgxArm.protocols.can_protocol.msgs.nero.default.transmit import (
    ArmMsgMotionCtrl,
)

# Sibling modules: shared connection, confirms, motor enable.
import connect_check as cc
import mode_control as mc
import move_to_pose as mp

TEACH_ENTER = 0x01   # 0x150 byte2: start teaching record (enter drag teach)
TEACH_EXIT = 0x02    # 0x150 byte2: end teaching record (exit drag teach)

WARN_TEACH = ("[ctrl] !! Entering DRAG-TEACH: motors are enabled to release "
              "the brakes, then the arm becomes hand-movable and may sag "
              "under gravity. Support it FIRST, before you confirm!\n"
              "[ctrl]   !! 드래그 티칭 진입 — 브레이크가 풀려 팔이 손으로 "
              "움직여지고 중력에 처질 수 있습니다. 확인 전에 먼저 팔을 "
              "받치세요!")

WARN_CAN = ("[ctrl] switching to CAN control: drag-teach is ended and the "
            "arm returns to controlled/hold.\n"
            "[ctrl]   CAN 제어로 전환 — 티칭이 끝나고 팔이 제어/홀드 상태로 "
            "돌아갑니다. 팔이 처져 있었다면 먼저 받치세요.")


def set_ctrl_mode(robot, target, confirm=None):
    """Switch an already-connected `robot` to `target` ('teach' | 'can').

    Reusable core for a CLI/GUI. `confirm(message) -> bool` gates the change
    (default: console y/N; ``lambda _: True`` skips). Returns
    ``(ok, before_modes, after_modes)``; after_modes is None when cancelled
    or when enabling failed for teach.
    """
    if target not in ("teach", "can"):
        raise ValueError("target must be 'teach' or 'can'")
    before = cc.read_modes(robot)
    confirm = confirm or mc._console_confirm
    if not confirm(WARN_TEACH if target == "teach" else WARN_CAN):
        return False, before, None

    if target == "teach":
        # Drag-teach needs the joint brakes released -> enable first
        # (same as entering leader mode in mode_control.py).
        if not mp.enable_with_timeout(robot):
            return False, before, None
        robot._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=TEACH_ENTER))
    else:
        robot._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=TEACH_EXIT))
        time.sleep(0.3)
        # Assert CAN command control (0x151); also leaves
        # LINKAGE_TEACHING_INPUT_MODE. Sends only mode config — no motion.
        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)

    time.sleep(0.7)  # let the mode settle before reading it back
    return True, before, cc.read_modes(robot)


def report(target, ok, after, tag="ctrl"):
    """Print the outcome of set_ctrl_mode() and return the process exit code."""
    if not ok:
        print(f"[{tag}] not switched (cancelled or enable failed).")
        return 3
    now = after.get("ctrl_mode") if after else None
    print(f"[{tag}] ctrl_mode now: {now or '?'}")
    want = "TEACHING_MODE" if target == "teach" else "CAN_CTRL"
    if now != want:
        print(f"[{tag}] !! expected {want} — the switch may not have taken. "
              "Retry, or check the arm state.")
        return 5
    if target == "teach":
        print(f"[{tag}] arm is hand-movable now. Move it by hand; switch "
              "back with 'can'.")
    return 0


def _parse_args():
    p = argparse.ArgumentParser(
        description="Switch the arm between drag-teach and CAN control mode "
                    "(ctrl_mode axis). ACTUATES — asks before switching.")
    p.add_argument("target", choices=["teach", "can"],
                   help="teach: enter drag-teach (hand-movable); "
                        "can: exit teach, back to CAN command control")
    p.add_argument("--can", default="can0", dest="can_if",
                   help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="v112",
                   choices=["auto", "default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default: v112)")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
    return p.parse_args()


def main():
    args = _parse_args()
    try:
        robot, _ = cc.connect(args.can_if, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[ctrl] cannot connect on '{args.can_if}': "
              f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)
    try:
        before = cc.read_modes(robot)
        print(f"[ctrl] current: {cc.format_mode_oneline(before)}")
        confirm = (lambda _: True) if args.yes else None
        ok, _before, after = set_ctrl_mode(robot, args.target, confirm)
        code = report(args.target, ok, after)
    finally:
        robot.disconnect()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
