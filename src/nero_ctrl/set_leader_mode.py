#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Switch the Nero arm to LEADER mode (zero-force drag master). ACTUATES.

LEADER puts the arm in zero-force drag: it becomes hand-movable / backdrivable
and broadcasts its joint angles (ArmMsgFeedbackLeaderJointStates) to drive a
follower arm. The arm may sag under gravity once it goes backdrivable, so
support it before switching.

Only the mode-config frame is sent: motors are NOT enabled and the arm is NOT
commanded to move. Asks for y/N confirmation unless --yes is given.

Usage:
    ./.venv/bin/python src/nero_ctrl/set_leader_mode.py --can can0

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import mode_control as mc


def main():
    args = mc.parse_args(
        "Switch the Nero arm to LEADER mode (zero-force drag master).")
    raise SystemExit(mc.switch("leader", args.can, args.fw, args.yes))


if __name__ == "__main__":
    main()
