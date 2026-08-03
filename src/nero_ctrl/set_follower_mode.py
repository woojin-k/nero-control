#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Switch the Nero arm to FOLLOWER mode (normal single-arm control). ACTUATES.

FOLLOWER receives CAN control commands -- this is the normal single-arm control
state on Nero firmware >= 1.12 (where there is no separate "normal" mode). Use
this to leave LEADER (zero-force drag) and put the arm back under control.

Only the mode-config frame is sent: motors are NOT enabled and the arm is NOT
moved. Asks for y/N confirmation unless --yes is given.

Usage:
    ./.venv/bin/python src/nero_ctrl/set_follower_mode.py --can can0

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import mode_control as mc


def main():
    args = mc.parse_args(
        "Switch the Nero arm to FOLLOWER mode (normal single-arm control).")
    raise SystemExit(mc.switch("follower", args.can, args.fw, args.yes))


if __name__ == "__main__":
    main()
