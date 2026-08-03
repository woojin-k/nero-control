#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared leader/follower mode-switching logic for the Nero arm. ACTUATES.

This is a library, not a script. The thin entry points set_follower_mode.py and
set_leader_mode.py call switch() with a fixed target so each does exactly one
thing with a clear name.

On Nero firmware >= 1.12 (V112/V120) the linkage axis has only LEADER and
FOLLOWER (Piper-aligned); FOLLOWER is the normal single-arm controlled state.
Switching sends only the mode-config frame -- it does NOT enable motors or move
the arm -- and always asks for y/N confirmation unless --yes is given.

The CAN interface must be up first:
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse
import time

# Sibling module in src/nero_ctrl/ (reuses its connect + read helpers).
import connect_check as cc


def _console_confirm(message: str) -> bool:
    """Print a warning and read y/N from the console (default No)."""
    print(message)
    try:
        return input("proceed? 진행할까요? (y/N): ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _warning(target: str, before: dict) -> str:
    """Confirmation text, with a support-the-arm warning for the risky change."""
    role = before.get("linkage_role")
    if target == "follower":
        w = ("[set_follower] switching to FOLLOWER (normal single-arm control). "
             "The mode-config frame is sent; motors are NOT enabled.")
        if role == "LEADER":
            w += ("\n[set_follower] !! Leaving LEADER (zero-force drag): the arm "
                  "returns to controlled/hold. Support it first!"
                  "\n[set_follower]   !! 리더(무저항 드래그) 해제 — 팔이 제어/홀드로 "
                  "전환됩니다. 먼저 팔을 받치세요!")
        return w
    # target == "leader"
    return ("[set_leader] switching to LEADER (zero-force drag master). Motors "
            "are ENABLED first to release the joint brakes, then the arm is put "
            "into zero-force drag."
            "\n[set_leader] !! Entering LEADER: once the brakes release the arm "
            "becomes hand-movable / backdrivable and may sag under gravity. "
            "Support it FIRST, before you confirm!"
            "\n[set_leader]   !! 리더 진입 — 모터를 켜 브레이크를 풀고 무저항 드래그로 "
            "전환합니다. 브레이크가 풀리면 팔이 손으로 움직이며 중력으로 처질 수 "
            "있으니, 확인 전에 먼저 팔을 받치세요!")


def _enable_with_timeout(robot, timeout=5.0):
    """Enable all joints (releases the brakes), bounded by `timeout`."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            if robot.enable():
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.1)
    return False


def set_mode(robot, target: str, confirm=None):
    """Switch an already-connected `robot` to `target` ('leader' | 'follower').

    Reusable core for a CLI/GUI. `confirm(message) -> bool` gates the change
    (default: console y/N); a GUI can inject a popup, or pass ``lambda _: True``
    to skip. Returns ``(ok, before_modes, after_modes)``; after_modes is None if
    cancelled or if enabling failed.

    FOLLOWER only sends the mode-config frame (no enable). LEADER (zero-force
    drag) needs the joint brakes released, so it ENABLES the motors first; if
    the arm is disabled/braked, set_leader_mode() alone does nothing.
    """
    if target not in ("leader", "follower"):
        raise ValueError("target must be 'leader' or 'follower'")
    before = cc.read_modes(robot)
    confirm = confirm or _console_confirm
    if not confirm(_warning(target, before)):
        return False, before, None
    if target == "leader":
        # Zero-force drag requires the brakes released -> enable first.
        if not _enable_with_timeout(robot):
            print("[set_leader] could not enable motors / release brakes — "
                  "aborting.")
            return False, before, None
        robot.set_leader_mode()
    else:
        robot.set_follower_mode()
    time.sleep(1.0)  # let the mode settle before reading it back
    after = cc.read_modes(robot)
    return True, before, after


def switch(target: str, can: str = "can0", fw: str = "v112",
           yes: bool = False) -> int:
    """Connect, switch the arm to `target`, verify, disconnect. Returns exit code.

    Thin CLI wrapper around set_mode() that owns the connection; a CLI/GUI with
    its own shared connection should call set_mode(robot, target, confirm).
    """
    if target not in ("leader", "follower"):
        raise ValueError("target must be 'leader' or 'follower'")
    tag = f"set_{target}"

    try:
        robot, _ = cc.connect(can, fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] cannot connect on '{can}': {type(exc).__name__}: {exc}")
        print(f"[{tag}] is the interface up? bring it up with:")
        print("  sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/"
              "scripts/linux/can_activate.sh can0 1000000")
        return 1

    try:
        print(f"[{tag}] current mode : {cc.format_mode_oneline(cc.read_modes(robot))}")
        print(f"[{tag}] target mode  : {target}")

        confirm = (lambda _: True) if yes else None
        ok, _before, after = set_mode(robot, target, confirm)
        if not ok:
            print(f"[{tag}] not switched (cancelled or enable failed).")
            return 3

        print(f"[{tag}] sent mode-config: {target}")
        print(f"[{tag}] mode now     : {cc.format_mode_oneline(after)}")
        print(f"[{tag}] comm ok / fps: {robot.is_ok()} / {robot.get_fps()}")

        # Verify the switch actually took (leader needs brakes released + drag).
        role = after.get("linkage_role")
        if target == "leader" and role != "LEADER":
            print(f"[{tag}] !! still {role}, not LEADER. If it stays this way, "
                  "make sure the arm is supported and try again.")
            return 5
        if target == "follower" and role == "LEADER":
            print(f"[{tag}] !! still LEADER — retry.")
            return 5
        return 0
    finally:
        robot.disconnect()


def parse_args(description: str) -> argparse.Namespace:
    """Shared CLI args for the set_*_mode.py entry points."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="v112",
                   choices=["default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default: v112)")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
    return p.parse_args()
