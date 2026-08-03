#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save a Nero waypoint JSON — from the arm's current posture, or typed values.

Two ways to save (both write one JSON per waypoint under src/data/waypoints/):

  * Current capture (default): connect and read the arm's current Cartesian pose
    and joint angles. Read-only — does NOT enable motors or move the arm. Use
    --cartesian-only / --joints-only to capture just one part.

  * Manual entry: pass --cart and/or --joints with typed values. No connection
    is made; the values you type are saved as-is.

Cartesian values are mm + degrees; joints are degrees (7 values). A waypoint may
hold the Cartesian pose, the joints, or both.

Usage:
    # capture the current posture (both parts)
    ./.venv/bin/python src/nero_ctrl/save_waypoint.py --name home --desc "folded"
    # manual entry (no arm connection)
    ./.venv/bin/python src/nero_ctrl/save_waypoint.py --name drop \\
        --cart -400 0 400 -90 0 -180 --joints 0 -100 20 120 0 0 -30
    ./.venv/bin/python src/nero_ctrl/save_waypoint.py --list

The CAN interface must be up first (for current capture):
    sudo bash .venv/lib/python3.10/site-packages/pyAgxArm/scripts/linux/can_activate.sh can0 1000000
"""
import argparse

# Sibling modules in src/nero_ctrl/.
import connect_check as cc
import waypoints as wp


def _parse_args():
    p = argparse.ArgumentParser(
        description="Save a Nero waypoint JSON from the current posture or from "
                    "typed values, or list waypoints.")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="auto",
                   choices=["auto", "default", "v111", "v112", "v120"],
                   help="firmware/parser; 'auto' detects it (default)")
    p.add_argument("--name", help="waypoint name (no spaces, not starting with "
                                  "a digit)")
    p.add_argument("--desc", default="", help="description")
    # Manual entry: providing either switches off the arm connection.
    p.add_argument("--cart", nargs=6, type=float,
                   metavar=("Xmm", "Ymm", "Zmm", "ROLLdeg", "PITCHdeg", "YAWdeg"),
                   help="manual Cartesian pose (mm, deg)")
    p.add_argument("--joints", nargs=7, type=float,
                   metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
                   help="manual joint angles (deg, 7 values)")
    # Capture mode: which part(s) to read from the arm.
    g = p.add_mutually_exclusive_group()
    g.add_argument("--cartesian-only", action="store_true",
                   help="capture only the Cartesian pose (current-capture mode)")
    g.add_argument("--joints-only", action="store_true",
                   help="capture only the joint angles (current-capture mode)")
    p.add_argument("--no-overwrite", action="store_true",
                   help="fail if a waypoint with this name already exists")
    p.add_argument("--list", action="store_true",
                   help="list saved waypoints and exit (no connection)")
    return p.parse_args()


def _do_list():
    names = wp.list_waypoints()
    if not names:
        print(f"[wp] no waypoints yet ({wp.WAYPOINTS_DIR})")
    else:
        print(f"[wp] {len(names)} waypoint(s) in {wp.WAYPOINTS_DIR}:")
        for name in names:
            print(f"  {wp.format_waypoint(wp.load_waypoint(name))}")


def _save_manual(args):
    """Save typed values without connecting to the arm."""
    cart = wp.cartesian_from_mm_deg(args.cart) if args.cart else None
    jdeg = wp.joints_deg_manual(args.joints) if args.joints else None
    record = wp.make_waypoint(args.name, args.desc, cartesian=cart, joints_deg=jdeg)
    path, existed = wp.save_waypoint(record, overwrite=not args.no_overwrite)
    verb = "overwrote" if existed else "saved"
    print(f"[wp] {verb} '{args.name}' (manual) -> {path}")
    print(f"[wp] {wp.format_waypoint(record)}")


def _save_current(args):
    """Connect, capture the current posture, and save it (read-only)."""
    want_cart = not args.joints_only
    want_joints = not args.cartesian_only
    robot, _ = cc.connect(args.can, args.fw)
    try:
        path, existed, record = wp.capture(
            robot, args.name, args.desc,
            cartesian=want_cart, joints=want_joints,
            overwrite=not args.no_overwrite)
    finally:
        robot.disconnect()
    verb = "overwrote" if existed else "saved"
    print(f"[wp] {verb} '{args.name}' (current capture) -> {path}")
    print(f"[wp] {wp.format_waypoint(record)}")


def main():
    args = _parse_args()

    if args.list:
        _do_list()
        raise SystemExit(0)

    if not args.name:
        print("[wp] --name is required (or use --list).")
        raise SystemExit(2)
    err = wp.check_name(args.name)
    if err:
        print(f"[wp] invalid name: {err}")
        raise SystemExit(2)

    manual = args.cart is not None or args.joints is not None
    try:
        if manual:
            _save_manual(args)
        else:
            _save_current(args)
    except (ValueError, FileExistsError) as exc:
        print(f"[wp] not saved: {exc}")
        raise SystemExit(2)
    except Exception as exc:  # noqa: BLE001 - connection/other failures
        print(f"[wp] error: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
