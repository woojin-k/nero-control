#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save / load named Nero waypoints as one JSON file per waypoint.

A waypoint records a saved arm posture. Each is a single JSON file named
``<name>.json`` under src/data/waypoints/. A waypoint may hold the Cartesian
pose, the joint angles, or both; by default both are captured. Joint replay is
the reliable one (reaches poses IK rejects), while the Cartesian pose is handy
for MOVE P and for reading off coordinates.

JSON layout (units are explicit in the keys):
    {
      "name": "home",
      "description": "folded home pose",
      "saved_at": "2026-08-03T17:25:00",
      "cartesian": {                       // may be null
        "x_mm": -167.5, "y_mm": 105.4, "z_mm": 322.9,
        "roll_deg": -42.82, "pitch_deg": -86.17, "yaw_deg": -66.79
      },
      "joints_deg": [157.54, -101.65, ...] // 7 values; may be null
    }

Writers may add optional provenance keys (all read with .get(), so older files
stay valid): "frame" of the cartesian pose, "rpy_convention", the "source" that
saved it, and the "tcp_offset" the pose was measured with. jog_gui.py's
"저장된 자세" panel writes these; the direct-CAN capture() does not.

Reading the current posture (read_pose / read_joints) does NOT enable motors or
move the arm, so capture() is safe to run freely.

Name rules follow the Piper project: non-empty, no whitespace, first character
not a digit. Because the name is now a filename, path separators and '..' are
also rejected.
"""
import datetime
import json
import math
import os

# The robot-reading helpers (show_pose / show_joints) are imported lazily inside
# capture(), because they pull in pyAgxArm: everything else here is pure JSON, so
# a ROS-side GUI can save / load waypoints without the direct-CAN stack.

# src/data/waypoints/ — next to the package, independent of the caller's cwd.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAYPOINTS_DIR = os.path.join(_SRC_DIR, "data", "waypoints")


def _now():
    """Current local date-time as an ISO string (seconds precision)."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def check_name(name):
    """Return an error string if `name` is invalid, else "" (Piper rules).

    Non-empty, no whitespace, first char not a digit; plus, since the name is a
    filename here, no path separators or '..'.
    """
    if not name:
        return "name is empty"
    if any(ch.isspace() for ch in name):
        return f"name '{name}' must not contain spaces"
    if name[0].isdigit():
        return f"name '{name}' must not start with a digit"
    if "/" in name or "\\" in name or ".." in name:
        return f"name '{name}' must not contain path separators or '..'"
    return ""


def waypoint_path(name, directory=WAYPOINTS_DIR):
    """Absolute path of the JSON file for `name`."""
    return os.path.join(directory, f"{name}.json")


def cartesian_from_pose(pose6):
    """[x,y,z(m), roll,pitch,yaw(rad)] -> the JSON 'cartesian' dict (mm, deg)."""
    x, y, z, roll, pitch, yaw = pose6
    return {
        "x_mm": round(x * 1000, 3),
        "y_mm": round(y * 1000, 3),
        "z_mm": round(z * 1000, 3),
        "roll_deg": round(math.degrees(roll), 4),
        "pitch_deg": round(math.degrees(pitch), 4),
        "yaw_deg": round(math.degrees(yaw), 4),
    }


def joints_deg_from_rad(joints_rad):
    """[j1..jn] radians -> list of degrees for the JSON 'joints_deg' field."""
    return [round(math.degrees(j), 4) for j in joints_rad]


def cartesian_from_mm_deg(vals):
    """Typed [x,y,z (mm), roll,pitch,yaw (deg)] -> the JSON 'cartesian' dict."""
    x, y, z, roll, pitch, yaw = (float(v) for v in vals)
    return {
        "x_mm": round(x, 3), "y_mm": round(y, 3), "z_mm": round(z, 3),
        "roll_deg": round(roll, 4), "pitch_deg": round(pitch, 4),
        "yaw_deg": round(yaw, 4),
    }


def joints_deg_manual(vals):
    """Typed joint degrees -> rounded 'joints_deg' list (for manual entry)."""
    return [round(float(v), 4) for v in vals]


def make_waypoint(name, description="", cartesian=None, joints_deg=None,
                  extra=None):
    """Build a waypoint record. At least one of cartesian/joints_deg is required.

    `extra` merges additional keys (provenance: frame, source, tcp_offset, ...)
    into the record. Readers use .get(), so extra keys are always optional.
    """
    if cartesian is None and joints_deg is None:
        raise ValueError("a waypoint needs a cartesian pose, joints, or both")
    rec = {
        "name": name,
        "description": description,
        "saved_at": _now(),
        "cartesian": cartesian,
        "joints_deg": joints_deg,
    }
    if extra:
        rec.update(extra)
    return rec


def save_waypoint(waypoint, directory=WAYPOINTS_DIR, overwrite=True):
    """Write a waypoint record to <directory>/<name>.json. Returns (path, existed).

    Raises FileExistsError if the file exists and overwrite is False.
    """
    name = waypoint["name"]
    err = check_name(name)
    if err:
        raise ValueError(err)
    os.makedirs(directory, exist_ok=True)
    path = waypoint_path(name, directory)
    existed = os.path.exists(path)
    if existed and not overwrite:
        raise FileExistsError(f"waypoint '{name}' already exists: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(waypoint, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path, existed


def capture(robot, name, description="", cartesian=True, joints=True,
            directory=WAYPOINTS_DIR, overwrite=True):
    """Read the arm's current posture and save it as a waypoint (read-only).

    Reusable core for a CLI/GUI: takes an already-connected `robot`. Captures
    the Cartesian pose and/or the joint angles (both by default). Returns
    (path, existed, waypoint). Raises ValueError if a requested part is
    unavailable (so the caller can report it).
    """
    import show_pose as sp
    import show_joints as sj

    err = check_name(name)
    if err:
        raise ValueError(err)

    cart = None
    if cartesian:
        pose6, _src = sp.read_pose(robot)
        if pose6 is None:
            raise ValueError("cartesian pose unavailable (parser mismatch?)")
        cart = cartesian_from_pose(pose6)

    jdeg = None
    if joints:
        ja, _src = sj.read_joints(robot)
        if ja is None:
            raise ValueError("joint angles unavailable (parser mismatch?)")
        jdeg = joints_deg_from_rad(ja.msg)

    wp = make_waypoint(name, description, cartesian=cart, joints_deg=jdeg)
    path, existed = save_waypoint(wp, directory, overwrite)
    return path, existed, wp


def load_waypoint(name, directory=WAYPOINTS_DIR):
    """Load a waypoint record by name. Raises FileNotFoundError if missing."""
    path = waypoint_path(name, directory)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_waypoint(name, directory=WAYPOINTS_DIR):
    """Delete a saved waypoint's JSON file. Returns the path removed.

    Raises FileNotFoundError if there is no such waypoint.
    """
    err = check_name(name)
    if err:
        raise ValueError(err)
    path = waypoint_path(name, directory)
    os.remove(path)
    return path


def list_waypoints(directory=WAYPOINTS_DIR):
    """Return saved waypoint names (sorted), or [] if none."""
    if not os.path.isdir(directory):
        return []
    return sorted(f[:-5] for f in os.listdir(directory)
                  if f.endswith(".json"))


def to_cartesian_target(waypoint):
    """Waypoint 'cartesian' -> [x,y,z(m), roll,pitch,yaw(rad)] for move_p, or None."""
    c = waypoint.get("cartesian")
    if not c:
        return None
    return [c["x_mm"] / 1000.0, c["y_mm"] / 1000.0, c["z_mm"] / 1000.0,
            math.radians(c["roll_deg"]), math.radians(c["pitch_deg"]),
            math.radians(c["yaw_deg"])]


def to_joints_rad(waypoint):
    """Waypoint 'joints_deg' -> list of joint angles in radians for move_j, or None."""
    j = waypoint.get("joints_deg")
    if not j:
        return None
    return [math.radians(d) for d in j]


def format_waypoint(waypoint):
    """One-line human summary of a waypoint."""
    parts = [waypoint.get("name", "?")]
    c = waypoint.get("cartesian")
    if c:
        parts.append(f"XYZ=({c['x_mm']:.1f},{c['y_mm']:.1f},{c['z_mm']:.1f})mm "
                     f"RPY=({c['roll_deg']:.1f},{c['pitch_deg']:.1f},"
                     f"{c['yaw_deg']:.1f})deg")
    if waypoint.get("joints_deg"):
        parts.append("J=[" + ",".join(f"{d:.1f}" for d in waypoint["joints_deg"])
                     + "]deg")
    return "  ".join(parts)
