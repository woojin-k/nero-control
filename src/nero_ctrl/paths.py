#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save / load named Nero paths as one JSON file per path.

A path is an ordered program that runs between saved waypoints. It holds four
kinds of step:

    move     a straight TCP line to a saved waypoint, IK'd ONCE when the path
             is planned; the solved joint waypoints are stored
    gripper  open / close, using the hand-taught jaw range
    wait     a fixed pause, in seconds
    signal   block until something happens. Today that means a human pressing
             계속 in the GUI; the schema reserves a ROS-topic source so paths
             saved now stay valid once such a topic exists.

The IK is the reason for the format. plan_line's solver is iterative and seeded
from the previous waypoint, so the branch it converges on depends on the exact
start configuration: a path that plans today may pick a different elbow
tomorrow, or fail outright near a singularity. Measured over the saved
waypoints, re-planning a known-good segment with only +/-2 deg of start noise
failed about 1 time in 12 and landed on a visibly different arm shape about 1
in 30. Storing the solved joints removes both.

That is also why a path is planned as ONE chain rather than as independent
segments: move step k starts from the joints step k-1 actually ended on, not
from the target waypoint's own saved joints. Those two disagree by more than
1 deg in 90% of segments (median 27 deg), because the same TCP pose is
reachable in several arm shapes, so stitching independently planned segments
together would put a large joint jump at every junction.

Files live under src/data/paths/ — like waypoints and sequences, the saved JSON
is git-ignored and only the folder is tracked.

JSON layout (units are explicit in the keys):
    {
      "name": "pick_place",
      "description": "",
      "saved_at": "2026-09-21T14:30:00",
      "format": 2,
      "start": {"waypoint": "d1", "joints_deg": [...]},   // where it begins
      "planner": {"step_m": 0.005, "ang_step_deg": 2.0, "frame": "base_link",
                  "rpy_convention": "Rz*Ry*Rx", "source": "jog_gui sim",
                  "tcp_offset": {"xyz_m": [...], "rpy_rad": [...]}},
      "steps": [
        {"type": "move", "to": "d2",
         "joints_deg": [[j1..j7], ...],   // N x 7, solved; [0] == the step's start
         "arc_len_m": [...], "angle_deg": [...],
         "metrics": {"points": 19, "dist_mm": 86.7, "angle_deg": 2.1,
                     "max_joint_step_deg": 2.6}},
        {"type": "gripper", "action": "close"},
        {"type": "wait", "seconds": 1.5},
        {"type": "signal", "source": "operator", "message": "부품을 올려놓으세요"}
      ]
    }

An unplanned move step simply has no "joints_deg": the editor can build the
step list first and solve the whole chain afterwards. planned() says whether a
path is ready to run.

Nothing in this module moves the arm, talks to ROS, or imports pyAgxArm.
"""
import datetime
import json
import math
import os

# Sibling module: reuse the waypoint name rules (name == filename).
import waypoints as wp

# src/data/paths/ — next to the package, independent of the caller's cwd.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS_DIR = os.path.join(_SRC_DIR, "data", "paths")

FORMAT = 2

STEP_TYPES = ("move", "gripper", "wait", "signal")

# What a "signal" step can wait for.
#   operator  a human presses 계속 in the GUI. Implemented; this is what the
#             editor creates today.
#   topic     a ROS publisher says go. RESERVED: the schema and the editor's
#             storage accept it so paths saved now stay valid, but nothing
#             publishes such a topic on this robot yet and the runner does not
#             implement it, so the editor does not offer it.
# There is no digital IO, fieldbus, or vision on this arm, so a topic is the
# only automatic source it could ever have without new hardware.
SIGNAL_SOURCES = ("operator", "topic")

# Topic signal conditions. "any" fires on the first message of any type;
# "true"/"false" expect a message with a boolean `data` field (std_msgs/Bool).
TOPIC_CONDITIONS = ("any", "true", "false")

# A move step replays from its own start posture. Past this the arm is
# somewhere else entirely and must be driven to the start first.
START_TOL_DEG = 5.0

# Bounds that keep a typo from parking the arm forever.
MAX_WAIT_S = 3600.0
MAX_TIMEOUT_S = 3600.0


def _now():
    """Current local date-time as an ISO string (seconds precision)."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def check_name(name):
    """Return an error string if `name` is invalid, else "" (waypoint rules)."""
    return wp.check_name(name)


def path_file(name, directory=PATHS_DIR):
    """Absolute path of the JSON file for `name`."""
    return os.path.join(directory, f"{name}.json")


# ----------------------------------------------------------------- steps
def move_step(to, q=None, s=None, ang=None, metrics=None):
    """A straight line to waypoint `to`. Solved when q/s/ang are given.

    q   (N, 7) joint waypoints in RADIANS (plan_line's "q")
    s   (N,)   cumulative translation in METRES  (plan_line's "s")
    ang (N,)   cumulative rotation in RADIANS    (plan_line's "ang")
    """
    step = {"type": "move", "to": to}
    if q is None:
        return step
    if s is None or ang is None:
        raise ValueError("a solved move step needs q, s and ang together")
    if not (len(q) == len(s) == len(ang)):
        raise ValueError(f"length mismatch: q={len(q)} s={len(s)} ang={len(ang)}")
    if len(q) < 2:
        raise ValueError("a move step needs at least 2 joint waypoints")
    step["joints_deg"] = [[round(math.degrees(j), 4) for j in row] for row in q]
    step["arc_len_m"] = [round(float(v), 6) for v in s]
    step["angle_deg"] = [round(math.degrees(float(v)), 4) for v in ang]
    step["metrics"] = metrics or {}
    return step


def gripper_step(action):
    """Open or close the gripper, using the hand-taught range."""
    if action not in ("open", "close"):
        raise ValueError(f"gripper action must be open/close, got '{action}'")
    return {"type": "gripper", "action": action}


def wait_step(seconds):
    """Pause for a fixed time."""
    seconds = float(seconds)
    if not 0 < seconds <= MAX_WAIT_S:
        raise ValueError(f"wait must be within 0-{MAX_WAIT_S:g} s, got {seconds:g}")
    return {"type": "wait", "seconds": round(seconds, 3)}


def signal_step(source, *, message="", topic="", condition="",
                timeout_s=0.0):
    """Block until a signal arrives. timeout_s = 0 means wait indefinitely."""
    if source not in SIGNAL_SOURCES:
        raise ValueError(f"signal source must be one of {SIGNAL_SOURCES}")
    timeout_s = float(timeout_s or 0.0)
    if not 0 <= timeout_s <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout must be within 0-{MAX_TIMEOUT_S:g} s")
    step = {"type": "signal", "source": source,
            "timeout_s": round(timeout_s, 3)}
    if source == "operator":
        step["message"] = message or "계속하려면 확인을 누르세요"
    elif source == "topic":
        if not topic.strip():
            raise ValueError("a topic signal needs a topic name")
        if condition not in TOPIC_CONDITIONS:
            raise ValueError(f"topic condition must be one of {TOPIC_CONDITIONS}")
        step["topic"] = topic.strip()
        step["condition"] = condition
    return step


def describe_step(step, index=None):
    """One human-readable line for a step, for the editor list and the log."""
    head = "" if index is None else f"{index + 1:>2}. "
    kind = step.get("type")
    if kind == "move":
        n = len(step.get("joints_deg", []))
        m = step.get("metrics", {})
        if not n:
            return f"{head}이동 → {step.get('to', '?')}   (미계획)"
        return (f"{head}이동 → {step.get('to', '?'):<10} {n:>4}점 "
                f"{m.get('dist_mm', 0):>7.1f}mm {m.get('angle_deg', 0):>6.1f}°")
    if kind == "gripper":
        return f"{head}그리퍼 {'열기' if step.get('action') == 'open' else '닫기'}"
    if kind == "wait":
        return f"{head}대기 {step.get('seconds', 0):g}초"
    if kind == "signal":
        src = step.get("source")
        t = step.get("timeout_s") or 0
        tail = f" (제한 {t:g}초)" if t else ""
        if src == "operator":
            return f"{head}신호: 계속 버튼 대기{tail}"
        if src == "topic":
            return (f"{head}신호: 토픽 {step.get('topic', '?')} "
                    f"= {step.get('condition', '?')}{tail}  (미구현)")
        return f"{head}신호: 알 수 없는 소스 '{src}'"
    return f"{head}알 수 없는 스텝 '{kind}'"


def check_steps(steps):
    """Return an error string if the step list is unusable, else ""."""
    if not steps:
        return "스텝이 하나도 없습니다"
    for i, step in enumerate(steps):
        kind = step.get("type")
        if kind not in STEP_TYPES:
            return f"{i + 1}번 스텝의 종류 '{kind}'를 모릅니다"
        if kind == "move" and not step.get("to"):
            return f"{i + 1}번 스텝에 목표 자세가 없습니다"
    if not any(s.get("type") == "move" for s in steps):
        return "이동 스텝이 하나도 없습니다"
    return ""


def planned(path_rec):
    """True when every move step carries a solved joint trajectory."""
    moves = [s for s in path_rec.get("steps", []) if s.get("type") == "move"]
    return bool(moves) and all(s.get("joints_deg") for s in moves)


def move_steps(path_rec):
    """(index, step) for each move step, in order."""
    return [(i, s) for i, s in enumerate(path_rec.get("steps", []))
            if s.get("type") == "move"]


# ------------------------------------------------------------- the record
def make_path(name, start, steps, *, description="", planner=None):
    """Build a path record. `start` is the {"waypoint", "joints_deg"} dict."""
    err = check_steps(steps)
    if err:
        raise ValueError(err)
    if not start.get("joints_deg"):
        raise ValueError("the start posture needs joint angles")
    return {
        "name": name,
        "description": description,
        "saved_at": _now(),
        "format": FORMAT,
        "start": start,
        "planner": planner or {},
        "steps": steps,
    }


def save_path(path_rec, directory=PATHS_DIR, overwrite=True):
    """Write a path record to <directory>/<name>.json. Returns (file, existed)."""
    name = path_rec["name"]
    err = check_name(name)
    if err:
        raise ValueError(err)
    os.makedirs(directory, exist_ok=True)
    f_path = path_file(name, directory)
    existed = os.path.exists(f_path)
    if existed and not overwrite:
        raise FileExistsError(f"path '{name}' already exists: {f_path}")
    with open(f_path, "w", encoding="utf-8") as f:
        json.dump(path_rec, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return f_path, existed


def _migrate_v1(rec):
    """The first single-segment layout -> the step format, in memory only.

    Version 1 stored one A->B move at the top level ("from"/"to" plus the
    joint arrays). It becomes a path with exactly one move step. The file on
    disk is left alone; it is rewritten in the new format the next time the
    path is saved.
    """
    step = {"type": "move", "to": rec.get("to", {}).get("waypoint", "?"),
            "metrics": rec.get("metrics", {})}
    for key in ("joints_deg", "arc_len_m", "angle_deg"):
        if key in rec:
            step[key] = rec[key]
    return {
        "name": rec.get("name", "?"),
        "description": rec.get("description", ""),
        "saved_at": rec.get("saved_at", ""),
        "format": FORMAT,
        "start": rec.get("from", {}),
        "planner": rec.get("planner", {}),
        "steps": [step],
        "migrated_from": 1,
    }


def load_path(name, directory=PATHS_DIR):
    """Load a path record by name. Raises FileNotFoundError if missing.

    A version-1 file (one A->B segment, no "steps") is migrated on the fly.
    """
    with open(path_file(name, directory), "r", encoding="utf-8") as f:
        rec = json.load(f)
    fmt = rec.get("format")
    if fmt is None and "steps" not in rec and "joints_deg" in rec:
        return _migrate_v1(rec)
    if fmt != FORMAT:
        raise ValueError(f"'{name}' is format {fmt}, this build reads {FORMAT}")
    return rec


def delete_path(name, directory=PATHS_DIR):
    """Delete a saved path's JSON file. Returns the file removed."""
    err = check_name(name)
    if err:
        raise ValueError(err)
    f_path = path_file(name, directory)
    os.remove(f_path)
    return f_path


def list_paths(directory=PATHS_DIR):
    """Return saved path names (sorted), or [] if none."""
    if not os.path.isdir(directory):
        return []
    return sorted(f[:-5] for f in os.listdir(directory) if f.endswith(".json"))


# ------------------------------------------------------------- conversion
def step_joints_rad(step):
    """A move step's 'joints_deg' -> (N, 7) joint angles in radians."""
    return [[math.radians(d) for d in row] for row in step["joints_deg"]]


def step_arc_len(step):
    """A move step's cumulative translation array [m], for retiming."""
    return list(step["arc_len_m"])


def step_angle_rad(step):
    """A move step's cumulative rotation array [rad], for retiming."""
    return [math.radians(d) for d in step["angle_deg"]]


def start_joints_rad(path_rec):
    """The posture the path must start from, in radians."""
    return [math.radians(d) for d in path_rec["start"]["joints_deg"]]


def gap_deg(joints_rad, target_deg):
    """Largest per-joint difference [deg] between live joints and a stored set.

    Raises ValueError when the joint counts disagree, so a caller cannot
    silently run a path saved for a different arm.
    """
    if len(joints_rad) != len(target_deg):
        raise ValueError(f"path has {len(target_deg)} joints, arm reports "
                         f"{len(joints_rad)}")
    return max(abs(math.degrees(j) - d)
               for j, d in zip(joints_rad, target_deg))


def start_gap_deg(path_rec, joints_rad):
    """How far the arm is from where this path begins [deg]."""
    return gap_deg(joints_rad, path_rec["start"]["joints_deg"])


def totals(path_rec):
    """(points, dist_mm, angle_deg) summed over the solved move steps."""
    pts = dist = ang = 0
    for _i, s in move_steps(path_rec):
        m = s.get("metrics", {})
        pts += len(s.get("joints_deg", []))
        dist += m.get("dist_mm", 0.0)
        ang += m.get("angle_deg", 0.0)
    return pts, dist, ang


def format_path(path_rec):
    """One-line human summary of a path."""
    steps = path_rec.get("steps", [])
    n_move = len(move_steps(path_rec))
    pts, dist, ang = totals(path_rec)
    bits = [f"{path_rec.get('name', '?')}",
            f"시작 {path_rec.get('start', {}).get('waypoint', '?')}",
            f"스텝 {len(steps)} (이동 {n_move})"]
    if planned(path_rec):
        bits.append(f"{pts}점 {dist:.1f}mm {ang:.1f}°")
    else:
        bits.append("미계획")
    return "  ".join(bits)
