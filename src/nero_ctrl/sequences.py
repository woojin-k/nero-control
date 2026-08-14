#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save / load named command sequences as one JSON file per sequence.

A sequence is an ordered list of nero_cli command LINES, stored verbatim, so
running one is exactly like typing the same commands at the nero> prompt (the
CLI's dispatcher executes each step; nothing is reimplemented here). Files live
under src/data/sequences/ — like waypoints, the data dir is git-ignored.

JSON layout:
    {
      "name": "pick_demo",
      "description": "",
      "saved_at": "2026-08-05T12:00:00",
      "steps": [
        "movej 0 -100 20 120 0 0 -30",
        "gripper close 2",
        "movep rel 0 0 50",
        "sleep 1",
        "goto home joints"
      ]
    }

Only motion/gripper/timing commands are allowed as steps — no 'seq' nesting,
no mode switches (those need a human judging the arm's state).
"""
import datetime
import json
import os

# Sibling module: reuse the waypoint name rules (name == filename).
import waypoints as wp

# src/data/sequences/ — next to the package, independent of the caller's cwd.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQUENCES_DIR = os.path.join(_SRC_DIR, "data", "sequences")

# Command heads a sequence step may start with (checked before running).
ALLOWED_HEADS = ("movep", "movel", "movej", "joints", "goto", "gripper",
                 "sleep", "speed")


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def check_step(line):
    """Return an error string if `line` is not a valid sequence step, else ""."""
    tokens = line.split()
    if not tokens:
        return "empty step"
    head = tokens[0].lower()
    if head not in ALLOWED_HEADS:
        return (f"'{head}' is not allowed in a sequence "
                f"(allowed: {', '.join(ALLOWED_HEADS)})")
    return ""


def is_recordable(line):
    """Whether a typed command line should be captured into a sequence.

    Only commands that DO something replayable: moves, gripper actions, speed
    and sleep. Read-only commands (status/pose/bare gripper check/…) and
    interactive ones (gripper calib needs a hand on the jaws) are skipped.
    """
    tokens = line.split()
    if not tokens:
        return False
    head = tokens[0].lower()
    if head in ("movep", "movel", "movej", "goto", "sleep", "speed"):
        return len(tokens) > 1
    if head == "joints":                     # only 'joints rel …' moves
        return len(tokens) > 1 and tokens[1].lower() == "rel"
    if head == "gripper":
        return (len(tokens) > 1
                and tokens[1].lower() in ("open", "close", "to", "reset"))
    return False


def sequence_path(name, directory=SEQUENCES_DIR):
    return os.path.join(directory, f"{name}.json")


def make_sequence(name, steps, description=""):
    """Build a sequence record (steps: list of command-line strings)."""
    err = wp.check_name(name)
    if err:
        raise ValueError(err)
    if not steps:
        raise ValueError("a sequence needs at least one step")
    for i, line in enumerate(steps, start=1):
        serr = check_step(line)
        if serr:
            raise ValueError(f"step {i} ('{line}'): {serr}")
    return {
        "name": name,
        "description": description,
        "saved_at": _now(),
        "steps": list(steps),
    }


def save_sequence(seq, directory=SEQUENCES_DIR, overwrite=True):
    """Write a sequence to <directory>/<name>.json. Returns (path, existed)."""
    os.makedirs(directory, exist_ok=True)
    path = sequence_path(seq["name"], directory)
    existed = os.path.exists(path)
    if existed and not overwrite:
        raise FileExistsError(f"sequence '{seq['name']}' already exists: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seq, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path, existed


def load_sequence(name, directory=SEQUENCES_DIR):
    """Load a sequence by name. Raises FileNotFoundError if missing."""
    with open(sequence_path(name, directory), "r", encoding="utf-8") as f:
        return json.load(f)


def list_sequences(directory=SEQUENCES_DIR):
    """Return saved sequence names (sorted), or [] if none."""
    if not os.path.isdir(directory):
        return []
    return sorted(f[:-5] for f in os.listdir(directory) if f.endswith(".json"))


def format_sequence(seq):
    """One-line human summary of a sequence."""
    steps = seq.get("steps") or []
    desc = seq.get("description") or ""
    tail = f"  - {desc}" if desc else ""
    return f"{seq.get('name', '?')}  ({len(steps)} steps){tail}"
