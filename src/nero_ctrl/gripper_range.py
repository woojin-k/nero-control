#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persist a hand-taught gripper travel range. No hardware access.

Why this exists: the AGX zero-set (0xAE) only fixes ONE end of the scale — it
stores "wherever the jaws are right now" as zero. Everything else (the full-open
target, the valid target range) comes from the teaching-pendant param, which
reports the gripper's FACTORY stroke (70 / 100 mm). On a gripper with modified
jaw tips those two disagree:

  * thicker tips make the jaws meet EARLY, so the new zero sits part-way up the
    factory scale and the usable stroke shrinks by twice the tip thickness;
  * commanding the factory 70 mm then drives past the physical end stop, where
    move_gripper's continuous resend keeps pushing into the stall -> the driver
    trips (overcurrent / undervoltage) and the gripper drops off the CAN bus.

So instead of trusting the factory number we teach BOTH ends by hand, with the
motor disabled, and store what the encoder actually read. That needs no tip
measurement and it cannot ram an end stop, because a human sets the limits.

JSON layout (src/data/gripper.json), values in the gripper's NATIVE units --
metres in width mode, degrees in angle mode, so `mode` fixes the unit:
    {
      "mode": "width",
      "saved_at": "2026-08-14T16:40:00",
      "closed": 0.0,          // reading with the jaws fully closed by hand
      "open": 0.0583,         // reading with the jaws fully open by hand
      "backoff": 0.002,       // stay this far off the taught open end
      "note": "thick modified tips"
    }

The record is advisory: move_gripper falls back to the pendant param whenever
no record exists or its mode does not match the gripper's live mode.
"""
import datetime
import json
import os

# src/data/gripper.json — next to the package, independent of the caller's cwd.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_SRC_DIR, "data")
RANGE_PATH = os.path.join(DATA_DIR, "gripper.json")

# Default stand-off from the taught open end, in native units. The taught end
# IS the mechanical stop, so aiming exactly at it would stall there.
BACKOFF_WIDTH_M = 0.002
BACKOFF_ANGLE_DEG = 1.0

# A taught range narrower than this is almost certainly a mis-teach (the user
# never moved the jaws between the two prompts).
MIN_SPAN_WIDTH_M = 0.005
MIN_SPAN_ANGLE_DEG = 5.0


def _now():
    """Current local date-time as an ISO string (seconds precision)."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def default_backoff(mode):
    """Default stand-off from the open end, in `mode`'s native units."""
    return BACKOFF_ANGLE_DEG if mode == "angle" else BACKOFF_WIDTH_M


def min_span(mode):
    """Smallest believable taught span, in `mode`'s native units."""
    return MIN_SPAN_ANGLE_DEG if mode == "angle" else MIN_SPAN_WIDTH_M


def make_range(mode, closed, opened, backoff=None, note=""):
    """Build a range record. `closed`/`opened` are native units for `mode`."""
    return {
        "mode": mode,
        "saved_at": _now(),
        "closed": round(float(closed), 6),
        "open": round(float(opened), 6),
        "backoff": (default_backoff(mode) if backoff is None
                    else round(float(backoff), 6)),
        "note": note,
    }


def save_range(rec, path=RANGE_PATH):
    """Write a range record to `path`. Returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def load_range(path=RANGE_PATH, mode=None):
    """Load the stored range record, or None.

    Returns None when the file is missing, unreadable, malformed, or (when
    `mode` is given) recorded for a different gripper mode — a width-mode
    record must not be applied to a gripper reporting angles.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    if "closed" not in rec or "open" not in rec:
        return None
    if mode is not None and rec.get("mode") != mode:
        return None
    return rec


def clear_range(path=RANGE_PATH):
    """Delete the stored range record. Returns True if a file was removed."""
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def span(rec):
    """Usable travel of a record, in native units (always >= 0).

    The two keys are SEMANTIC, not ordered: in angle mode "open" may well be
    the numerically smaller reading, so everything here works off magnitudes.
    """
    return abs(float(rec["open"]) - float(rec["closed"]))


def limits(rec):
    """(low, high) numeric bounds of a record, whichever end is which."""
    a, b = float(rec["closed"]), float(rec["open"])
    return (a, b) if a <= b else (b, a)


def open_target(rec):
    """Target for 'open': the taught open end, held off by `backoff`.

    The stand-off is applied TOWARDS the closed end, and never past it (a
    backoff wider than the whole span would otherwise invert the range).
    """
    closed, opened = float(rec["closed"]), float(rec["open"])
    back = float(rec.get("backoff", default_backoff(rec.get("mode"))))
    back = min(back, span(rec))
    return opened - back if opened >= closed else opened + back


def close_target(rec):
    """Target for 'close': the taught closed end."""
    return float(rec["closed"])


def fmt(value, mode):
    """Human units for a native value: mm (width) / deg (angle)."""
    if mode == "angle":
        return f"{value:+.1f} deg"
    return f"{value * 1000:.1f} mm"


def format_range(rec, tag="range"):
    """Human-readable report lines for a range record (or its absence)."""
    if rec is None:
        return [f"[{tag}] no taught range stored — using the teaching-pendant "
                f"stroke. Run 'gripper calib' to teach the real travel."]
    mode = rec.get("mode", "width")
    return [
        f"[{tag}] taught range  : closed={fmt(rec['closed'], mode)}  "
        f"open={fmt(rec['open'], mode)}  span={fmt(span(rec), mode)}  "
        f"mode={mode}",
        f"[{tag}] applied targets: 'close' -> {fmt(close_target(rec), mode)}  "
        f"'open' -> {fmt(open_target(rec), mode)} "
        f"(backoff {fmt(float(rec.get('backoff', 0.0)), mode)})",
        f"[{tag}] taught at     : {rec.get('saved_at', '?')}"
        + (f"   note: {rec['note']}" if rec.get("note") else ""),
    ]
