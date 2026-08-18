#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automated pick&place accuracy experiment: N cycles of place -> measure ->
put back, with a USB microscope measuring the placement offset each cycle.

One cycle:
  1. FORWARD sequence : pick the object at A -> place it at B (under the
     microscope)
  2. settle, then capture with the USB microscope and measure the marker's
     (dx, dy, theta) offset from the image center -> CSV + images
  3. RETURN sequence  : pick the object back up at B -> place it at A,
     ready for the next cycle

The capture is blind: it shoots whatever the camera sees when the forward
sequence ends. If the arm/gripper still covers the marker the row is logged
with detected=0 and the run simply continues — check the saved *_det.png
images to see what the camera actually saw.

ZERO REFERENCE: the zero pose is loaded from microscope_capture's calib.json
(set it beforehand with the z key in the interactive tool; --calib picks the
file, and if the default path has no file, ~/Downloads/calib.json is tried).
Every trial then also logs the zero-relative offsets (d_dx_mm, d_dy_mm,
d_r_mm, d_theta_deg) — use THOSE for the analysis; the raw dx/dy columns
still contain the image-center offset of the zero itself. A saved um/px
scale in calib.json is applied automatically too.

The robot motion is NOT hardcoded here: both halves are ordinary nero_cli
sequences (seq record / seq add), executed through the same dispatch core the
CLI and GUI use. The camera side reuses microscope_capture.py (burst capture,
subpixel marker detection). This file only orchestrates and logs.

JOINT-SPACE ONLY: every arm move is forced to MOVE J. 'goto <name>' steps are
rewritten to 'goto <name> joints' (the waypoint must have a joints part), and
Cartesian steps (movep / movel) are rejected before the run starts — teach an
extra waypoint instead (e.g. a hover posture above the pick/place point).

Setup (once, in nero_cli — teach every posture as a waypoint):
    save pick_hover ...  save pick ...  save place_hover ...  save place ...
    seq add pp_forward goto pick_hover
    seq add pp_forward goto pick
    seq add pp_forward gripper close 2
    seq add pp_forward goto pick_hover
    seq add pp_forward goto place_hover
    seq add pp_forward goto place
    seq add pp_forward gripper open
    seq add pp_forward goto place_hover
    ... and pp_return with the mirrored steps back to A.

Usage:
    ./.venv/bin/python src/nero_ctrl/pick_place_experiment.py \\
        --count 100 --index 2

Every trial appends a row to experiment_log.csv in the session directory
(src/data/pickplace/<date>/ — data/ is git-ignored). Ctrl+C stops cleanly
after the current step; a failed robot step stops the run (arm holds pose).
"""
import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reusable cores: robot connection + the CLI's command dispatcher, the saved
# command sequences, and the microscope capture/measurement toolbox.
import connect_check as cc
import nero_cli
import sequences as sq
import waypoints as wp
import microscope_capture as mcap

_SRC_DIR = Path(__file__).resolve().parent.parent
OUT_ROOT = _SRC_DIR / "data" / "pickplace"

CSV_HEADER = [
    "time", "cycle", "label", "file", "frames_used", "detected",
    "cx_px", "cy_px", "theta_deg", "dx_px", "dy_px", "r_off_px",
    "scale_um_per_px", "dx_mm", "dy_mm", "r_off_mm",
    "fit_rms_px", "fallback",
    # zero-relative deltas (same columns as microscope_capture's own log)
    "d_dx_mm", "d_dy_mm", "d_r_mm", "d_theta_deg",
    "cycle_s",
]


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_runnable(name):
    """Load sequence `name`, validate it like 'seq run' does, and force every
    arm move into joint space (MOVE J):
      * 'goto <wp> [method]' is rewritten to 'goto <wp> joints' — the waypoint
        must exist and have a joints part
      * 'movep' / 'movel' (Cartesian / linear IK moves) are rejected — teach a
        waypoint for that posture instead

    Returns (steps, problems) — problems is a list of strings, empty if OK.
    """
    try:
        seq = sq.load_sequence(name)
    except FileNotFoundError:
        return [], [f"sequence '{name}' not found — record it in nero_cli "
                    f"first (seq add {name} ... / seq record {name})"]
    raw = seq.get("steps") or []
    steps, problems = [], []
    for i, line in enumerate(raw, start=1):
        err = sq.check_step(line)
        if err:
            problems.append(f"step {i} ('{line}'): {err}")
            continue
        tokens = line.split()
        head = tokens[0].lower()
        if head in ("movep", "movel"):
            problems.append(
                f"step {i} ('{line}'): Cartesian move — joint-only run. "
                "Teach that posture as a waypoint and use 'goto <name>' "
                "(or a 'movej rel ...' delta) instead.")
            continue
        if head == "goto":
            wpname = tokens[1] if len(tokens) > 1 else ""
            try:
                rec = wp.load_waypoint(wpname)
            except FileNotFoundError:
                problems.append(f"step {i} ('{line}'): waypoint '{wpname}' "
                                "not found")
                continue
            if rec.get("joints_deg") is None:
                problems.append(f"step {i} ('{line}'): waypoint '{wpname}' "
                                "has no joints part — cannot MOVE J to it")
                continue
            forced = f"goto {wpname} joints"
            if forced != line:
                print(f"[seq] {name} step {i}: '{line}' -> '{forced}' "
                      "(joint-only)")
            steps.append(forced)
            continue
        steps.append(line)
    if not raw:
        problems.append("no steps")
    return steps, problems


def run_steps(robot, state, name, steps):
    """Run one sequence through the CLI dispatcher; False on first failure."""
    for i, step in enumerate(steps, start=1):
        print(f"[run] {name} step {i}/{len(steps)}: {step}")
        if not nero_cli.dispatch(robot, state, step):
            print(f"[run] {name} step {i} FAILED.")
            return False
    return True


def capture_measure(cap, outdir, label, n, zero_ref=None):
    """Burst-capture, measure the marker, save images.

    Returns (measurement_or_None, rel_or_None, filename, frames_used);
    rel is the zero-relative delta (mcap.rel_to_zero) when a zero exists.
    """
    ts = datetime.datetime.now().strftime("%H%M%S")
    print(f"[capture] {label} #{n:03d} ... ", end="", flush=True)
    avg16, _raw, used = mcap.capture_burst(cap)
    if avg16 is None:
        print("FAILED (no frames from camera)")
        return None, None, "", 0
    m = mcap.measure_offset(avg16)
    rel = mcap.rel_to_zero(m, zero_ref)
    base = f"{label}_{n:03d}_{ts}"
    fname = f"{base}.png"
    mcap.imwrite_unicode(outdir / fname, avg16)
    if mcap.SAVE_ANNOTATED:
        disp = (avg16 // 257).astype(np.uint8)
        mcap.imwrite_unicode(outdir / f"{base}_det.png",
                             mcap.draw_detection(disp, m, zero_ref))
    print(f"saved {fname} ({used}/{mcap.N_BURST} frames)")
    print(f"[measure] {mcap.fmt_offset(m)}")
    if zero_ref is not None:
        print(f"[zero-Δ ] {mcap.fmt_rel(rel)}")
    return m, rel, fname, used


def csv_row(cycle, label, fname, used, m, rel, cycle_s=None):
    row = [_now_iso(), cycle, label, fname, used, 1 if m else 0]
    if m is None:
        row += [""] * 12
    else:
        d = m["det"]
        row += [f"{d['cx']:.3f}", f"{d['cy']:.3f}", f"{m['theta_deg']:.3f}",
                f"{m['dx_px']:.3f}", f"{m['dy_px']:.3f}",
                f"{m['r_off_px']:.3f}",
                f"{m['scale_um']:.2f}" if m["scale_um"] else "",
                f"{m['dx_mm']:.4f}" if m["dx_mm"] is not None else "",
                f"{m['dy_mm']:.4f}" if m["dy_mm"] is not None else "",
                f"{m['r_off_mm']:.4f}" if m["r_off_mm"] is not None else "",
                f"{d['rms']:.3f}" if d["rms"] is not None else "",
                "1" if d.get("fallback") else "0"]
    # same blank-unless-mm rule as microscope_capture's own CSV
    if rel is not None and rel["dr_mm"] is not None:
        row += [f"{rel['ddx_mm']:.4f}", f"{rel['ddy_mm']:.4f}",
                f"{rel['dr_mm']:.4f}", f"{rel['dtheta_deg']:.3f}"]
    else:
        row += [""] * 4
    row += [f"{cycle_s:.1f}" if cycle_s is not None else ""]
    return row


def print_summary(trials, zero_ref=None):
    done = len(trials)
    ms = [m for m in trials if m is not None]
    print(f"\n[summary] cycles measured: {done}   marker detected: {len(ms)}"
          f"   missed: {done - len(ms)}")
    if not ms:
        return
    if zero_ref is not None:
        rels = [mcap.rel_to_zero(m, zero_ref) for m in ms]
        dx = np.array([r["ddx_px"] for r in rels])
        dy = np.array([r["ddy_px"] for r in rels])
        th = np.array([r["dtheta_deg"] for r in rels])
        print("[summary] (zero-relative Δ)")
    else:
        dx = np.array([m["dx_px"] for m in ms])
        dy = np.array([m["dy_px"] for m in ms])
        th = np.array([m["theta_deg"] for m in ms])
    r = np.hypot(dx, dy)
    scale = next((m["scale_um"] for m in ms if m["scale_um"]), None)
    mm = f" = {np.mean(r) * scale / 1000.0:.4f} mm" if scale else ""
    print(f"[summary] dx    mean {np.mean(dx):+8.2f} px  std {np.std(dx):6.2f} px")
    print(f"[summary] dy    mean {np.mean(dy):+8.2f} px  std {np.std(dy):6.2f} px")
    print(f"[summary] |r|   mean {np.mean(r):8.2f} px  std {np.std(r):6.2f} px"
          f"  max {np.max(r):.2f} px{mm}")
    print(f"[summary] theta mean {np.mean(th):+8.2f} deg std {np.std(th):6.2f} deg")


def _confirm(fwd_name, fwd_steps, ret_name, ret_steps, count):
    print(f"\n[plan] {count} cycles of:  {fwd_name} ({len(fwd_steps)} steps)"
          f"  ->  capture/measure  ->  {ret_name} ({len(ret_steps)} steps)")
    for name, steps in ((fwd_name, fwd_steps), (ret_name, ret_steps)):
        print(f"  [{name}]")
        for i, s in enumerate(steps, start=1):
            print(f"    {i:2d}. {s}")
    print("\n!! ALL steps of EVERY cycle run AUTO-CONFIRMED — the arm and "
          "gripper move without further prompts until the run ends.\n"
          "!! 모든 사이클이 추가 확인 없이 연속 실행됩니다. 주변을 비우고 "
          "비상정지를 준비하세요.")
    try:
        return input("start? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def run_experiment(robot, state, cap, outdir, args, fwd_steps, ret_steps,
                   zero_ref=None):
    logf = open(outdir / "experiment_log.csv", "a", newline="",
                encoding="utf-8-sig")
    logw = csv.writer(logf)
    if logf.tell() == 0:
        logw.writerow(CSV_HEADER)

    trials = []
    state["auto_yes"] = True
    try:
        for cycle in range(1, args.count + 1):
            t0 = time.time()
            print(f"\n[cycle] ===== {cycle}/{args.count} =====")
            if not run_steps(robot, state, args.forward, fwd_steps):
                print("[cycle] forward sequence failed — run stopped "
                      "(arm holds pose, object state unknown).")
                return 1
            time.sleep(args.settle)

            m, rel, fname, used = capture_measure(cap, outdir, args.label,
                                                  cycle, zero_ref)
            trials.append(m)

            if not run_steps(robot, state, args.ret, ret_steps):
                logw.writerow(csv_row(cycle, args.label, fname, used, m, rel,
                                      time.time() - t0))
                print("[cycle] return sequence failed — run stopped "
                      "(arm holds pose, object state unknown).")
                return 1
            logw.writerow(csv_row(cycle, args.label, fname, used, m, rel,
                                  time.time() - t0))
            logf.flush()
            print(f"[cycle] {cycle}/{args.count} done in "
                  f"{time.time() - t0:.1f}s")
        return 0
    except KeyboardInterrupt:
        print("\n[cycle] interrupted by user — arm holds its pose.")
        return 1
    finally:
        state["auto_yes"] = False
        logf.close()
        print_summary(trials, zero_ref)
        print(f"[session] logs and images: {outdir}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Automated pick&place accuracy experiment "
                    "(robot sequences + USB microscope measurement).")
    p.add_argument("--can", default="can0", help="CAN channel (default can0)")
    p.add_argument("--fw", default="v112",
                   choices=["auto", "default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default v112)")
    p.add_argument("--speed", type=int, default=20,
                   help="move speed percent (default 20)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="seconds to wait for each move to confirm arrival "
                        "(default 30 — slow/long moves need more than the "
                        "CLI's 15)")
    p.add_argument("--count", type=int, default=100,
                   help="number of pick&place cycles (default 100)")
    p.add_argument("--forward", default="pp_forward",
                   help="sequence A->B: pick, place under scope, retreat "
                        "(default pp_forward)")
    p.add_argument("--return", dest="ret", default="pp_return",
                   help="sequence B->A: pick back, place at A, retreat "
                        "(default pp_return)")
    p.add_argument("--index", type=int, default=0,
                   help="microscope camera index (probe with "
                        "microscope_capture.py --probe; default 0)")
    p.add_argument("--settle", type=float, default=1.0,
                   help="seconds to wait after placing before the capture "
                        "(default 1.0)")
    p.add_argument("--label", default="trial",
                   help="file/CSV label for the cycles (default trial)")
    p.add_argument("--calib", default=str(mcap.CALIB_FILE),
                   help="calib.json with the zero pose + um/px scale, saved "
                        "by microscope_capture.py's z key (default: "
                        f"{mcap.CALIB_FILE}; falls back to "
                        "~/Downloads/calib.json if missing)")
    p.add_argument("--yes", action="store_true",
                   help="skip the start confirmation (unattended runs)")
    return p.parse_args()


def _load_zero(calib_arg):
    """Resolve the calib file, apply its saved scale, return its zero (or None).

    The default 'calib.json' is cwd-relative (microscope_capture convention);
    if it does not exist, ~/Downloads/calib.json is tried — that is where the
    interactive tool writes it when run from Downloads.
    """
    path = Path(calib_arg).expanduser()
    if not path.exists():
        alt = Path.home() / "Downloads" / "calib.json"
        if alt.exists():
            print(f"[zero] '{path}' not found — using {alt}")
            path = alt
    calib = mcap.load_calib(path)
    mcap.apply_calib_scale(calib)     # saved um/px scale (if any) wins over auto
    zero_ref = mcap.zero_from_calib(calib)
    if zero_ref is not None:
        print(f"[zero] loaded from {path}: dx0={zero_ref['dx_px']:+.2f}px "
              f"dy0={zero_ref['dy_px']:+.2f}px th0={zero_ref['theta_deg']:+.2f}deg "
              f"(set at {zero_ref.get('time', '-')})")
        print("[zero] ※ 카메라·스테이지를 옮겼거나 초점을 바꿨다면 "
              "microscope_capture.py에서 z로 영점을 다시 잡을 것")
    else:
        print(f"[zero] no zero in {path} — only raw image-center offsets will "
              "be logged. Set one first: microscope_capture.py (z key).")
    return zero_ref, path


def main():
    args = _parse_args()

    fwd_steps, fwd_bad = load_runnable(args.forward)
    ret_steps, ret_bad = load_runnable(args.ret)
    if fwd_bad or ret_bad:
        for name, bad in ((args.forward, fwd_bad), (args.ret, ret_bad)):
            for msg in bad:
                print(f"[seq] {name}: {msg}")
        raise SystemExit(1)

    zero_ref, calib_path = _load_zero(args.calib)

    if not args.yes and not _confirm(args.forward, fwd_steps,
                                     args.ret, ret_steps, args.count):
        print("[experiment] cancelled. 취소되었습니다.")
        raise SystemExit(0)

    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[experiment] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(f"[experiment] robot connected on {args.can}.")
    state = {"speed": max(1, min(args.speed, 100)),
             "timeout": max(1.0, args.timeout),
             "effector": None, "auto_yes": False, "seq_rec": None,
             "confirm": None, "interact": None}
    try:
        cap = mcap.open_camera(args.index)   # sys.exits with a hint on failure
        try:
            outdir = OUT_ROOT / datetime.datetime.now().strftime(
                "%Y-%m-%d_%H%M")
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "run_info.json").write_text(json.dumps(
                {"started": _now_iso(), "args": vars(args),
                 "calib_file": str(calib_path), "zero": zero_ref,
                 "forward_steps": fwd_steps, "return_steps": ret_steps},
                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(run_experiment(robot, state, cap, outdir, args,
                                            fwd_steps, ret_steps, zero_ref))
        finally:
            cap.release()
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
