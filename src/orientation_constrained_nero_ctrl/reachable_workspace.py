"""Constant-orientation reachable workspace of the Nero flange.

Task: work on a plane parallel to the base YZ plane at x < 0, with the
end-effector orientation fixed to

    G = [ -1  0  0 ]
        [  0  0 -1 ]
        [  0 -1  0 ]

G is ambiguous between two frames, so both are computed:
  A) G is the gripper_flange orientation  -> tool axis (flange Z) = -Y_base
  B) G is the link7 orientation           -> tool axis (flange Z) = -X_base
     (i.e. tool points straight at the work plane)

Method: Monte-Carlo projection onto the orientation constraint manifold
(see nero_kinematics.project_orientation), then slice the resulting flange
position cloud into x-slabs and rasterize the y-z footprint.
"""
import os
import time

import numpy as np

import nero_kinematics as nk

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

G = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)
TARGETS = {
    "A_flange": G @ nk.R_FLANGE.T,  # R7 such that R_flange == G
    "B_link7": G,                   # R7 == G
}
N_SAMPLES = 400_000
CELL = 0.02          # occupancy cell size in y-z [m]
SLAB = 0.02          # x-slab thickness [m]


def footprint(points_yz: np.ndarray):
    """Occupied-cell mask and area for a set of (y, z) points."""
    if len(points_yz) == 0:
        return 0.0, None, None
    ij = np.floor(points_yz / CELL).astype(int)
    cells = np.unique(ij, axis=0)
    return len(cells) * CELL * CELL, cells, ij


def analyze(name: str, R7_target: np.ndarray, seed: int):
    t0 = time.time()
    p, q = nk.sample_constant_orientation_workspace(R7_target, N_SAMPLES, seed=seed)
    print(f"[{name}] {len(p)}/{N_SAMPLES} samples on the constraint manifold "
          f"({time.time() - t0:.0f}s)")
    x = p[:, 0]
    print(f"[{name}] flange x range: [{x.min():+.3f}, {x.max():+.3f}] m")
    edges = np.arange(-0.66, 0.001, SLAB)
    rows = []
    for lo in edges[:-1]:
        m = (x >= lo) & (x < lo + SLAB)
        area, _, _ = footprint(p[m][:, 1:])
        if area > 0:
            yz = p[m][:, 1:]
            rows.append((lo + SLAB / 2, area, len(yz),
                         yz[:, 0].min(), yz[:, 0].max(),
                         yz[:, 1].min(), yz[:, 1].max()))
    rows = np.array(rows)
    np.savez_compressed(os.path.join(OUT, f"workspace_{name}.npz"),
                        p=p.astype(np.float32), q=q.astype(np.float32),
                        slabs=rows)
    return p, rows


def main():
    results = {}
    for i, (name, tgt) in enumerate(TARGETS.items()):
        results[name] = analyze(name, tgt, seed=10 + i)

    for name, (p, rows) in results.items():
        print(f"\n=== {name}: y-z footprint per plane (slab {SLAB*100:.0f} cm) ===")
        print(f"{'x0 [m]':>8} {'area [m2]':>10} {'y range [m]':>18} {'z range [m]':>18}")
        for x0, area, n, ylo, yhi, zlo, zhi in rows:
            if x0 > -0.05:
                continue
            print(f"{x0:8.2f} {area:10.3f}   [{ylo:+.2f}, {yhi:+.2f}]   "
                  f"[{zlo:+.2f}, {zhi:+.2f}]")
        best = rows[np.argmax(rows[:, 1])]
        print(f"[{name}] largest footprint: x0 = {best[0]:+.2f} m, "
              f"area = {best[1]:.3f} m^2")


if __name__ == "__main__":
    main()
