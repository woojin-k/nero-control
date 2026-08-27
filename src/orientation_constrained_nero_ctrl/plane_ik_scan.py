"""Reachability of a work plane by direct 6-DoF IK on a grid.

For every 2 cm cell (y, z) on the plane x = X0, solve full IK for the flange
pose (position + fixed orientation). A cell is reachable iff IK converges to a
within-limits solution. Strategy per cell: a few random restarts, then
wavefront propagation (seed unsolved cells from a solved neighbor — warm
starts track the solution branch smoothly across the plane).

Compares the result against the Monte-Carlo cloud slab from
reachable_workspace.py and plots both.
"""
import os

import numpy as np

import nero_kinematics as nk

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

X0 = -0.31
CELL = 0.02
INTERP = "B"          # B: given matrix is the link7 orientation
N_RANDOM_WAVES = 6

G = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)
R7_TARGET = G if INTERP == "B" else G @ nk.R_FLANGE.T

ys = np.arange(-0.70, 0.70 + 1e-9, CELL)
zs = np.arange(-0.35, 0.85 + 1e-9, CELL)
YY, ZZ = np.meshgrid(ys, zs, indexing="ij")
targets = np.stack([np.full(YY.size, X0), YY.ravel(), ZZ.ravel()], axis=1)
n = len(targets)

rng = np.random.default_rng(0)
lo, hi = nk.JOINT_LIMITS[:, 0], nk.JOINT_LIMITS[:, 1]
solved = np.zeros(n, dtype=bool)
q_sol = np.zeros((n, 7))

# --- random-restart waves ----------------------------------------------------
for wave in range(N_RANDOM_WAVES):
    idx = np.flatnonzero(~solved)
    if not len(idx):
        break
    q0 = rng.uniform(lo, hi, size=(len(idx), 7))
    q, ok = nk.ik_pose(q0, R7_TARGET, targets[idx])
    q_sol[idx[ok]] = q[ok]
    solved[idx[ok]] = True
    print(f"random wave {wave}: {solved.sum()}/{n} solved")

# --- wavefront propagation from solved neighbors -----------------------------
shape = YY.shape
for round_ in range(60):
    s2 = solved.reshape(shape)
    grown = np.zeros_like(s2)
    seed_from = np.full(n, -1)
    for dy, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.roll(s2, (dy, dz), axis=(0, 1))
        if dy == 1:
            nb[0, :] = False
        if dy == -1:
            nb[-1, :] = False
        if dz == 1:
            nb[:, 0] = False
        if dz == -1:
            nb[:, -1] = False
        cand = nb & ~s2 & ~grown
        ci = np.flatnonzero(cand.ravel())
        src = np.flatnonzero(np.roll(cand, (-dy, -dz), (0, 1)).ravel())
        seed_from[ci] = src
        grown |= cand
    idx = np.flatnonzero(grown.ravel() & ~solved)
    if not len(idx):
        break
    q, ok = nk.ik_pose(q_sol[seed_from[idx]], R7_TARGET, targets[idx], iters=80)
    q_sol[idx[ok]] = q[ok]
    solved[idx[ok]] = True
    if not ok.any():
        break
print(f"after propagation: {solved.sum()}/{n} solved")

# --- final wave: seed unsolved cells from nearby Monte-Carlo witnesses -------
mc = np.load(os.path.join(OUT, "workspace_B_link7.npz"))
mc_p, mc_q = mc["p"].astype(float), mc["q"].astype(float)
near = np.abs(mc_p[:, 0] - X0) < 0.03
mc_p, mc_q = mc_p[near], mc_q[near]
# map each occupied y-z cell to one MC witness config in/near it
witness = {}
for i, c in enumerate(map(tuple, np.floor(mc_p[:, 1:] / CELL).astype(int))):
    witness.setdefault(c, i)
idx = np.flatnonzero(~solved)
cells_t = np.floor(targets[idx][:, 1:] / CELL).astype(int)
have = np.array([tuple(c) in witness for c in cells_t])
idx = idx[have]
if len(idx):
    q0 = mc_q[[witness[tuple(c)] for c in cells_t[have]]]
    q, ok = nk.ik_pose(q0, R7_TARGET, targets[idx], iters=80)
    q_sol[idx[ok]] = q[ok]
    solved[idx[ok]] = True
    print(f"after MC-seeded wave: {solved.sum()}/{n} solved")

np.savez_compressed(os.path.join(OUT, f"plane_ik_{X0:+.2f}.npz"),
                    targets=targets, solved=solved, q=q_sol, x0=X0)

# --- compare with the Monte-Carlo slab ---------------------------------------
d = np.load(os.path.join(OUT, "workspace_B_link7.npz"))
p = d["p"]
slab = p[np.abs(p[:, 0] - X0) < CELL / 2][:, 1:]
mc_cells = set(map(tuple, np.floor(slab / CELL).astype(int)))
ik_cells = set(map(tuple, np.floor(targets[solved][:, 1:] / CELL).astype(int)))
both = mc_cells & ik_cells
print(f"\nplane x0={X0}: IK {len(ik_cells)} cells ({len(ik_cells)*CELL*CELL:.3f} m^2), "
      f"MC {len(mc_cells)} cells ({len(mc_cells)*CELL*CELL:.3f} m^2)")
print(f"both {len(both)}, IK-only {len(ik_cells - mc_cells)}, "
      f"MC-only {len(mc_cells - ik_cells)}")

# --- figure ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "NanumGothic", "axes.unicode_minus": False,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
})
fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)


def draw(cells, color, label):
    if cells:
        a = (np.array(sorted(cells)) + 0.5) * CELL
        ax.scatter(a[:, 0], a[:, 1], s=14, marker="s", color=color,
                   linewidths=0, label=f"{label} ({len(cells)})")


draw(both, BLUE, "IK와 몬테카를로 일치")
draw(ik_cells - mc_cells, ORANGE, "IK만 도달 (MC 표본 누락)")
draw(mc_cells - ik_cells, AQUA, "MC만 (IK 미수렴)")
ax.plot(0, 0, "+", color=INK2, ms=10, mew=1.5)
ax.set_aspect("equal")
ax.set_xlabel("y [m]")
ax.set_ylabel("z [m]")
ax.set_title(f"작업 평면 x = {X0} m 도달 판정 — 격자 6-DoF IK vs 몬테카를로", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
path = os.path.join(OUT, f"plane_ik_scan_{X0:+.2f}.png")
fig.tight_layout()
fig.savefig(path, facecolor=SURFACE)
print(path)
