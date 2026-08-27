"""Plot the constant-orientation reachable workspace from reachable_workspace.py."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CELL = 0.02

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e7e6e2"
BLUE = "#2a78d6"    # series 1: B (tool facing the plane)
ORANGE = "#eb6834"  # series 2: A (matrix taken as flange orientation)

plt.rcParams.update({
    "font.family": "NanumGothic",
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.unicode_minus": False,
})

DATA = {
    "B": (np.load(os.path.join(OUT, "workspace_B_link7.npz")), BLUE,
          "해석 B: 행렬=link7 (공구축 -X, 평면을 향함)"),
    "A": (np.load(os.path.join(OUT, "workspace_A_flange.npz")), ORANGE,
          "해석 A: 행렬=flange (공구축 -Y, 평면과 평행)"),
}
PLANES = [-0.15, -0.25, -0.35, -0.45, -0.55]

fig = plt.figure(figsize=(13, 9.6), dpi=150)
gs = fig.add_gridspec(3, 5, height_ratios=[1.15, 1, 1], hspace=0.42, wspace=0.28,
                      left=0.06, right=0.98, top=0.90, bottom=0.07)

# ---- top: footprint area vs plane offset ------------------------------------
ax = fig.add_subplot(gs[0, :])
for key, (d, color, label) in DATA.items():
    s = d["slabs"]
    m = s[:, 0] <= -0.05
    ax.plot(s[m, 0], s[m, 1], color=color, lw=2, label=label)
    i = np.argmax(s[m, 1])
    x0, a = s[m][i, 0], s[m][i, 1]
    ax.plot([x0], [a], "o", ms=8, color=color)
    ax.annotate(f"최대 {a:.2f} m² @ x={x0:.2f}", (x0, a),
                textcoords="offset points", xytext=(8, 6), color=INK2, fontsize=9)
ax.set_xlabel("작업 평면 위치 x0 [m] (base 프레임)")
ax.set_ylabel("y–z 도달 면적 [m²]")
ax.set_title("고정 자세에서 플랜지가 도달 가능한 y–z 면적 vs 평면 거리", fontsize=12, pad=10)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.invert_xaxis()
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# ---- rows 2-3: y-z footprints at selected planes ----------------------------
for row, key in enumerate(("B", "A"), start=1):
    d, color, label = DATA[key]
    p = d["p"]
    for col, x0 in enumerate(PLANES):
        axs = fig.add_subplot(gs[row, col])
        m = np.abs(p[:, 0] - x0) < 0.01
        yz = p[m][:, 1:]
        if len(yz):
            ij = np.unique(np.floor(yz / CELL).astype(int), axis=0)
            axs.scatter(ij[:, 0] * CELL + CELL / 2, ij[:, 1] * CELL + CELL / 2,
                        s=4.2, marker="s", color=color, linewidths=0)
        axs.plot(0, 0, "+", color=INK2, ms=8, mew=1.2)  # base axis through plane
        axs.set_xlim(-0.72, 0.72)
        axs.set_ylim(-0.35, 0.85)
        axs.set_aspect("equal")
        axs.set_title(f"x0 = {x0:.2f} m", fontsize=9, color=INK2)
        if col == 0:
            axs.set_ylabel("z [m]")
        else:
            axs.set_yticklabels([])
        if row == 2:
            axs.set_xlabel("y [m]")
        else:
            axs.set_xticklabels([])
        for sp in ("top", "right"):
            axs.spines[sp].set_visible(False)
    # row label at left margin
    fig.text(0.06, 0.635 if row == 1 else 0.33, label, color=color, fontsize=10,
             fontweight="bold", ha="left")

fig.suptitle("Nero 플랜지 도달 범위 — 방향 고정, 작업 평면은 base YZ와 평행 (x<0)\n"
             "R = [-1 0 0; 0 0 -1; 0 -1 0], 조인트 리밋 반영, 40만 샘플 몬테카를로",
             fontsize=13, y=0.985)
path = os.path.join(OUT, "reachable_workspace.png")
fig.savefig(path, facecolor=SURFACE)
print(path)
