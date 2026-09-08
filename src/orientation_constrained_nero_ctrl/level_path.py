"""Level-constrained Cartesian path planning for carrying a liquid container.

Setup: the Nero is mounted sideways, so world "up" (against gravity) is +X in
base_link. The gripper holds a rectangular container of liquid from above, so
one TCP axis (~ -X of tcp_link) must stay aligned with world up the whole way.

Constraint structure:
  - endpoints: full 6-DoF poses (position + orientation, yaw included)
  - along the path: 5 DoF — 3 position + 2 leveling; yaw about the world up
    axis follows a smooth start->goal reference with a LOW weight in the
    middle, so the solver may trade yaw for joint-limit clearance (the arm has
    7 joints, so the constrained path still has 2+ redundant DoF).

Position profiles (plan_level_move(profile=...)):
  "lift"     lift by `lift` above the higher endpoint, transfer horizontally,
             lower onto the goal (all measured along the up axis) — the
             in-plane move.
  "straight" one straight segment start -> goal with no lift — the
             plane-to-plane move: diagonal, or purely vertical when the
             in-plane coordinates are kept.

The solver is the same damped-least-squares marching used in plane_ik_scan.py,
warm-started waypoint to waypoint so the solution branch stays continuous.
tcp_link is link7 plus the fixed tcp_offset from the MoveIt launch (default 0).
"""
from __future__ import annotations

import os
import threading

import numpy as np

import nero_kinematics as nk


# Cooperative cancel for long plans: set CANCEL (from any thread) and the
# planner raises PlanCancelled at its next waypoint / attempt boundary.
# Callers clear it before starting a plan (make_plan does).
CANCEL = threading.Event()


class PlanCancelled(Exception):
    """Raised by the planner when CANCEL was set mid-computation."""


def _check_cancel():
    if CANCEL.is_set():
        raise PlanCancelled("planning cancelled")

_HERE = os.path.dirname(os.path.abspath(__file__))
UP = np.array([1.0, 0.0, 0.0])   # world up in base_link (sideways mount)


# --- small rotation utilities -------------------------------------------------

def quat_to_mat(q):
    """(x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R):
    """3x3 rotation matrix -> (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.empty(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def rot_axis(axis, angle):
    """Rodrigues rotation about a unit axis."""
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def level(R, a, u=UP):
    """Minimal rotation applied to R so that R @ a == u. Returns (R', tilt)."""
    b = R @ a
    c = float(np.clip(b @ u, -1.0, 1.0))
    tilt = float(np.arccos(c))
    if tilt < 1e-12:
        return R.copy(), tilt
    axis = np.cross(b, u)
    axis /= np.linalg.norm(axis)
    return rot_axis(axis, tilt) @ R, tilt


def yaw_between(R_a, R_b, u=UP):
    """Signed angle about u taking R_a to R_b (both must be level w.r.t. u)."""
    v = np.cross(u, [0.0, 0.0, 1.0])
    if np.linalg.norm(v) < 1e-6:
        v = np.cross(u, [0.0, 1.0, 0.0])
    v /= np.linalg.norm(v)
    R_rel = R_b @ R_a.T
    v2 = R_rel @ v
    return float(np.arctan2(u @ np.cross(v, v2), v @ v2))


# Absolute-yaw reference: the gripper-down orientation of the arm stretched
# straight along base +Z with no twist (joints 1/3/5/6 = 0, the GUI's "+z
# pose"). Absolute yaw is the rotation about world up from this orientation.
YAW_REF = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])


def yaw_reference(R_cur, u=UP):
    """Yaw-zero orientation for the container-up axis that R_cur carries:
    YAW_REF tilted minimally so that same tcp axis points up (identity when
    container-up = tcp -x, the normal carry case)."""
    a = np.asarray(R_cur, float).T @ u
    R_ref, _ = level(YAW_REF, a, u)
    return R_ref


def abs_yaw(R_cur, u=UP):
    """Absolute yaw [rad] of R_cur about u, measured from yaw_reference."""
    R_lvl, _ = level(np.asarray(R_cur, float), np.asarray(R_cur, float).T @ u, u)
    return yaw_between(yaw_reference(R_cur, u), R_lvl, u)


def orientation_at_yaw(R_cur, yaw, u=UP):
    """Level orientation with absolute yaw `yaw` [rad] (same container-up
    axis as R_cur)."""
    return rot_axis(u, float(yaw)) @ yaw_reference(R_cur, u)


# --- TCP-frame kinematics (link7 + fixed tcp_offset) --------------------------

def tcp_transform(tcp_xyz=(0, 0, 0), tcp_rpy=(0, 0, 0)):
    """link7 -> tcp_link fixed transform, matching the MoveIt tcp_offset arg."""
    return nk.rpy_matrix(*tcp_rpy), np.asarray(tcp_xyz, float)


def fk_tcp(q, R_t=np.eye(3), t_t=np.zeros(3)):
    """Batched tcp pose + Jacobian ingredients: (R_tcp, p_tcp, z_axes, p_joints)."""
    R7, p7, z, pj = nk.fk_chain(q)
    return R7 @ R_t, p7 + np.einsum("nij,j->ni", R7, t_t), z, pj


def solve_pose(q0, p_target, R_target, w_yaw=1.0, u=UP,
               iters=60, damping=1e-4, pos_tol=5e-4, ang_tol=2e-4,
               k_ns=0.05, R_t=np.eye(3), t_t=np.zeros(3),
               anchor=None, radius=None, attract=None):
    """Batched weighted DLS to a tcp pose.

    The component of the orientation error parallel to u (yaw) is scaled by
    w_yaw; w_yaw=1 is full 6-DoF IK, w_yaw->0 constrains leveling only.
    Includes a small null-space pull toward mid-range joints (k_ns).
    anchor/radius: optional trust region — the solution is kept inside the
    joint-space box |q - anchor| <= radius (used to walk a null-space
    reconfiguration in bounded steps).
    Returns (q, pos_err, tilt_err, yaw_err_weighted).
    """
    q = np.atleast_2d(q0).astype(float).copy()
    p_tt = np.atleast_2d(p_target)
    lo, hi = nk.JOINT_LIMITS[:, 0], nk.JOINT_LIMITS[:, 1]
    q_mid = 0.5 * (lo + hi) if attract is None else attract
    M = np.eye(3) - (1.0 - w_yaw) * np.outer(u, u)
    for _ in range(iters):
        _check_cancel()          # finest cancel granularity: one DLS step
        R, p, z, pj = fk_tcp(q, R_t, t_t)
        e_p = p_tt - p
        e_o = nk.orientation_error(R, R_target) @ M.T
        if (np.einsum("ij,ij->i", e_p, e_p).max() < pos_tol ** 2
                and np.einsum("ij,ij->i", e_o, e_o).max() < ang_tol ** 2):
            break
        arm = p[:, None, :] - pj
        Jp = np.transpose(np.cross(z, arm), (0, 2, 1))       # (n, 3, 7)
        Jw = M @ np.transpose(z, (0, 2, 1))                  # (n, 3, 7)
        J_full = np.concatenate([Jp, Jw], axis=1)            # (n, 6, 7)
        e = np.concatenate([e_p, e_o], axis=1)
        # saturation handling: a joint pinned at a limit and pushed further out
        # contributes nothing — drop its column and re-solve so the remaining
        # joints absorb the task instead of stalling against the clip.
        free = np.ones_like(q, dtype=bool)
        for _sat in range(3):
            J = J_full * free[:, None, :]
            JJt0 = J @ np.transpose(J, (0, 2, 1))
            y = np.linalg.solve(JJt0 + damping * np.eye(6), e[:, :, None])
            dq = (np.transpose(J, (0, 2, 1)) @ y)[:, :, 0]
            # null-space: pull toward joint mid-range without disturbing the
            # task (projector solved near-undamped, or it leaks task error)
            dq_ns = k_ns * (q_mid - q) * free
            y2 = np.linalg.solve(JJt0 + 1e-8 * np.eye(6),
                                 (J @ dq_ns[:, :, None]))
            dq += dq_ns - (np.transpose(J, (0, 2, 1)) @ y2)[:, :, 0]
            pinned = (((q <= lo + 1e-9) & (dq < 0))
                      | ((q >= hi - 1e-9) & (dq > 0)))
            if not (pinned & free).any():
                break
            free &= ~pinned
        q = q + np.clip(dq * free, -0.3, 0.3)
        if anchor is not None:
            q = anchor + np.clip(q - anchor, -radius, radius)
        q = np.clip(q, lo, hi)
    R, p, _, _ = fk_tcp(q, R_t, t_t)
    e_o_full = nk.orientation_error(R, R_target)
    pos_err = np.linalg.norm(p_tt - p, axis=1)
    # true leveling error (valid for arbitrarily large yaw error): angle
    # between where the target's up-axis actually points and up
    b = np.einsum("nij,j->ni", R, R_target.T @ u)
    tilt = np.arccos(np.clip(b @ u, -1.0, 1.0))
    yaw_w = np.abs(e_o_full @ u)
    return q, pos_err, tilt, yaw_w


_WITNESS = None


def _witnesses(R_t=np.eye(3), t_t=np.zeros(3)):
    """Monte-Carlo constraint-manifold configs from reachable_workspace.py,
    with their tcp positions. Cached; None if the npz has not been generated."""
    global _WITNESS
    key = (np.round(R_t, 9).tobytes(), np.round(t_t, 9).tobytes())
    if _WITNESS is None or (_WITNESS is not False and _WITNESS[0] != key):
        path = os.path.join(_HERE, "out", "workspace_B_link7.npz")
        if not os.path.exists(path):
            _WITNESS = False
        else:
            q = np.load(path)["q"].astype(float)
            _, p, _, _ = fk_tcp(q, R_t, t_t)
            _WITNESS = (key, q, p)
    return None if _WITNESS is False else _WITNESS[1:]


def _log_rot(R):
    """Rotation matrix -> (unit axis, angle)."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    ang = float(np.arccos(c))
    if ang < 1e-9:
        return np.array([1.0, 0.0, 0.0]), 0.0
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(v)
    if n < 1e-9:                      # angle ~ pi
        ax = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, 1))
        ax[ax.argmax()] *= np.sign(ax.max()) or 1.0
        return ax / np.linalg.norm(ax), ang
    return v / n, ang


def _pick(q, ok, q_seed):
    lo, hi = nk.JOINT_LIMITS[:, 0], nk.JOINT_LIMITS[:, 1]
    if q_seed is not None:
        return q[ok][np.argmin(np.linalg.norm(q[ok] - q_seed, axis=1))]
    # no seed: prefer the config with the largest margin to its closest limit
    margin = np.minimum(q[ok] - lo, hi - q[ok]) / (hi - lo)
    return q[ok][np.argmax(margin.min(axis=1))]


def ik_tcp(R_target, p_target, q_seed=None, n_restarts=64, seed=0,
           R_t=np.eye(3), t_t=np.zeros(3), w_yaw=1.0, u=UP, **kw):
    """Tcp IK (full 6-DoF, or 5-DoF leveling-only when w_yaw < 1). Three
    stages: random restarts; re-solve from the Monte-Carlo witnesses nearest
    the target position; march from the single nearest witness, interpolating
    pose (handles large yaw offsets plain DLS cannot cross). Returns (q, ok)."""
    R_target = np.asarray(R_target, float)
    p_target = np.asarray(p_target, float)
    rng = np.random.default_rng(seed)
    lo, hi = nk.JOINT_LIMITS[:, 0], nk.JOINT_LIMITS[:, 1]
    q0 = rng.uniform(lo, hi, size=(n_restarts, 7))
    if q_seed is not None:
        q0[0] = q_seed
    need_yaw = w_yaw >= 0.99

    def _try(q0_batch, iters=150):
        _check_cancel()
        q, pe, te, ye = solve_pose(q0_batch, np.tile(p_target,
                                                     (len(q0_batch), 1)),
                                   R_target, w_yaw=w_yaw, u=u, iters=iters,
                                   R_t=R_t, t_t=t_t, **kw)
        return q, (pe < 5e-4) & (te < 2e-4) & ((ye < 2e-4) | (not need_yaw))

    q, ok = _try(q0)
    if ok.any():
        return _pick(q, ok, q_seed), True

    wit = _witnesses(R_t, t_t)
    if wit is None:
        return None, False
    q_w, p_w = wit
    near = np.argsort(np.linalg.norm(p_w - p_target, axis=1))[:64]
    q, ok = _try(q_w[near])
    if ok.any():
        return _pick(q, ok, q_seed), True

    # march from the nearest witness, interpolating position + orientation
    for j in near[:4]:
        q_m = q_w[j][None].copy()
        R_from, p_from, _, _ = fk_tcp(q_m, R_t, t_t)
        axis, ang = _log_rot(R_target @ R_from[0].T)
        n_steps = max(int(np.ceil(abs(ang) / np.deg2rad(3))),
                      int(np.ceil(np.linalg.norm(p_target - p_from[0]) / 0.005)),
                      1)
        good = True
        for i in range(1, n_steps + 1):
            t = i / n_steps
            R_i = rot_axis(axis, ang * t) @ R_from[0]
            p_i = p_from[0] + t * (p_target - p_from[0])
            q_m, pe, te, ye = solve_pose(q_m, p_i[None], R_i, w_yaw=w_yaw,
                                         u=u, iters=40, R_t=R_t, t_t=t_t, **kw)
            if pe[0] > 5e-3 or te[0] > 5e-3:
                good = False
                break
        if good:
            q_m, pe, te, ye = solve_pose(q_m, p_target[None], R_target,
                                         w_yaw=w_yaw, u=u, iters=150,
                                         R_t=R_t, t_t=t_t, **kw)
            if pe[0] < 5e-4 and te[0] < 2e-4 and (ye[0] < 2e-4
                                                  or not need_yaw):
                return q_m[0], True
    return None, False


# --- path construction --------------------------------------------------------

def _perp_basis(e, u):
    """(n_side, n_up): unit vectors perpendicular to the chord direction e.
    n_side = u x e (horizontal, left of travel), n_up = the up axis with the
    chord component removed. For a chord along u (purely vertical move)
    both are just two perpendicular horizontal directions."""
    e = np.asarray(e, float) / np.linalg.norm(e)
    side = np.cross(u, e)
    if np.linalg.norm(side) < 1e-6:                  # vertical chord
        side = np.cross(e, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(e, np.array([0.0, 1.0, 0.0]))
    side /= np.linalg.norm(side)
    up = np.cross(side, e)                           # perpendicular to both
    if up @ u < 0:
        up = -up
    return side, up / np.linalg.norm(up)


def _bulge_normal(e, u, bulge_dir):
    """Unit bulge direction for an arc: "side" (legacy: u x e) or an angle
    [deg] measured from n_side toward n_up in the plane perpendicular to
    the chord (0 = side, 90 = up, 180 = other side, 270 = down)."""
    side, up = _perp_basis(e, u)
    if bulge_dir == "side":
        return side
    ang = np.deg2rad(float(bulge_dir))
    return np.cos(ang) * side + np.sin(ang) * up


def _arc_points(a, b, u, sagitta, step, bulge_dir="side"):
    """Circular arc from a to b bulging `sagitta` metres toward
    _bulge_normal(bulge_dir) (sign flips it; the legacy "side" case: +ve
    toward u x (b - a), an arc in the plane normal to u). Returns the
    interior+end points (excluding a)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    chord = b - a
    c = np.linalg.norm(chord)
    e = chord / c
    n = _bulge_normal(e, u, bulge_dir)
    d = abs(float(sagitta))
    n = n * np.sign(sagitta)
    R = (c * c / 4 + d * d) / (2 * d)
    phi = float(np.arccos(np.clip((R - d) / R, -1.0, 1.0)))   # half-angle
    centre = 0.5 * (a + b) - (R - d) * n
    L = 2 * phi * R
    m = max(int(np.ceil(L / step)), 1)
    ang = -phi + (np.arange(1, m + 1) / m) * 2 * phi
    pts = centre + R * (np.cos(ang)[:, None] * n + np.sin(ang)[:, None] * e)
    pts[-1] = b
    return pts


PROFILES = ("lift", "straight")


def _segment_points(p_s, p_g, u, lift, step, bulge=0.0, profile="lift",
                    bulge_dir="side"):
    """Waypoints sampled at ~step spacing. profile "lift" (default): lift ->
    transfer -> lower, the transfer a straight line or (bulge != 0) a
    circular arc in the transfer plane with that sagitta [m] (sign = side,
    see _arc_points). profile "straight": ONE segment start -> goal,
    whatever the height difference (no lift) — the plane-to-plane move
    (diagonal, or purely vertical when the in-plane coordinates match); a
    straight line, or with bulge != 0 a single circular arc bulging
    `bulge` m toward bulge_dir (see _bulge_normal). Returns (pts, s,
    s_corners): s_corners are the arc lengths of the interior corners
    (lift top, transfer end) that actually exist (none for "straight")."""
    if profile == "straight":
        knots = [np.asarray(p_s, float), np.asarray(p_g, float)]
        arc_seg = 0
    elif profile == "lift":
        arc_seg = 1
        h_s, h_g = float(p_s @ u), float(p_g @ u)
        apex = max(h_s, h_g) + lift
        knots = [p_s, p_s + (apex - h_s) * u, p_g + (apex - h_g) * u, p_g]
    else:
        raise ValueError(f"unknown profile {profile!r} (use {PROFILES})")
    pts = [np.asarray(p_s, float)]
    corners = []
    for j, (a, b) in enumerate(zip(knots[:-1], knots[1:])):
        L = np.linalg.norm(b - a)
        if L < 1e-6:          # degenerate (e.g. purely vertical move)
            continue
        if j == arc_seg and abs(bulge) > 1e-6:
            pts.extend(_arc_points(a, b, u, bulge, step, bulge_dir))
        else:
            n = max(int(np.ceil(L / step)), 1)
            t = np.arange(1, n + 1)[:, None] / n
            pts.extend(np.asarray(a) + t * (np.asarray(b) - np.asarray(a)))
        corners.append(len(pts) - 1)
    pts = np.array(pts)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0),
                                                        axis=1))])
    s_corners = [float(s[i]) for i in corners[:-1]]     # drop the goal
    return pts, s, s_corners


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def _yaw_weight(t, w_mid, ramp):
    """1 at both ends, w_mid in the middle, cosine ramps of width `ramp`."""
    w = np.full_like(t, w_mid)
    m = t < ramp
    w[m] = w_mid + (1 - w_mid) * 0.5 * (1 + np.cos(np.pi * t[m] / ramp))
    m = t > 1 - ramp
    w[m] = w_mid + (1 - w_mid) * 0.5 * (1 + np.cos(np.pi * (1 - t[m]) / ramp))
    return w


def plan_level_move(q_start, p_goal, R_goal=None, *, u=UP, lift=0.08,
                    step=0.005, w_yaw_mid=0.15, ramp_frac=0.15,
                    tcp_xyz=(0, 0, 0), tcp_rpy=(0, 0, 0), max_step=0.08,
                    max_jumps=6, bulge=0.0, profile="lift", bulge_dir="side",
                    pos_tol=5e-4, tilt_tol=2e-4, yaw_tol=2e-4,
                    level_warn=np.deg2rad(2.0), verbose=True):
    """Plan a level-constrained joint path: lift/transfer/lower (profile
    "lift") or a single straight line start -> goal (profile "straight").

    q_start: (7,) current joint config, holding the container at the start pose.
    p_goal:  (3,) goal tcp position in base_link.
    R_goal:  (3,3) goal tcp orientation; None keeps the start yaw (Delta-yaw=0).
    bulge:   0 = straight transfer; otherwise the transfer is a circular arc
             in the transfer plane with this sagitta [m], bulging to the side
             given by the sign (+ = toward up x (goal - start)).
    profile: "lift" (lift above the higher endpoint, transfer, lower) or
             "straight" (one segment start -> goal, no lift — the plane-to-
             plane move, diagonal or purely vertical; bulge != 0 makes it a
             single arc bulging toward bulge_dir).
    bulge_dir: "side" (u x chord, the transfer-plane arc) or an angle [deg]
             from side toward up in the plane perpendicular to the chord.

    The container "up" axis is measured from the start pose (a = R_start^T u),
    so whatever attitude the container has at the start is preserved exactly —
    no assumption about which tcp axis it is.

    Returns dict with:
      q (N,7), p_ref (N,3), s (N,), yaw_ref (N,), pos_err/tilt_err (N,),
      tilt_mid (N-1,), jumps (list), ok (bool), fail_index (int or None),
      s_corners (list): arc lengths of the path corners (lift top / transfer
      end) — where the executor may come to a full stop
    """
    R_t, t_t = tcp_transform(tcp_xyz, tcp_rpy)
    u = np.asarray(u, float) / np.linalg.norm(u)
    q_start = np.asarray(q_start, float)

    R_s, p_s, _, _ = fk_tcp(q_start[None], R_t, t_t)
    R_s, p_s = R_s[0], p_s[0]
    a = R_s.T @ u                       # container-up axis, tcp frame
    ax = int(np.argmax(np.abs(a)))
    if verbose:
        print(f"container-up in tcp frame: {np.round(a, 3)} "
              f"(~ {'+' if a[ax] > 0 else '-'}{'xyz'[ax]})")

    if R_goal is None:
        R_g = R_s.copy()
    else:
        R_g, tilt_g = level(np.asarray(R_goal, float), a, u)
        if tilt_g > level_warn and verbose:
            print(f"warning: goal orientation was {np.rad2deg(tilt_g):.1f} deg "
                  f"off level; re-leveled it")
    d_yaw = yaw_between(R_s, R_g, u)
    p_g = np.asarray(p_goal, float)

    # fail fast if the goal pose itself has no in-limits IK solution; keep the
    # config — it doubles as a null-space attractor so the redundancy drifts
    # toward the landing configuration early instead of unwinding at the end
    q_goal_cfg, goal_ok = ik_tcp(R_g, p_g, q_seed=q_start, R_t=R_t, t_t=t_t)
    if not goal_ok:
        wit = _witnesses(R_t, t_t)
        hint = ""
        if wit is not None:
            d_near = np.linalg.norm(wit[1] - p_g, axis=1).min()
            hint = (f" (nearest constant-orientation workspace sample is "
                    f"{d_near * 1000:.0f} mm away)")
        if verbose:
            print(f"goal pose is unreachable{hint}")
        return dict(q=q_start[None], p_ref=None, s=None, yaw_ref=None,
                    pos_err=None, tilt_err=None, tilt_mid=None, jumps=[],
                    ok=False, fail_index=0, s_corners=[], bulge=bulge,
                    bulge_dir=bulge_dir, profile=profile)

    p_ref, s, s_corners = _segment_points(p_s, p_g, u, lift, step, bulge,
                                          profile, bulge_dir)
    if len(p_ref) < 2:
        if verbose:
            print("start and goal positions coincide — nothing to plan "
                  "(use the lift profile to change yaw in place)")
        return dict(q=q_start[None], p_ref=None, s=None, yaw_ref=None,
                    pos_err=None, tilt_err=None, tilt_mid=None, jumps=[],
                    ok=False, fail_index=0, s_corners=[], bulge=bulge,
                    bulge_dir=bulge_dir, profile=profile)
    t_norm = s / max(s[-1], 1e-9)
    yaw_ref = d_yaw * _smoothstep(t_norm)
    w = _yaw_weight(t_norm, w_yaw_mid, ramp_frac)
    n = len(p_ref)
    if verbose:
        if profile == "straight":
            shape = (("straight line" if abs(bulge) < 1e-6 else
                      f"arc, sagitta {abs(bulge):.3f} m toward "
                      f"{_dir_label(bulge, bulge_dir)}")
                     + f" start->goal, height change "
                     f"{float((p_g - p_s) @ u):+.3f} m, no lift")
        else:
            shape = (("straight transfer" if abs(bulge) < 1e-6
                      else f"arc transfer, sagitta {bulge:+.3f} m")
                     + f", apex +{lift:.3f} m above the higher endpoint")
        print(f"path: {s[-1]:.3f} m, {n} waypoints, {shape}, delta-yaw "
              f"{np.rad2deg(d_yaw):.1f} deg")

    out_q = [q_start.copy()]
    out_p = [p_ref[0]]
    out_s = [s[0]]
    out_yaw = [yaw_ref[0]]
    inserted = 0
    jumps = []
    fail = None
    q_mid = 0.5 * (nk.JOINT_LIMITS[:, 0] + nk.JOINT_LIMITS[:, 1])
    for k in range(1, n):
        _check_cancel()
        R_ref = rot_axis(u, yaw_ref[k]) @ R_s
        last = k == n - 1
        w_k = 1.0 if last else float(w[k])
        q_prev = out_q[-1]
        # null-space attractor: mid-range at the start blending into the goal
        # config, so redundant joints pre-align instead of bunching up late
        attract_k = q_mid + _smoothstep(t_norm[k]) * (q_goal_cfg - q_mid)
        q_sol, pe, te, ye = solve_pose(
            q_prev[None], p_ref[k][None], R_ref, w_yaw=w_k,
            u=u, iters=150 if last else 40, R_t=R_t, t_t=t_t,
            pos_tol=pos_tol, ang_tol=tilt_tol, attract=attract_k)
        if pe[0] > pos_tol or te[0] > tilt_tol or (last and ye[0] > yaw_tol):
            # The marched branch died (pinched off, or landing needs a
            # different branch) — find a config for this waypoint globally,
            # then connect from q_prev via a null-space walk. Mid-path, yaw is
            # free: if the reference yaw is infeasible here, sweep yaw offsets
            # and re-route the remaining yaw reference through what works.
            found = False
            offsets = ((0,) if last else
                       (0, -15, 15, -30, 30, -45, 45, -60, 60, -90, 90))
            for off in offsets:
                R_try = rot_axis(u, yaw_ref[k] + np.deg2rad(off)) @ R_s
                q_g, found = ik_tcp(R_try, p_ref[k], q_seed=q_prev,
                                    R_t=R_t, t_t=t_t,
                                    w_yaw=1.0 if (last or off) else w_k, u=u)
                if found:
                    break
            if not found:
                fail = k
                break
            if not last:
                # rebase the remaining yaw reference through the found yaw
                R_f, _, _, _ = fk_tcp(q_g[None], R_t, t_t)
                psi = yaw_between(R_s, level(R_f[0], a, u)[0], u)
                psi += 2 * np.pi * np.round((yaw_ref[k] - psi) / (2 * np.pi))
                rem = (s[k:] - s[k]) / max(s[-1] - s[k], 1e-9)
                yaw_ref[k:] = psi + (yaw_ref[-1] - psi) * _smoothstep(rem)
                R_ref = rot_axis(u, yaw_ref[k]) @ R_s
            q_sol = q_g[None]
            # the block below connects q_prev -> q_sol if they are far apart
        # A large jump means a null-space reconfiguration happened inside one
        # step (the tracked branch gets pinched off ahead). Park the EE at the
        # last feasible point and walk the self-motion manifold toward the
        # post-jump config in bounded steps, emitting an on-constraint
        # waypoint each step. If the walk stalls, keep the jump but flag it
        # (it gets executed slowly; leveling between those two waypoints is
        # then not guaranteed).
        if np.abs(q_sol[0] - q_prev).max() > max_step:
            R_prev = rot_axis(u, yaw_ref[k - 1]) @ R_s
            walked = _null_walk(q_prev, q_sol[0], p_ref[k - 1], R_prev,
                                float(w[k - 1]), u, R_t, t_t, max_step,
                                pos_tol, tilt_tol)
            for q_w in walked:
                out_q.append(q_w)
                out_p.append(p_ref[k - 1])
                out_s.append(s[k - 1])
                out_yaw.append(yaw_ref[k - 1])
            inserted += len(walked)
            if walked:
                q_prev = walked[-1]
                # re-track from the walked config; on failure keep the
                # already-valid q_sol (worst case it stays a flagged jump)
                q_re, pe, te, ye = solve_pose(
                    q_prev[None], p_ref[k][None], R_ref, w_yaw=w_k, u=u,
                    iters=150 if last else 60, R_t=R_t, t_t=t_t,
                    pos_tol=pos_tol, ang_tol=tilt_tol, attract=attract_k)
                if (pe[0] <= pos_tol and te[0] <= tilt_tol
                        and not (last and ye[0] > yaw_tol)):
                    q_sol = q_re
            if np.abs(q_sol[0] - q_prev).max() > 1.5 * max_step:
                jumps.append((len(out_q), float(np.abs(q_sol[0] - q_prev).max())))
                if len(jumps) > max_jumps:
                    fail = k        # thrashing — let the caller try another lift
                    break
        out_q.append(q_sol[0])
        out_p.append(p_ref[k])
        out_s.append(s[k])
        out_yaw.append(yaw_ref[k])

    q_path = np.array(out_q)
    p_out = np.array(out_p)
    s_out = np.array(out_s)
    yaw_out = np.array(out_yaw)
    if fail is not None:
        if verbose:
            print(f"FAILED at waypoint {fail}/{n} "
                  f"(p={np.round(p_ref[fail], 3)})")
        return dict(q=q_path, p_ref=p_out, s=s_out, yaw_ref=yaw_out,
                    pos_err=None, tilt_err=None, tilt_mid=None,
                    s_corners=s_corners, bulge=bulge, bulge_dir=bulge_dir,
                    profile=profile, ok=False, fail_index=fail)

    R_chk, p_chk, _, _ = fk_tcp(q_path, R_t, t_t)
    tilt_all = _tilt(R_chk, a, u)
    tilt_mid = _tilt(fk_tcp(0.5 * (q_path[:-1] + q_path[1:]), R_t, t_t)[0],
                     a, u)
    pos_err = np.linalg.norm(p_chk - p_out, axis=1)
    step_max = np.abs(np.diff(q_path, axis=0)).max() if len(q_path) > 1 else 0
    if verbose:
        if inserted:
            print(f"inserted {inserted} reconfiguration waypoints "
                  f"(null-space walks)")
        if len(jumps) > 3:
            print(f"warning: {len(jumps)} residual joint jumps (max "
                  f"{max(sz for _, sz in jumps):.3f} rad) — executed slowly, "
                  f"but leveling between those waypoint pairs is not "
                  f"guaranteed; check tilt_mid")
        else:
            for at, sz in jumps:
                print(f"warning: residual joint jump {sz:.3f} rad at "
                      f"waypoint {at} — executed slowly, but leveling "
                      f"between those two waypoints is not guaranteed")
        print(f"solved {len(q_path)} waypoints: max pos err "
              f"{pos_err.max():.2e} m, max tilt "
              f"{np.rad2deg(tilt_all.max()):.4f} deg "
              f"(midpoints {np.rad2deg(tilt_mid.max()):.4f} deg), "
              f"max joint step {step_max:.3f} rad")
    return dict(q=q_path, p_ref=p_out, s=s_out, yaw_ref=yaw_out,
                pos_err=pos_err, tilt_err=tilt_all, tilt_mid=tilt_mid,
                jumps=jumps, ok=True, fail_index=None, s_corners=s_corners,
                bulge=bulge, bulge_dir=bulge_dir, profile=profile)


def _tilt(R, a, u):
    """Angle between R @ a and u, batched."""
    return np.arccos(np.clip(np.einsum("nij,j->ni", R, a) @ u, -1.0, 1.0))


def _null_walk(q_from, q_goal, p_hold, R_hold, w_yaw, u, R_t, t_t,
               max_step, pos_tol, tilt_tol, max_iters=120):
    """Walk the self-motion manifold from q_from toward q_goal while the tcp
    holds (p_hold, R_hold up to w_yaw). Each step: project a bounded joint
    move into the constraint null space, then re-solve locally (anchored) to
    stay on the constraint. Returns the list of intermediate configs (may be
    empty if the walk stalls immediately)."""
    M = np.eye(3) - (1.0 - w_yaw) * np.outer(u, u)
    step = 0.5 * max_step
    out = []
    q = q_from.copy()
    for _ in range(max_iters):
        _check_cancel()
        d = np.abs(q - q_goal).max()
        if d <= max_step:
            break
        _, p_c, z, pj = fk_tcp(q[None], R_t, t_t)
        arm = p_c[:, None, :] - pj
        Jp = np.transpose(np.cross(z, arm), (0, 2, 1))
        Jw = M @ np.transpose(z, (0, 2, 1))
        J = np.concatenate([Jp, Jw], axis=1)[0]
        JJt = J @ J.T + 1e-8 * np.eye(6)
        dq = np.clip(q_goal - q, -step, step)
        dq -= J.T @ np.linalg.solve(JJt, J @ dq)      # null-space projection
        q_new, pe, te, _ = solve_pose(
            (q + dq)[None], p_hold[None], R_hold, w_yaw=w_yaw, u=u,
            iters=40, R_t=R_t, t_t=t_t, pos_tol=pos_tol, ang_tol=tilt_tol,
            anchor=q + dq, radius=step, k_ns=0.0)
        d_new = np.abs(q_new[0] - q_goal).max()
        if pe[0] > pos_tol or te[0] > tilt_tol or d_new > d - 5e-4:
            break                                     # stalled
        q = q_new[0]
        out.append(q.copy())
    return out


def _dir_label(bulge, bulge_dir):
    """Human label for an arc's bulge direction ('left'/'right' are about
    u x chord; angles are from side toward up)."""
    if bulge_dir == "side":
        return "left (u x chord)" if bulge > 0 else "right"
    ang = (float(bulge_dir) + (180.0 if bulge < 0 else 0.0)) % 360.0
    names = {0: "left", 45: "left-up", 90: "up", 135: "right-up",
             180: "right", 225: "right-down", 270: "down", 315: "left-down"}
    return names.get(int(round(ang)) % 360, f"{ang:.0f} deg")


def _candidate_path(q_start, p_goal, lift, bulge, tcp_xyz, tcp_rpy, u=UP,
                    profile="lift", bulge_dir="side"):
    """Reference positions of a candidate path (no IK) sampled every 1 cm."""
    R_t, t_t = tcp_transform(tcp_xyz, tcp_rpy)
    _, p_s, _, _ = fk_tcp(np.asarray(q_start, float)[None], R_t, t_t)
    pts, _, _ = _segment_points(p_s[0], np.asarray(p_goal, float), u, lift,
                                0.01, bulge, profile, bulge_dir)
    return p_s[0], pts


_WITNESS_TREE = None


def _witness_gap(pts, tcp_xyz, tcp_rpy):
    """Largest distance from any path point to its nearest Monte-Carlo
    workspace witness [m] — the path's sparsest spot. Paths the level IK can
    follow stay at ~1 cm; paths that cut through the pocket around the base
    column or leave the workspace reach 3-5 cm. None without witnesses."""
    global _WITNESS_TREE
    R_t, t_t = tcp_transform(tcp_xyz, tcp_rpy)
    wit = _witnesses(R_t, t_t)
    if wit is None:
        return None
    key = wit[1].ctypes.data
    if _WITNESS_TREE is None or _WITNESS_TREE[0] != key:
        from scipy.spatial import cKDTree
        _WITNESS_TREE = (key, cKDTree(wit[1]))
    d, _ = _WITNESS_TREE[1].query(pts)
    return float(d.max())


ARC_SAGITTAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
# straight profile: bulge directions [deg from side toward up] tried when
# the straight line fails — sideways, upward and the upward diagonals.
# Downward bulges are generated too but dropped when they dip below the
# lower endpoint (that is where the work surface is).
STRAIGHT_ARC_DIRS = (0.0, 180.0, 90.0, 45.0, 135.0, 270.0, 315.0, 225.0)
FLOOR_TOL = 0.01          # m, how far below the lower endpoint a path may dip


def plan_level_move_auto(q_start, p_goal, R_goal=None, *, lift=0.08,
                         clean_tilt=np.deg2rad(0.1), arcs=True,
                         max_arc_tries=4, profile="lift", verbose=True, **kw):
    """plan_level_move, retrying with different lift heights if the first
    attempt fails or needs flagged reconfiguration jumps (some transfer
    altitudes cross tight regions of the constrained workspace). If no
    straight transfer works, curved (circular-arc) transfers bulging to
    either side are tried next, the ones that steer furthest from where the
    straight path failed first. Returns the first clean plan, else the best
    attempt.

    profile "straight" has no lift height to vary: the straight line first;
    if it fails, single circular arcs start -> goal bulging sideways, upward
    and diagonally (never below the lower endpoint — the work surface) are
    tried, densest-workspace first; finally the best shape without the jump
    cap. The endpoints and the leveling constraint are never relaxed."""
    best = None
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r} (use {PROFILES})")
    if profile == "straight":
        lift = 0.0
        max_arc_tries = max(max_arc_tries, 6)

    def score(p):
        if not p["ok"]:
            return (2, -(p["fail_index"] or 0), 0)
        return (1, len(p["jumps"]), float(p["tilt_mid"].max()))

    def clean(p):
        return (p["ok"] and not p["jumps"]
                and p["tilt_mid"].max() < clean_tilt)

    def attempt(lift_k, bulge, label, bulge_dir="side"):
        nonlocal best
        _check_cancel()
        if verbose and label:
            print(label)
        plan = plan_level_move(q_start, p_goal, R_goal, lift=lift_k,
                               bulge=bulge, profile=profile,
                               bulge_dir=bulge_dir, verbose=verbose, **kw)
        plan["lift"] = lift_k
        if best is None or score(plan) < score(best):
            best = plan
        return plan

    lifts = (1.0, 0.5, 0.25, 1.5) if profile == "lift" else (1.0,)
    for f in lifts:
        plan = attempt(lift * f, 0.0,
                       None if f == 1.0 else f"retrying with lift {lift * f:.3f} m")
        if clean(plan):
            return plan
        if plan["fail_index"] == 0:
            # the goal pose itself has no IK solution (or start == goal):
            # no path shape helps
            return best

    if arcs:
        tcp_xyz = kw.get("tcp_xyz", (0, 0, 0))
        tcp_rpy = kw.get("tcp_rpy", (0, 0, 0))
        u = kw.get("u", UP)
        p_s, pts0 = _candidate_path(q_start, p_goal, lift, 0.0, tcp_xyz,
                                    tcp_rpy, u, profile)
        p_g = np.asarray(p_goal, float)
        chord = float(np.linalg.norm(p_g - p_s))
        sag = [d for d in ARC_SAGITTAS if d <= max(0.5 * chord,
                                                   ARC_SAGITTAS[0])]
        if profile == "lift":
            # (lift, signed sagitta, dir): arcs in the transfer plane
            cands = [(lift * f, sgn * d, "side") for d in sag
                     for sgn in (1.0, -1.0) for f in (1.0, 0.5)]
        else:
            # single arcs start -> goal in 8 directions around the chord
            cands = [(0.0, d, ang) for d in sag for ang in STRAIGHT_ARC_DIRS]
        floor = min(float(p_s @ u), float(p_g @ u)) - FLOOR_TOL
        paths = {c: _candidate_path(q_start, p_goal, c[0], c[1], tcp_xyz,
                                    tcp_rpy, u, profile, c[2])[1]
                 for c in cands}
        # never dip below the lower endpoint: that is the work surface
        cands = [c for c in cands if (paths[c] @ u).min() >= floor]
        gap0 = _witness_gap(pts0, tcp_xyz, tcp_rpy)
        if gap0 is None:
            order = sorted(cands, key=lambda c: abs(c[1]))   # gentle first
            gap = {}
        else:
            gap = {c: _witness_gap(paths[c], tcp_xyz, tcp_rpy) for c in cands}
            # arcs through denser workspace than the straight path first;
            # if none looks denser, still try the gentlest ones (the
            # straight line may have failed on joint limits, which the
            # witness density does not see)
            order = sorted((c for c in cands if gap[c] < gap0),
                           key=lambda c: (gap[c], abs(c[1])))
            if not order:
                order = sorted(cands, key=lambda c: (abs(c[1]), gap[c]))
        if verbose:
            what = "straight transfer" if profile == "lift" else "straight line"
            print(f"{what} failed — trying up to "
                  f"{min(len(order), max_arc_tries)} curved path(s)")
        for lk, b, dr in order[:max_arc_tries]:
            note = (f", sparsest spot {gap[(lk, b, dr)] * 1000:.0f} mm from "
                    f"a witness (straight: {gap0 * 1000:.0f} mm)" if gap
                    else "")
            where = (f"at lift {lk:.3f} m" if profile == "lift"
                     else f"toward {_dir_label(b, dr)}")
            plan = attempt(lk, b, f"trying arc sagitta {abs(b):.2f} m "
                                  f"{where}{note}", dr)
            if clean(plan):
                return plan

    if not best["ok"]:
        # last resort: allow unlimited flagged jumps at the nominal lift
        _check_cancel()
        if verbose:
            print(f"retrying lift {lift:.3f} m without the jump cap")
        plan = plan_level_move(q_start, p_goal, R_goal, lift=lift,
                               bulge=best["bulge"], profile=profile,
                               bulge_dir=best.get("bulge_dir", "side"),
                               max_jumps=np.inf, verbose=verbose, **kw)
        plan["lift"] = lift
        if plan["ok"]:
            best = plan
    if verbose:
        if profile == "straight":
            shape = ("straight line" if abs(best["bulge"]) < 1e-6 else
                     f"arc {abs(best['bulge']):.2f} m toward "
                     f"{_dir_label(best['bulge'], best['bulge_dir'])}")
        else:
            shape = (f"lift {best['lift']:.3f} m, "
                     + ("straight transfer" if abs(best["bulge"]) < 1e-6
                        else f"arc {best['bulge']:+.2f} m"))
        print("no clean path found; returning the best attempt "
              f"({shape}, ok={best['ok']})")
    return best


# --- time parameterization ----------------------------------------------------

def time_parameterize(s, q, v_max=0.06, a_max=0.05, qdot_max=1.0):
    """Trapezoidal (gentle, for liquid) timing over arc length, then floor each
    interval so no joint exceeds qdot_max. Returns t (N,) seconds."""
    s = np.asarray(s, float) - float(s[0])
    L = float(s[-1])
    if L <= 1e-9:                      # tcp stands still: joint-rate floor only
        need = np.abs(np.diff(q, axis=0)).max(axis=1) / qdot_max
        return np.concatenate([[0.0], np.cumsum(need)])
    v = min(v_max, np.sqrt(a_max * L))
    s_a = v * v / (2 * a_max)
    t_a = v / a_max
    T = 2 * t_a + (L - 2 * s_a) / v

    def t_of(si):
        if si <= s_a:
            return np.sqrt(2 * si / a_max)
        if si >= L - s_a:
            return T - np.sqrt(2 * max(L - si, 0.0) / a_max)
        return t_a + (si - s_a) / v

    t = np.array([t_of(si) for si in s])
    dt = np.diff(t)
    need = np.abs(np.diff(q, axis=0)).max(axis=1) / qdot_max
    return np.concatenate([[0.0], np.cumsum(np.maximum(dt, need))])


def retime_toppra(q, s_cart, v_max=0.06, a_max=0.2, qdot_max=1.0,
                  qddot_max=3.0, dt=0.1):
    """TOPP-RA retiming of a waypoint path: the fastest timing that keeps
    every joint under (qdot_max, qddot_max) AND the tcp under (v_max, a_max)
    everywhere along the path — velocity stays continuous, so no acceleration
    spikes like time_parameterize's per-interval floors.

    The tcp limit rides along as an 8th path coordinate (the cartesian arc
    length s_cart), so both joint-space and task-space bounds live in one
    joint-limits constraint.

    Returns (t, q, qd, qdd) sampled every dt seconds, or None if toppra is
    unavailable or the instance fails (caller should fall back to
    time_parameterize)."""
    try:
        import logging
        import toppra as ta
        logging.getLogger("toppra").setLevel(logging.ERROR)
    except Exception:       # not just ImportError: numpy-ABI errors etc.
        return None
    q = np.asarray(q, float)
    if len(q) < 3:
        return None
    # path parameter: joint-space arc length, nudged strictly increasing
    # (null-walk stretches hold the tcp still, so s_cart alone can stall)
    step = np.linalg.norm(np.diff(q, axis=0), axis=1)
    ss = np.concatenate([[0.0], np.cumsum(np.maximum(step, 1e-6))])
    q_ext = np.column_stack([q, np.asarray(s_cart, float)])
    path = ta.SplineInterpolator(ss, q_ext)
    vlim = np.array([[-qdot_max, qdot_max]] * 7 + [[-v_max, v_max]])
    alim = np.array([[-qddot_max, qddot_max]] * 7 + [[-a_max, a_max]])
    try:
        inst = ta.algorithm.TOPPRA(
            [ta.constraint.JointVelocityConstraint(vlim),
             ta.constraint.JointAccelerationConstraint(alim)], path)
        traj = inst.compute_trajectory(0.0, 0.0)
    except Exception:
        return None
    if traj is None:
        return None
    t = np.arange(0.0, traj.duration, dt)
    if traj.duration - t[-1] > 1e-6:
        t = np.append(t, traj.duration)
    qs = traj(t)[:, :7]
    qd = traj(t, 1)[:, :7]
    qdd = traj(t, 2)[:, :7]
    qd[0] = qd[-1] = 0.0
    # pin the sampled endpoints to the exact planned poses
    qs[0], qs[-1] = q[0], q[-1]
    return t, qs, qd, qdd


def corner_indices(s, s_corners, tol=1e-6):
    """Waypoint index of each corner: the first waypoint that reaches the
    corner's arc length (null-walk waypoints inserted *at* a corner share its
    arc length and land in the following segment)."""
    s = np.asarray(s, float)
    idx = []
    for sc in s_corners:
        hit = np.nonzero(s >= sc - tol)[0]
        if len(hit) and 0 < hit[0] < len(s) - 1 and hit[0] not in idx:
            idx.append(int(hit[0]))
    return sorted(idx)


def retime_corner_stops(q, s_cart, s_corners, dwell=0.1, v_max=0.06,
                        a_max=0.2, qdot_max=1.0, qddot_max=3.0, dt=0.1):
    """Retime the path one straight segment at a time (start -> lift top ->
    transfer end -> goal), each with zero velocity at both ends, and hold
    `dwell` seconds at every corner. Each segment goes through retime_toppra
    (smooth accel/decel ramps under the same limits), falling back to the
    trapezoidal time_parameterize when toppra is unavailable.

    Returns (t, q, qd, qdd, all_toppra); qdd is None when any segment fell
    back (its accelerations are unknown), qd is always filled."""
    q = np.asarray(q, float)
    s_cart = np.asarray(s_cart, float)
    cuts = [0] + corner_indices(s_cart, s_corners) + [len(q) - 1]
    ts, qs, qds, qdds = [], [], [], []
    all_toppra = True
    t0 = 0.0
    for k, (i0, i1) in enumerate(zip(cuts[:-1], cuts[1:])):
        q_seg, s_seg = q[i0:i1 + 1], s_cart[i0:i1 + 1]
        res = retime_toppra(q_seg, s_seg, v_max=v_max, a_max=a_max,
                            qdot_max=qdot_max, qddot_max=qddot_max, dt=dt)
        if res is not None:
            t_seg, q_smp, qd_seg, qdd_seg = res
        else:
            all_toppra = False
            t_seg = time_parameterize(s_seg, q_seg, v_max=v_max, a_max=a_max,
                                      qdot_max=qdot_max)
            q_smp = q_seg
            qd_seg = (np.gradient(q_smp, t_seg, axis=0) if len(q_smp) > 2
                      else np.zeros_like(q_smp))
            qd_seg[0] = qd_seg[-1] = 0.0
            qdd_seg = np.zeros_like(q_smp)
        if k > 0 and dwell <= 0.0:
            # no hold: the corner sample already ends the previous segment
            t_seg, q_smp, qd_seg, qdd_seg = (t_seg[1:], q_smp[1:],
                                             qd_seg[1:], qdd_seg[1:])
        ts.append(t_seg + t0)
        qs.append(q_smp)
        qds.append(qd_seg)
        qdds.append(qdd_seg)
        t0 = ts[-1][-1] + max(dwell, 0.0)
    t = np.concatenate(ts)
    qd = np.concatenate(qds)
    qdd = np.concatenate(qdds) if all_toppra else None
    return t, np.concatenate(qs), qd, qdd, all_toppra


# --- self-test ----------------------------------------------------------------

if __name__ == "__main__":
    # start: the known-good working orientation from the workspace study
    # (R7 == G -> tcp x = -X_base = straight down -> container-up = tcp -x)
    G = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)
    p_start = np.array([-0.31, 0.25, 0.45])
    q0, ok = ik_tcp(G, p_start, seed=1)
    assert ok, "could not find a start config"
    print(f"start config: {np.round(q0, 3)}")

    p_goal = np.array([-0.31, -0.15, 0.55])
    R_goal = rot_axis(UP, np.deg2rad(45)) @ G      # 45 deg yaw at the goal
    plan = plan_level_move_auto(q0, p_goal, R_goal, lift=0.08)
    assert plan["ok"], "planning failed"

    t = time_parameterize(plan["s"], plan["q"])
    rate = (np.abs(np.diff(plan["q"], axis=0)).max(axis=1)
            / np.maximum(np.diff(t), 1e-9)).max()
    print(f"duration: {t[-1]:.1f} s (max joint rate {rate:.2f} rad/s)")

    # independent verification: FK every waypoint, check leveling of tcp -x
    R, p, _, _ = fk_tcp(plan["q"])
    a = R[0].T @ UP
    tilt = np.rad2deg(np.arccos(np.clip(np.einsum("nij,j->ni", R, a) @ UP, -1, 1)))
    perr = np.linalg.norm(p - plan["p_ref"], axis=1)
    print(f"verify: max tilt {tilt.max():.4f} deg, max pos err {perr.max():.2e} m")
    # goal pose check
    Rg_err = np.linalg.norm(nk.orientation_error(R[-1:], R_goal))
    print(f"goal: pos err {np.linalg.norm(p[-1] - p_goal):.2e} m, "
          f"orient err {Rg_err:.2e} rad")
