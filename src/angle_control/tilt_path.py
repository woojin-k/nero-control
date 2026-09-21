"""Carrying the container at a fixed angle to gravity.

orientation_constrained_nero_ctrl/level_path.py keeps the container level:
its up axis (tcp -X) stays aligned with world up while the tcp moves, and
yaw about world up is free / commanded. This module generalises the carried
ATTITUDE: the container is held tilted by a fixed angle theta from gravity,
leaning toward one of its own sides (default: its FRONT), and the level
planner is then used unchanged — it keeps whatever tcp axis points up at the
start pointing up, so a tilted start pose is carried tilted, and the lean
direction turns with the commanded yaw exactly like the container itself.

Conventions (base_link, sideways mount — see level_path.py):
  UP             world up = base +X
  container-up   tcp -X (gripper pointing down = the carry pose)
  yaw            level_path's absolute yaw about UP; 0 = the arm stretched
                 straight along base +Z ("ahead"), gripper down
  lean sides     in the tcp frame at yaw 0: front = ahead (+Z base) = tcp -Y,
                 back = tcp +Y, left = tcp +Z, right = tcp -Z

For tilt theta toward side e (tcp frame), the tcp-frame vector that points
straight up is
    a_theta = cos(theta) * (-X) - sin(theta) * e
and the constraint carried along every path is R(q) @ a_theta == UP.

plan_tilt() rotates the container IN PLACE (about the tcp point, yaw held)
from its current attitude to the tilt-theta attitude (theta = 0 brings it
back to level). carry_orientation() gives the tilted carry orientation for a
setup move from an arbitrary pose.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEVEL_DIR = os.path.join(os.path.dirname(_HERE),
                          "orientation_constrained_nero_ctrl")
if _LEVEL_DIR not in sys.path:
    sys.path.insert(0, _LEVEL_DIR)
import level_path as lp  # noqa: E402

UP = lp.UP
CONTAINER_UP = np.array([-1.0, 0.0, 0.0])   # container-up axis, tcp frame
LEAN = {                                    # container side, tcp frame
    "front": np.array([0.0, -1.0, 0.0]),
    "back": np.array([0.0, 1.0, 0.0]),
    "left": np.array([0.0, 0.0, 1.0]),
    "right": np.array([0.0, 0.0, -1.0]),
}
LEAN_KO = {"front": "앞", "back": "뒤", "left": "왼쪽", "right": "오른쪽"}
LEVEL_EPS = np.deg2rad(0.1)                 # below this: level / no direction


def lean_axis(lean):
    if isinstance(lean, str):
        return LEAN[lean]
    e = np.asarray(lean, float)
    return e / np.linalg.norm(e)


def up_axis(theta, lean="front"):
    """tcp-frame vector that must point to world up for tilt `theta` [rad]
    toward container side `lean`."""
    e = lean_axis(lean)
    return np.cos(theta) * CONTAINER_UP - np.sin(theta) * e


def tilt_state(R, u=UP):
    """(theta [rad], lean direction in the world, horizontal unit vector or
    None when level) of orientation R: the angle between the container-up
    axis and world up, and the direction it leans toward."""
    b = np.asarray(R, float) @ CONTAINER_UP
    theta = float(np.arccos(np.clip(b @ u, -1.0, 1.0)))
    h = b - (b @ u) * u
    n = np.linalg.norm(h)
    if n < 1e-6 or theta < LEVEL_EPS:
        return theta, None
    return theta, h / n


def tilt_orientation(R_cur, theta, lean="front", u=UP):
    """Orientation with tilt `theta` toward `lean`, at the absolute yaw of
    R_cur (R_cur must be a carry attitude: level or tilted)."""
    psi = lp.abs_yaw(np.asarray(R_cur, float), u)
    R_ref, _ = lp.level(lp.YAW_REF, up_axis(theta, lean), u)
    return lp.rot_axis(u, psi) @ R_ref


def carry_orientation(R_cur, theta, lean="front", u=UP):
    """Tilted carry orientation for a setup move from ANY pose: first the
    minimal rotation that points the gripper down (container-up == up, as
    move_level's carry), then the tilt at that yaw."""
    R_lvl, _ = lp.level(np.asarray(R_cur, float), CONTAINER_UP, u)
    return tilt_orientation(R_lvl, theta, lean, u)


def lean_world(yaw, lean="front", u=UP):
    """World direction the container leans toward at absolute yaw `yaw`
    [rad] (horizontal unit vector) — for drawing."""
    return lp.rot_axis(u, float(yaw)) @ (lp.YAW_REF @ lean_axis(lean))


def azimuth_of(d, u=UP):
    """Azimuth [deg] about u of horizontal direction d, 0 = base +Z (ahead),
    right-handed — for display only."""
    e1 = np.array([0.0, 0.0, 1.0])
    e2 = np.cross(u, e1)
    return float(np.rad2deg(np.arctan2(d @ e2, d @ e1)))


# --- planning -----------------------------------------------------------------

def plan_tilt(q_start, theta, lean="front", *, omega_max=np.deg2rad(10.0),
              dt=0.1, t_min=2.0, qdot_max=1.0, tcp_xyz=(0, 0, 0),
              tcp_rpy=(0, 0, 0), pos_tol=5e-4, ang_tol=2e-4, u=UP,
              verbose=True):
    """In-place rotation of the container to tilt `theta` [rad] toward its
    side `lean`, holding the tcp position and the absolute yaw. theta = 0
    returns to level. Smoothstep angle profile, peak rate omega_max.

    Returns dict: q (N,7), t (N,), theta (N,) tilt of each sample, pos_err,
    ang_err (N,), ok, fail_index, dtheta (total rotation [rad])."""
    lp.CANCEL.clear()
    R_t, t_t = lp.tcp_transform(tcp_xyz, tcp_rpy)
    u = np.asarray(u, float) / np.linalg.norm(u)
    q_start = np.asarray(q_start, float)
    R_s, p_s, _, _ = lp.fk_tcp(q_start[None], R_t, t_t)
    R_s, p_s = R_s[0], p_s[0]
    theta0, _ = tilt_state(R_s, u)

    R_g = tilt_orientation(R_s, theta, lean, u)
    axis, dtheta = lp._log_rot(R_g @ R_s.T)
    if verbose:
        side = LEAN_KO.get(lean, str(lean)) if isinstance(lean, str) else lean
        print(f"tilt: {np.rad2deg(theta0):.2f} deg -> {np.rad2deg(theta):.2f} "
              f"deg toward {side} (rotation {np.rad2deg(dtheta):.2f} deg, "
              f"yaw {np.rad2deg(lp.abs_yaw(R_s, u)):+.1f} deg held, tcp held "
              f"at {np.round(p_s, 4).tolist()})")
    if dtheta < LEVEL_EPS:
        return dict(q=q_start[None], t=np.zeros(1), theta=np.array([theta0]),
                    pos_err=np.zeros(1), ang_err=np.zeros(1), ok=True,
                    fail_index=None, dtheta=0.0)

    T = max(float(t_min), 1.5 * dtheta / float(omega_max))
    t = np.arange(0.0, T, dt)
    if T - t[-1] > 1e-9:
        t = np.append(t, T)
    s = lp._smoothstep(t / T)

    q_path = np.empty((len(t), 7))
    q_path[0] = q_start
    pos_err = np.zeros(len(t))
    ang_err = np.zeros(len(t))
    q_prev = q_start[None].copy()
    fail = None
    for k in range(1, len(t)):
        R_k = lp.rot_axis(axis, dtheta * s[k]) @ R_s
        q_sol, pe, _, _ = lp.solve_pose(
            q_prev, p_s[None], R_k, w_yaw=1.0, u=u, iters=80,
            R_t=R_t, t_t=t_t, pos_tol=pos_tol, ang_tol=ang_tol)
        R_c, _, _, _ = lp.fk_tcp(q_sol, R_t, t_t)
        e_full = np.linalg.norm(lp.nk.orientation_error(R_c, R_k)[0])
        q_path[k] = q_sol[0]
        pos_err[k] = pe[0]
        ang_err[k] = e_full
        if pe[0] > pos_tol or e_full > ang_tol:
            fail = k
            break
        q_prev = q_sol

    if fail is not None:
        if verbose:
            print(f"tilt FAILED at sample {fail}/{len(t) - 1}: pos err "
                  f"{pos_err[fail]:.2e} m, orientation err {ang_err[fail]:.2e} "
                  f"rad — joint limit or reach; try another spot / yaw")
        return dict(q=q_path[:fail], t=t[:fail], theta=None,
                    pos_err=pos_err[:fail], ang_err=ang_err[:fail], ok=False,
                    fail_index=fail, dtheta=dtheta)

    qd = np.gradient(q_path, t, axis=0)
    rate = np.abs(qd).max()
    if rate > qdot_max:
        t = t * (rate / qdot_max)
        if verbose:
            print(f"joint rate {rate:.2f} rad/s > {qdot_max} — slowed to "
                  f"{t[-1]:.1f} s")
    R_all, _, _, _ = lp.fk_tcp(q_path, R_t, t_t)
    theta_all = np.arccos(np.clip(
        np.einsum("nij,j->ni", R_all, CONTAINER_UP) @ u, -1.0, 1.0))
    if verbose:
        print(f"solved {len(q_path)} samples over {t[-1]:.1f} s: max pos err "
              f"{pos_err.max():.2e} m, max orientation err {ang_err.max():.2e} "
              f"rad, final tilt {np.rad2deg(theta_all[-1]):.3f} deg")
    return dict(q=q_path, t=t, theta=theta_all, pos_err=pos_err,
                ang_err=ang_err, ok=True, fail_index=None, dtheta=dtheta)


OUT = os.path.join(_HERE, "out")


def tilt_workspace(theta, lean="front", n_samples=200_000, seed=0,
                   tcp_xyz=(0, 0, 0), tcp_rpy=(0, 0, 0), u=UP, verbose=True):
    """Monte-Carlo reachable workspace for the TILTED carry attitude at
    yaw 0 (the same construction as reachable_workspace.py's B_link7 set,
    which is the level attitude): joint configs with R_link7 == the tilted
    reference orientation, all within joint limits, and their tcp
    positions. Cached per (theta, lean) in out/. Returns (p (M,3), q (M,7))
    or None when the sampler finds nothing."""
    import nero_kinematics as nk
    os.makedirs(OUT, exist_ok=True)
    tag = f"{np.rad2deg(theta):.1f}_{lean}"
    path = os.path.join(OUT, f"workspace_tilt_{tag}.npz")
    R_t, t_t = lp.tcp_transform(tcp_xyz, tcp_rpy)
    if os.path.exists(path):
        d = np.load(path)
        q = d["q"].astype(float)
        _, p, _, _ = lp.fk_tcp(q, R_t, t_t)
        return p, q
    if verbose:
        print(f"도달영역 계산 중 (기울기 {np.rad2deg(theta):g}° {lean}, "
              f"{n_samples // 1000}k 샘플)… 처음 한 번만, 이후 캐시 사용")
    R7, _ = lp.level(lp.YAW_REF, up_axis(theta, lean), u)
    _, q = nk.sample_constant_orientation_workspace(R7, n_samples, seed=seed)
    if len(q) == 0:
        return None
    np.savez_compressed(path, q=q.astype(np.float32), theta=float(theta),
                        lean=str(lean))
    _, p, _, _ = lp.fk_tcp(q, R_t, t_t)
    if verbose:
        print(f"도달영역 {len(q)} 샘플 — {path}")
    return p, q


def plan_carry_tilted(q_start, theta, lean="front", *, qdot_max=1.0,
                      tcp_xyz=(0, 0, 0), tcp_rpy=(0, 0, 0), u=UP):
    """Setup move (joint interpolation, NOT attitude-constrained) from any
    pose to the tilted carry attitude in place. Returns (q (N,7), t (N,),
    reorientation angle [rad], largest joint move [rad]) or None."""
    import move_level as ml
    R_t, t_t = lp.tcp_transform(tcp_xyz, tcp_rpy)
    q_start = np.asarray(q_start, float)
    R_s, p_s, _, _ = lp.fk_tcp(q_start[None], R_t, t_t)
    R_g = carry_orientation(R_s[0], theta, lean, u)
    _, ang = lp._log_rot(R_g @ R_s[0].T)
    q_goal, ok = lp.ik_tcp(R_g, p_s[0], q_seed=q_start, R_t=R_t, t_t=t_t)
    if not ok:
        return None
    qs, ts, dq = ml._joint_interp(q_start, q_goal, qdot_max)
    return qs, ts, ang, dq
