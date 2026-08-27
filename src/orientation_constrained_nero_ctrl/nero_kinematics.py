"""Batched forward kinematics for the AgileX Nero from its URDF.

Frames follow assets/nero/urdf/nero_description.urdf:
    base_link -> (joint1..joint7, each revolute about local Z) -> link7
    link7 -> gripper_flange: fixed, from nero_with_gripper_flange_description.xacro

All FK functions are batched: q has shape (N, 7).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(_HERE, "assets", "nero", "urdf", "nero_description.urdf")

# link7 -> gripper_flange fixed transform (from nero_with_gripper_flange_description.xacro)
FLANGE_XYZ = np.array([0.031, 0.0, -0.0235])
FLANGE_RPY = np.array([-1.5708, 0.0, -1.5708])


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _load_urdf(path: str = URDF_PATH):
    root = ET.parse(path).getroot()
    origins, limits = [], []
    for j in root.iter("joint"):
        if j.get("type") != "revolute":
            continue
        o = j.find("origin")
        xyz = np.fromstring(o.get("xyz") or "0 0 0", sep=" ")
        rpy = np.fromstring(o.get("rpy") or "0 0 0", sep=" ")
        li = j.find("limit")
        origins.append((rpy_matrix(*rpy), xyz))
        limits.append((float(li.get("lower")), float(li.get("upper"))))
    assert len(origins) == 7, f"expected 7 revolute joints, got {len(origins)}"
    return origins, np.array(limits)


_ORIGINS, JOINT_LIMITS = _load_urdf()
R_FLANGE = rpy_matrix(*FLANGE_RPY)


def fk_chain(q: np.ndarray):
    """Batched FK. q: (N, 7) -> (R7, p7, z_axes, p_joints)

    R7:       (N, 3, 3) rotation base->link7
    p7:       (N, 3)    link7 origin in base
    z_axes:   (N, 7, 3) joint rotation axes in base frame (for the Jacobian)
    p_joints: (N, 7, 3) joint origins in base frame
    """
    q = np.atleast_2d(q)
    n = q.shape[0]
    R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    p = np.zeros((n, 3))
    z_axes = np.empty((n, 7, 3))
    p_joints = np.empty((n, 7, 3))
    for i, (R0, t0) in enumerate(_ORIGINS):
        p = p + R @ t0
        R = R @ R0
        p_joints[:, i] = p
        z_axes[:, i] = R[:, :, 2]
        c, s = np.cos(q[:, i]), np.sin(q[:, i])
        Rz = np.zeros((n, 3, 3))
        Rz[:, 0, 0], Rz[:, 0, 1] = c, -s
        Rz[:, 1, 0], Rz[:, 1, 1] = s, c
        Rz[:, 2, 2] = 1.0
        R = R @ Rz
    return R, p, z_axes, p_joints


def fk_flange(q: np.ndarray):
    """Batched flange pose. Returns (R_flange (N,3,3), p_flange (N,3))."""
    R7, p7, _, _ = fk_chain(q)
    return R7 @ R_FLANGE, p7 + (R7 @ FLANGE_XYZ)


def orientation_error(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """Batched orientation error e (N,3): 0.5 * sum_i col_i(R_cur) x col_i(R_tgt)."""
    e = np.zeros((R_cur.shape[0], 3))
    for i in range(3):
        e += np.cross(R_cur[:, :, i], R_tgt[:, i])
    return 0.5 * e


def project_orientation(q0: np.ndarray, R7_target: np.ndarray,
                        iters: int = 80, damping: float = 1e-3,
                        tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Project joint configs onto the manifold R_link7(q) == R7_target.

    Damped least-squares on the 3-DoF orientation error only; positions stay
    free, so converged samples trace out the constant-orientation workspace.
    Returns (q, converged_mask).
    """
    q = q0.copy()
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    active = np.ones(q.shape[0], dtype=bool)
    for _ in range(iters):
        R7, _, z, _ = fk_chain(q[active])
        e = orientation_error(R7, R7_target)
        err2 = np.einsum("ij,ij->i", e, e)
        done = err2 < tol
        # J_w columns are the joint z-axes; solve (J J^T + λI) y = e, dq = J^T y
        J = np.transpose(z, (0, 2, 1))          # (n, 3, 7)
        JJt = J @ np.transpose(J, (0, 2, 1))    # (n, 3, 3)
        JJt += damping * np.eye(3)
        y = np.linalg.solve(JJt, e[:, :, None])  # (n, 3, 1)
        dq = (np.transpose(J, (0, 2, 1)) @ y)[:, :, 0]
        dq = np.clip(dq, -0.3, 0.3)
        idx = np.flatnonzero(active)
        q[idx[~done]] = np.clip(q[idx[~done]] + dq[~done], lo, hi)
        still = active.copy()
        still[idx[done]] = False
        if not still.any():
            break
        active = still
    R7, _, _, _ = fk_chain(q)
    e = orientation_error(R7, R7_target)
    ok = np.einsum("ij,ij->i", e, e) < tol
    return q, ok


def ik_pose(q0: np.ndarray, R7_target: np.ndarray, p_flange_target: np.ndarray,
            iters: int = 150, damping: float = 1e-4,
            pos_tol: float = 1e-3, ori_tol: float = 1e-4):
    """Batched full 6-DoF IK (damped least squares) to a flange pose.

    q0:              (N, 7) initial guesses
    R7_target:       (3, 3) target link7 orientation (shared by all targets)
    p_flange_target: (N, 3) target flange positions
    Returns (q, ok): joint solutions and a convergence mask. Solutions are
    clamped to joint limits every step, so ok==True implies a within-limits
    solution with pos error < pos_tol and orientation error < ori_tol.
    """
    q = np.atleast_2d(q0).copy()
    p_t = np.atleast_2d(p_flange_target)
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    for _ in range(iters):
        R7, p7, z, pj = fk_chain(q)
        p_fl = p7 + (R7 @ FLANGE_XYZ)
        e_p = p_t - p_fl
        e_o = orientation_error(R7, R7_target)
        if (np.einsum("ij,ij->i", e_p, e_p).max() < pos_tol ** 2
                and np.einsum("ij,ij->i", e_o, e_o).max() < ori_tol ** 2):
            break
        # J: 6x7 per sample — position rows z_i x (p_fl - p_i), orientation rows z_i
        arm = p_fl[:, None, :] - pj                      # (n, 7, 3)
        Jp = np.cross(z, arm)                            # (n, 7, 3)
        J = np.concatenate([np.transpose(Jp, (0, 2, 1)),
                            np.transpose(z, (0, 2, 1))], axis=1)  # (n, 6, 7)
        e = np.concatenate([e_p, e_o], axis=1)
        JJt = J @ np.transpose(J, (0, 2, 1)) + damping * np.eye(6)
        y = np.linalg.solve(JJt, e[:, :, None])
        dq = np.clip((np.transpose(J, (0, 2, 1)) @ y)[:, :, 0], -0.3, 0.3)
        q = np.clip(q + dq, lo, hi)
    R7, p7, _, _ = fk_chain(q)
    p_fl = p7 + (R7 @ FLANGE_XYZ)
    e_o = orientation_error(R7, R7_target)
    ok = ((np.einsum("ij,ij->i", (p_t - p_fl), (p_t - p_fl)) < pos_tol ** 2)
          & (np.einsum("ij,ij->i", e_o, e_o) < ori_tol ** 2))
    return q, ok


def sample_constant_orientation_workspace(R7_target: np.ndarray, n_samples: int,
                                          seed: int = 0):
    """Monte-Carlo the constant-orientation workspace for a link7 orientation.

    Returns flange positions (M, 3) and the joint configs (M, 7) that reach
    them with R_link7 == R7_target, all within joint limits.
    """
    rng = np.random.default_rng(seed)
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    q0 = rng.uniform(lo, hi, size=(n_samples, 7))
    q, ok = project_orientation(q0, R7_target)
    q = q[ok]
    _, p_fl = fk_flange(q)
    return p_fl, q
