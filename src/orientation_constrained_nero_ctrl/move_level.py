"""Move a liquid container between two poses, keeping it level the whole way.

Plans with level_path.plan_level_move_auto (5-DoF constrained Cartesian
marching: lift -> horizontal transfer -> lower, container-up locked to world
up = base_link +X, yaw free mid-path), then executes the joint trajectory
through MoveIt, previews it in RViz, or just saves it.

    source /opt/ros/humble/setup.bash
    # no --goal: interactive prompt — type coordinates, plan, preview, exec.
    # SIM by default (reads mock state, refuses to execute if a real arm
    # driver is detected); pass --real to work with the actual robot:
    python3 move_level.py
    python3 move_level.py --real
    # plan from the CURRENT joint state to a goal position, keeping yaw:
    python3 move_level.py --goal -0.31 -0.15 0.55
    # goal with explicit orientation (quaternion, base_link, xyzw):
    python3 move_level.py --goal -0.31 -0.15 0.55 0.507 -0.492 0.492 0.507
    # or with an absolute yaw (deg about world up; 0 = the gripper-down
    # orientation of the arm stretched straight along base +Z):
    python3 move_level.py --goal -0.31 -0.15 0.55 --goal-yaw 45
    # plane-to-plane: ONE straight line from the current pose (no lift);
    # same y z as now = purely vertical height change:
    python3 move_level.py --goal -0.10 -0.15 0.55 --profile straight
    # actions (default is plan + save only):
    python3 move_level.py --goal ... --preview    # RViz Planned Path display
    python3 move_level.py --goal ... --execute    # MoveIt /execute_trajectory

The start is the robot's current config (from --joint-states; default
control/joint_states in sim, feedback/joint_states with --real). Pass
--start "x y z qx qy qz qw" to plan offline from an IK'd start instead
(execution is disabled then, since the robot is not there).

tcp_link = link7 + the tcp_offset given to the MoveIt launch; pass the same
values via --tcp-xyz/--tcp-rpy if you launched with a non-zero offset.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import level_path as lp  # noqa: E402
import nero_kinematics as nk  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402
from moveit_msgs.action import ExecuteTrajectory  # noqa: E402
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory  # noqa: E402

JOINTS = [f"joint{i}" for i in range(1, 8)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def read_current_joints(node: Node, topic: str, timeout: float = 5.0):
    """Block for one JointState containing joint1..joint7."""
    got = {}

    def cb(msg):
        for n, p in zip(msg.name, msg.position):
            got[n] = p

    sub = node.create_subscription(JointState, topic, cb, 10)
    end = node.get_clock().now().nanoseconds + int(timeout * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if all(j in got for j in JOINTS):
            node.destroy_subscription(sub)
            return np.array([got[j] for j in JOINTS])
    node.destroy_subscription(sub)
    return None


def to_joint_trajectory(q, t, v=None, acc=None):
    jt = JointTrajectory()
    jt.joint_names = JOINTS
    if v is None:
        v = np.gradient(q, t, axis=0) if len(q) > 2 else np.zeros_like(q)
        v[0] = v[-1] = 0.0
    for i, (qi, vi, ti) in enumerate(zip(q, v, t)):
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in qi]
        pt.velocities = [float(x) for x in vi]
        if acc is not None:
            pt.accelerations = [float(x) for x in acc[i]]
        pt.time_from_start.sec = int(ti)
        pt.time_from_start.nanosec = int((ti % 1.0) * 1e9)
        jt.points.append(pt)
    return jt


def execute_moveit(node: Node, jt: JointTrajectory) -> bool:
    client = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")
    node.get_logger().info("waiting for /execute_trajectory ...")
    if not client.wait_for_server(timeout_sec=10.0):
        node.get_logger().error("MoveIt /execute_trajectory not available")
        return False
    goal = ExecuteTrajectory.Goal()
    goal.trajectory.joint_trajectory = jt
    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut)
    handle = fut.result()
    if handle is None or not handle.accepted:
        node.get_logger().error("execution rejected")
        return False
    node.get_logger().info("executing ...")
    res = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res)
    code = res.result().result.error_code.val
    node.get_logger().info(f"done (error_code={code})")
    return code == 1


def preview(node: Node, q, t, v=None, acc=None):
    """Send the plan to RViz's MotionPlanning display via display_planned_path.

    The "Planned Path" sub-display animates it once (tick Loop Animation to
    repeat). This works identically in sim and with --real, and doesn't fight
    whoever publishes the live joint states.
    """
    pub = node.create_publisher(DisplayTrajectory, "display_planned_path", 1)
    msg = DisplayTrajectory()
    rt = RobotTrajectory()
    rt.joint_trajectory = to_joint_trajectory(q, t, v, acc)
    msg.trajectory.append(rt)
    msg.trajectory_start.joint_state.name = JOINTS
    msg.trajectory_start.joint_state.position = [float(x) for x in q[0]]
    end = node.get_clock().now().nanoseconds + int(3e9)
    while (rclpy.ok() and pub.get_subscription_count() == 0
           and node.get_clock().now().nanoseconds < end):
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print("nobody subscribes to display_planned_path — is RViz with the "
              "MotionPlanning display running?")
    pub.publish(msg)
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_publisher(pub)
    print(f"sent {t[-1]:.1f} s trajectory to RViz — the Planned Path display "
          "animates it (MotionPlanning > Planned Path > Loop Animation to "
          "repeat)")


def exec_allowed(node: Node, args) -> bool:
    """In sim mode, refuse to execute when a real arm driver is up."""
    if args.real:
        return True
    if node.count_publishers("feedback/joint_states") > 0:
        print("a real arm driver is publishing feedback/joint_states — "
              "refusing to execute in sim mode (rerun with --real)")
        return False
    return True


def plan_carry(qj, p_tgt, qdot_max, R_t, t_t):
    """Joint-interpolated reorientation to the carrying pose (gripper down,
    container-up = tcp -x). p_tgt None = reorient in place. NOT level-
    constrained — never do this while holding liquid. Returns
    (qs, ts, tilt0, dq) or None when the pose has no IK solution."""
    R_cur, p_cur, _, _ = lp.fk_tcp(qj[None], R_t, t_t)
    R_cur, p_cur = R_cur[0], p_cur[0]
    if p_tgt is None:
        p_tgt = p_cur
    # the tool axis is tcp +x (at home it points along the arm);
    # carrying = gripper down, so -x_tcp must point up
    a = np.array([-1.0, 0.0, 0.0])
    if (R_cur @ a) @ lp.UP < -0.999:          # antiparallel: level() has
        R_cur = R_cur @ lp.rot_axis(           # no unique axis — pre-tip
            np.array([0.0, 1.0, 0.0]), np.pi / 2)   # 90 deg about y
    R_tgt, tilt0 = lp.level(R_cur, a)
    q_goal, ok = lp.ik_tcp(R_tgt, np.asarray(p_tgt, float), q_seed=qj,
                           R_t=R_t, t_t=t_t)
    if not ok:
        return None
    qs, ts, dq = _joint_interp(qj, q_goal, qdot_max)
    return qs, ts, tilt0, dq


def _joint_interp(qj, q_goal, qdot_max):
    """Straight joint-space interpolation qj -> q_goal at <= qdot_max/2,
    never faster than 2 s. Returns (qs, ts, dq_max)."""
    dq = float(np.abs(q_goal - qj).max())
    T = max(2.0, 2.0 * dq / qdot_max)
    n = max(2, int(T * 10))
    s = np.linspace(0.0, 1.0, n)[:, None]
    qs = qj[None] * (1 - s) + q_goal[None] * s
    ts = np.linspace(0.0, T, n)
    return qs, ts, dq


SAGITTAL_FREE = (1, 3, 6)          # joint2, joint4, joint7 (0-based indices)


def plan_sagittal_carry(qj, x_plane, z_tgt, qdot_max, R_t, t_t):
    """Joint-interpolated move to the canonical in-plane carry pose: joints
    1/3/5/6 at zero (the arm folded in the base X-Z plane, no sideways
    yaw), joints 2/4/7 solved so the tcp sits at height x_plane, at z_tgt
    along base +Z, with the gripper pointing straight down (level carry
    orientation, container-up = tcp -x). NOT level-constrained on the way —
    only the final pose is level. Returns (qs, ts, q_goal, p_goal, dq) or
    None when no in-limits solution reaches (x_plane, z_tgt)."""
    from scipy.optimize import least_squares
    lo, hi = nk.JOINT_LIMITS[:, 0], nk.JOINT_LIMITS[:, 1]
    free = list(SAGITTAL_FREE)
    a = np.array([-1.0, 0.0, 0.0])            # container-up in the tcp frame

    def resid(v):
        q = np.zeros(7)
        q[free] = v
        R, p, _, _ = lp.fk_tcp(q[None], R_t, t_t)
        up = R[0] @ a
        return [p[0][0] - x_plane, p[0][2] - z_tgt, up[1], up[2]]

    sols = []
    for j2 in np.linspace(lo[1], hi[1], 6):
        for j4 in np.linspace(lo[3], hi[3], 6):
            j7 = np.clip(np.pi / 2 - j2 - j4, lo[6], hi[6])
            r = least_squares(resid, [j2, j4, j7],
                              bounds=(lo[free], hi[free]))
            if r.cost < 1e-9 and not any(
                    np.abs(r.x - s).max() < 1e-3 for s in sols):
                sols.append(r.x)
    if not sols:
        return None
    # several elbow branches may reach: take the one nearest the current pose
    v = min(sols, key=lambda s: np.abs(s - qj[free]).max())
    q_goal = np.zeros(7)
    q_goal[free] = v
    _, p_goal, _, _ = lp.fk_tcp(q_goal[None], R_t, t_t)
    qs, ts, dq = _joint_interp(qj, q_goal, qdot_max)
    return qs, ts, q_goal, p_goal[0], dq


def make_plan(node: Node, q_start, p_goal, R_goal, args, profile=None):
    """Plan, time-parameterize, log and save. Returns (q, t, qd, qdd) or
    None; qd/qdd are None on the trapezoidal fallback path.

    profile: "lift" (lift -> horizontal transfer -> lower; the in-plane
    move) or "straight" (one straight line start -> goal, no lift; the
    plane-to-plane move). None = args.profile (default "lift").

    Default timing (args.dwell >= 0): every straight segment (lift, transfer,
    lower) is retimed on its own with zero velocity at both ends, so the arm
    ramps down to a full stop at each corner, holds args.dwell seconds and
    ramps up again. With a negative dwell the whole path is retimed in one go
    (TOPP-RA) and the corners are rounded at speed."""
    if profile is None:
        profile = getattr(args, "profile", "lift")
    lp.CANCEL.clear()
    try:
        plan = lp.plan_level_move_auto(
            q_start, p_goal, R_goal, lift=args.lift, step=args.step,
            profile=profile,
            tcp_xyz=tuple(args.tcp_xyz), tcp_rpy=tuple(args.tcp_rpy))
    except lp.PlanCancelled:
        node.get_logger().warn("planning cancelled by the user")
        print("planning cancelled — no plan")
        return None
    if not plan["ok"]:
        node.get_logger().error("planning FAILED — see messages above")
        return None
    dwell = float(getattr(args, "dwell", 0.1))
    if dwell >= 0.0:
        t, q, qd, qdd, all_toppra = lp.retime_corner_stops(
            plan["q"], plan["s"], plan["s_corners"], dwell=dwell,
            v_max=args.v, a_max=args.a, qdot_max=args.qdot,
            qddot_max=args.qddot)
        n_stop = len(lp.corner_indices(plan["s"], plan["s_corners"]))
        if not all_toppra:
            node.get_logger().warn(
                "toppra unavailable/failed — trapezoidal timing")
        timing = (f"{'toppra' if all_toppra else 'trapezoidal'} per segment, "
                  f"{n_stop} corner stop(s) of {dwell:.2f} s, max joint rate "
                  f"{np.abs(qd).max():.2f} rad/s")
    else:
        retimed = lp.retime_toppra(plan["q"], plan["s"], v_max=args.v,
                                   a_max=args.a, qdot_max=args.qdot,
                                   qddot_max=args.qddot)
        if retimed is not None:
            t, q, qd, qdd = retimed
            timing = f"toppra, max joint rate {np.abs(qd).max():.2f} rad/s"
        else:
            node.get_logger().warn(
                "toppra unavailable/failed — trapezoidal timing")
            q = plan["q"]
            t = lp.time_parameterize(plan["s"], q, v_max=args.v,
                                     a_max=args.a, qdot_max=args.qdot)
            qd = qdd = None
            timing = "trapezoidal"
    if len(q) != len(plan["q"]):
        # verify the resampled spline still keeps the container level
        R_t, t_t = lp.tcp_transform(args.tcp_xyz, args.tcp_rpy)
        R, _, _, _ = lp.fk_tcp(q, R_t, t_t)
        a_up = R[0].T @ lp.UP
        tilt = np.arccos(np.clip(
            np.einsum("nij,j->ni", R, a_up) @ lp.UP, -1.0, 1.0))
        timing += f", sampled-path tilt {np.rad2deg(tilt.max()):.3f} deg"
    if profile == "straight":
        b = plan.get("bulge", 0.0)
        shape = ("straight line start->goal (no lift)" if abs(b) < 1e-6 else
                 f"arc start->goal, sagitta {abs(b):.2f} m toward "
                 f"{lp._dir_label(b, plan.get('bulge_dir', 'side'))} (no lift)")
    else:
        shape = (("straight transfer" if abs(plan.get("bulge", 0.0)) < 1e-6
                  else f"arc transfer, sagitta {plan['bulge']:+.2f} m")
                 + f", lift {plan['lift']:.3f} m")
    node.get_logger().info(
        f"plan: {len(plan['q'])} waypoints, {t[-1]:.1f} s ({timing}), "
        f"{shape}, max tilt "
        f"{np.rad2deg(max(plan['tilt_err'].max(), plan['tilt_mid'].max())):.3f} deg")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(args.save, q=q, t=t, bulge=plan.get("bulge", 0.0),
                        bulge_dir=str(plan.get("bulge_dir", "side")),
                        profile=profile,
                        q_wp=plan["q"], s_wp=plan["s"], p_ref=plan["p_ref"],
                        yaw_ref=plan["yaw_ref"], tilt=plan["tilt_err"])
    node.get_logger().info(f"saved {args.save}")
    return q, t, qd, qdd


INTERACTIVE_HELP = """\
goal (plans from the CURRENT joint state, container kept level):
  x y z                  keep the current yaw          e.g.  -0.31 -0.15 0.55
  x y z yaw              absolute yaw [deg], 0 = straight-ahead (+z) pose
                                                       e.g.  -0.31 -0.15 0.55 45
  x y z qx qy qz qw      explicit orientation (base_link, xyzw)
commands:
  where                  print the current TCP pose
  carry [x y z]          point the gripper DOWN (base -X) — setup move,
                         NOT level-constrained; do this before grabbing
  preview  (p)           show the last plan in RViz (Planned Path display)
  exec     (go)          execute the last plan on the robot (asks first)
  set NAME VALUE         tune lift/step/v/a/qdot/qddot/dwell e.g. set lift 0.1
                         (dwell = hold at each corner [s]; set dwell -1 to
                         run through the corners without stopping)
  profile [lift|straight]
                         path shape for the next goals: lift = lift ->
                         horizontal transfer -> lower (in-plane move),
                         straight = one straight line start -> goal, no lift
                         (plane-to-plane move; keep y z = purely vertical)
  help     (?)           show this text
  quit     (q)           leave"""


def interactive(node: Node, args):
    """Prompt loop: type coordinates -> plan -> preview/exec, repeat."""
    R_t, t_t = lp.tcp_transform(args.tcp_xyz, args.tcp_rpy)
    last = None
    print(INTERACTIVE_HELP)
    while True:
        try:
            line = input("level> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        toks = line.split()
        cmd = toks[0].lower()

        if cmd in ("q", "quit", "exit"):
            return
        if cmd in ("h", "help", "?"):
            print(INTERACTIVE_HELP)
            continue
        if cmd == "where":
            qj = read_current_joints(node, args.joint_states)
            if qj is None:
                print(f"no JointState on '{args.joint_states}'")
                continue
            R, p, _, _ = lp.fk_tcp(qj[None], R_t, t_t)
            print(f"tcp pos: {np.round(p[0], 4).tolist()}  "
                  f"yaw {np.rad2deg(lp.abs_yaw(R[0])):+.1f} deg  "
                  f"quat(xyzw): {np.round(lp.mat_to_quat(R[0]), 4).tolist()}")
            continue
        if cmd == "carry":
            if len(toks) not in (1, 4):
                print("carry takes no args (reorient in place) or x y z")
                continue
            qj = read_current_joints(node, args.joint_states)
            if qj is None:
                print(f"no JointState on '{args.joint_states}'")
                continue
            p_tgt = np.array([float(x) for x in toks[1:]]) if len(toks) == 4 \
                else None
            carry = plan_carry(qj, p_tgt, args.qdot, R_t, t_t)
            if carry is None:
                print("no IK solution for the carry pose there — the spot "
                      "may be out of reach with the gripper pointing down; "
                      "try 'carry x y z' somewhere closer to the base")
                continue
            qs, ts, tilt0, dq = carry
            print(f"carry: reorient {np.rad2deg(tilt0):.1f} deg, largest "
                  f"joint move {np.rad2deg(dq):.1f} deg, {ts[-1]:.1f} s "
                  "(unconstrained joint move — don't do this holding liquid)")
            if not exec_allowed(node, args):
                continue
            try:
                mode = "REAL arm" if args.real else "sim"
                sure = input(f"execute carry ({mode})? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if sure in ("y", "yes"):
                execute_moveit(node, to_joint_trajectory(qs, ts))
                last = None
            continue
        if cmd == "set" and len(toks) == 3:
            name = toks[1].lower()
            if name not in ("lift", "step", "v", "a", "qdot", "qddot",
                            "dwell"):
                print("set takes: lift step v a qdot qddot dwell")
                continue
            try:
                setattr(args, name, float(toks[2]))
                print(f"{name} = {getattr(args, name)}")
            except ValueError:
                print(f"not a number: {toks[2]}")
            continue
        if cmd == "profile":
            if len(toks) == 1:
                print(f"profile = {args.profile}")
            elif len(toks) == 2 and toks[1].lower() in lp.PROFILES:
                args.profile = toks[1].lower()
                print(f"profile = {args.profile}")
            else:
                print(f"profile takes one of {' | '.join(lp.PROFILES)}")
            continue
        if cmd in ("p", "preview"):
            if last is None:
                print("no plan yet — type a goal first")
                continue
            preview(node, *last)
            continue
        if cmd in ("e", "exec", "go"):
            if last is None:
                print("no plan yet — type a goal first")
                continue
            if not exec_allowed(node, args):
                continue
            try:
                mode = "REAL arm" if args.real else "sim"
                sure = input(f"execute ({mode})? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if sure in ("y", "yes"):
                execute_moveit(node, to_joint_trajectory(*last))
                last = None    # robot moved: old plan's start is stale
            continue

        # anything else must be a goal: 3, 4 or 7 numbers
        try:
            vals = [float(x) for x in toks]
        except ValueError:
            print("unknown command (try 'help')")
            continue
        if len(vals) not in (3, 4, 7):
            print("goal takes 3 (x y z), 4 (x y z yaw_deg) or "
                  "7 (x y z + quat) numbers")
            continue
        q_start = read_current_joints(node, args.joint_states)
        if q_start is None:
            print(f"no JointState on '{args.joint_states}' — driver running?")
            continue
        p_goal = np.array(vals[:3])
        if len(vals) == 7:
            R_goal = lp.quat_to_mat(vals[3:])
        elif len(vals) == 4:
            R_cur, _, _, _ = lp.fk_tcp(q_start[None], R_t, t_t)
            R_goal = lp.orientation_at_yaw(R_cur[0], np.deg2rad(vals[3]))
        else:
            R_goal = None
        last = make_plan(node, q_start, p_goal, R_goal, args)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", type=float, nargs="+", default=None,
                    metavar="V", help="x y z [qx qy qz qw] in base_link; "
                    "omit to enter the interactive prompt")
    ap.add_argument("--goal-yaw", type=float, default=None,
                    help="absolute goal yaw [deg about world up; 0 = the "
                         "straight-ahead +z pose orientation]; "
                         "ignored if --goal has a quaternion")
    ap.add_argument("--start", type=float, nargs=7, default=None,
                    metavar="V", help="x y z qx qy qz qw — plan offline from "
                    "this pose instead of the current joint state")
    ap.add_argument("--lift", type=float, default=0.08)
    ap.add_argument("--profile", choices=lp.PROFILES, default="lift",
                    help="path shape: lift = lift -> horizontal transfer -> "
                         "lower (in-plane move, default); straight = one "
                         "straight line start -> goal with no lift (plane-"
                         "to-plane move; same y z = purely vertical)")
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--v", type=float, default=0.06, help="max speed [m/s]")
    ap.add_argument("--a", type=float, default=0.2, help="max accel [m/s^2]")
    ap.add_argument("--qdot", type=float, default=1.0,
                    help="max joint rate [rad/s]")
    ap.add_argument("--qddot", type=float, default=3.0,
                    help="max joint acceleration [rad/s^2] (toppra timing)")
    ap.add_argument("--dwell", type=float, default=0.1,
                    help="hold time at each path corner [s] (default 0.1); "
                         "negative = run through the corners without "
                         "stopping (single TOPP-RA pass)")
    ap.add_argument("--tcp-xyz", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--tcp-rpy", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--real", action="store_true",
                    help="work with the REAL arm: read feedback/joint_states "
                         "and allow execution while the driver is up. Without "
                         "this, sim mode: reads the mock state and refuses to "
                         "execute if a real driver is detected.")
    ap.add_argument("--joint-states", default=None,
                    help="current-state topic (default: control/joint_states "
                         "in sim, feedback/joint_states with --real)")
    ap.add_argument("--execute", action="store_true",
                    help="execute via MoveIt /execute_trajectory")
    ap.add_argument("--preview", action="store_true",
                    help="send the plan to RViz's Planned Path display "
                         "instead of executing")
    ap.add_argument("--save", default=os.path.join(OUT, "level_move.npz"))
    args = ap.parse_args()

    if args.joint_states is None:
        args.joint_states = ("feedback/joint_states" if args.real
                             else "control/joint_states")

    if args.goal is not None and len(args.goal) not in (3, 7):
        ap.error("--goal takes 3 (position) or 7 (position + quaternion) values")
    if args.goal is None and args.start is not None:
        ap.error("--start needs --goal (interactive mode plans from the "
                 "current joint state)")

    rclpy.init()
    node = Node("nero_level_move")

    if args.goal is None:
        try:
            interactive(node, args)
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    # --- start config --------------------------------------------------------
    if args.start is not None:
        R_start = lp.quat_to_mat(args.start[3:])
        R_t, t_t = lp.tcp_transform(args.tcp_xyz, args.tcp_rpy)
        q_start, ok = lp.ik_tcp(R_start, np.array(args.start[:3]),
                                R_t=R_t, t_t=t_t)
        if not ok:
            node.get_logger().error("no IK solution for --start pose")
            return
        if args.execute:
            node.get_logger().warn("--start given: disabling --execute "
                                   "(robot is not at the planned start)")
            args.execute = False
    else:
        q_start = read_current_joints(node, args.joint_states)
        if q_start is None:
            node.get_logger().error(
                f"no JointState on '{args.joint_states}' — is the driver "
                "(or mock) running? Or pass --start for offline planning.")
            return
    node.get_logger().info(f"start config: {np.round(q_start, 3).tolist()}")

    # --- goal pose -----------------------------------------------------------
    p_goal = np.array(args.goal[:3])
    if len(args.goal) == 7:
        R_goal = lp.quat_to_mat(args.goal[3:])
    elif args.goal_yaw is not None:
        R_cur, _, _, _ = lp.fk_tcp(q_start[None],
                                   *lp.tcp_transform(args.tcp_xyz, args.tcp_rpy))
        R_goal = lp.orientation_at_yaw(R_cur[0], np.deg2rad(args.goal_yaw))
    else:
        R_goal = None      # keep the start orientation

    # --- plan ----------------------------------------------------------------
    planned = make_plan(node, q_start, p_goal, R_goal, args)
    if planned is None:
        return
    q_path, t, qd, qdd = planned

    if args.execute:
        if exec_allowed(node, args):
            execute_moveit(node, to_joint_trajectory(q_path, t, qd, qdd))
    elif args.preview:
        preview(node, q_path, t, qd, qdd)
    else:
        node.get_logger().info("plan-only mode (use --execute or --preview)")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
