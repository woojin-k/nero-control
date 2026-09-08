#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tkinter GUI for level-constrained container moves — a shell over move_level.

Five horizontal work planes plane_1..plane_5 (world up = base_link +X, so a
plane is a fixed x; plane_1 is the LOWEST, heights strictly increasing). Pick
a plane, then either type the in-plane coordinates (y, z) or click them on
the top-down plane map (double-click = plan immediately); the map shades the
reachable region of the selected plane from the Monte-Carlo workspace study
(out/workspace_B_link7.npz) and shows the base origin, live TCP, saved
points and the pending goal. Yaw change is optional.

Points (포인트): a named point stores its plane, the plane height x at save
time, y, z and yaw. Select one in the list, pick a move mode and press
"포인트로 플랜" (or double-click to only load it into the fields): the plan
goes from the CURRENT pose to that point with the chosen mode — so a
point-to-point move is "be at point A, plan to point B". The plane heights
are the source of truth: a point whose saved x no longer matches its plane's
height is planned at the plane's current height (the log says so). Points
and plane heights persist in src/data/level_targets.json.

Move modes (이동 방식) — the container stays level in all of them:
  평면 내 이동   lift -> horizontal transfer -> lower ON the selected plane
                (level_path profile "lift"). Meant for a TCP already on that
                plane; if it is not, the log says so and the path still lifts
                above the higher endpoint — use a plane-change mode instead.
  평면 간 직선   plane change: ONE straight line from the current TCP to
                (plane x, y, z), no lift (profile "straight").
  평면 간 수직   plane change with y, z fixed to the current TCP: only the
                height changes (profile "straight", y/z fields ignored).

Teaching (--real only): a "티칭" panel talks to the arm DIRECTLY over CAN
with pyAgxArm (src/nero_ctrl mode_control / connect_check — the same code
the older nero_gui uses), alongside the ROS driver on the same socketcan
bus. "Leader" enables the motors and drops the arm into zero-force drag so
you can move it by hand (SUPPORT IT — it sags); "Follower" puts it back
under control. "Leader 캡처" is the one-button teach flow from nero_gui:
follower (arm holds) -> read the measured joints -> FK -> save a point
(nearest plane, measured x, y, z, absolute yaw, joints) -> leader again.
While the arm is a LEADER the driver publishes no feedback/joint_states, so
the TCP readout switches to the leader broadcast (marked) and every motion
button (go / carry / +z / gripper) is refused.

Joint zero calibration (same panel, second row): the firmware only offers
"set the CURRENT motor position as zero" (CAN 0x475 / SDK calibrate_joint),
no self-homing — so it is semi-automatic: align the joints to their
mechanical zero marks by hand in LEADER, switch to FOLLOWER (the GUI reports
how much each joint shifted when the controller took hold), fine-jog the
selected joint (+-0.2/1 deg via move_j at 5 %) until the mark lines up
again, then "영점 설정": disable -> calibrate_joint -> enable, one joint or
all (255), with the readings before/after logged. The readings before every
zero-set are appended to src/data/joint_zero_log.json: the old zero sits at
minus those values in the new frame, which is how a mistaken zero-set can
be undone (move there, zero again).

Gripper: a small panel shows the live AGX gripper state (feedback/
gripper_status: width, force, enable/homing/error bits) and has open / close /
move-to buttons. Commands go to MoveIt's gripper_controller
(gripper_controller/follow_joint_trajectory, joint "gripper" = width m) — the
same mock ros2_control stack the arm moves use. That matters: the mock's
joint_state_broadcaster already streams control/joint_states at 200 Hz, arm
joints AND "gripper", and the real driver turns every message into a gripper
command (width, force = its gripper_default_effort param). So the mock is the
single source of truth for the jaw target; publishing a second target on
control/joint_states directly makes the jaws chatter between the two. Moving
the mock joint along a ramp also gives the firmware the continuous 0x159
stream it needs. The grip force is therefore the driver's
gripper_default_effort (real.launch.py gripper_force:=), not per command.
"Open" aims at the full stroke (--grip-open, default 0.1 m; the taught open
end from src/data/gripper.json when one exists) and, if the jaws stall short
of it — the modified tips hit the end stop early — steps the target back
2 mm inside so the 200 Hz stream stops pushing the stop. "Close" aims at the
closed end (0, or the taught closed reading) and deliberately LEAVES the
target there when the jaws stall on an object: the force-limited push is the
grasp, so the object stays held. Buttons are never gated on the reported
state: "close" is always sent, even when the jaws already read closed
(re-grip). Preset slots take mm values within [0, --grip-open] (or the taught
range). In SIM the same path moves the mock gripper (visible in RViz).

Everything runs through the same functions as the move_level.py CLI
(make_plan / preview / execute_moveit / plan_carry) on one worker thread, so
GUI moves behave exactly like typed ones. Planner output lands in the log
pane; rclpy logger lines (plan summary, execution result) go to the terminal.

    source /opt/ros/humble/setup.bash
    source third_party/agx_arm_ros/install/setup.bash
    python3 level_gui.py            # sim (refuses to execute if a real
    python3 level_gui.py --real     # driver is up); --real for the robot
"""
import argparse
import contextlib
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "nero_ctrl")))
import level_path as lp  # noqa: E402
import move_level as ml  # noqa: E402
import gripper_range as gr  # noqa: E402  (hand-taught jaw range, no HW)
try:                          # direct-CAN teach helpers (pyAgxArm)
    import connect_check as cc  # noqa: E402
    import mode_control as mc  # noqa: E402
    import show_joints as sj  # noqa: E402
    import move_to_pose as mp  # noqa: E402  (enable / watch_motion)
except ImportError as _exc:   # pyAgxArm missing: teach panel disabled
    cc = mc = sj = mp = None
    _TEACH_IMPORT_ERR = str(_exc)

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from control_msgs.action import FollowJointTrajectory  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402
try:
    from agx_arm_msgs.msg import GripperStatus  # noqa: E402
except ImportError:            # agx_arm_ros install not sourced
    GripperStatus = None

CFG_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    "level_targets.json"))
POLL_MS = 2000        # live TCP pose refresh period (read-only)
PUMP_MS = 50          # UI queue pump period
DANGER_BG = "#c62828"
DANGER_ACTIVE = "#8e0000"
PARAMS = ("lift", "step", "v", "a", "qdot", "qddot", "dwell")
MAP_SIZE = 380        # px, square top-down map of the work plane
MAP_SLAB = 0.025      # m, half-thickness of the plane slab for reachability
MAP_CELL = 0.01       # m, reachability shading cell size
N_PLANES = 5
PLANE_NAMES = tuple(f"plane_{i}" for i in range(1, N_PLANES + 1))
PLANE_DEFAULT_X = (-0.36, -0.31, -0.26, -0.21, -0.16)   # lowest first
PLANE_STYLE = {       # region fill / label accent per plane (lowest first)
    "plane_1": ("#cfe5f7", "#1565c0"),
    "plane_2": ("#d7ecd9", "#2e7d32"),
    "plane_3": ("#fde9c9", "#ef6c00"),
    "plane_4": ("#e8d9f3", "#6a1b9a"),
    "plane_5": ("#f8d7da", "#b71c1c"),
}
MOVE_MODES = (        # (key, radio label, level_path profile)
    ("plane", "평면 내 이동 (들어올림→수평→내림)", "lift"),
    ("line", "평면 간 직선 이동 (현재→목표 한 직선)", "straight"),
    ("vertical", "평면 간 수직 이동 (y·z 고정, 높이만)", "straight"),
)
MODE_LABEL = {k: lbl.split(" (")[0] for k, lbl, _ in MOVE_MODES}
MODE_PROFILE = {k: prof for k, _, prof in MOVE_MODES}

GRIP_STATUS_TOPIC = "feedback/gripper_status"
GRIP_MOCK_TOPIC = "control/joint_states"   # mock broadcaster (SIM status)
GRIP_ACTION = "gripper_controller/follow_joint_trajectory"
GRIP_JOINT = "gripper"
GRIP_TOL = 0.002          # m, "reached" tolerance (2 mm)
GRIP_SPEED = 0.04         # m/s, mock ramp rate (real jaws follow the ramp)
GRIP_MIN_T = 0.5          # s, shortest ramp (re-grip when already there)
GRIP_TIMEOUT = 8.0        # s, give up waiting for the jaws
GRIP_WIDTH_MAX = 0.1      # m, driver wrapper hard limit
GRIP_STALE_S = 2.0        # s, status older than this is shown as stale
GRIP_OPEN_DEFAULT = 0.1   # m, "open" target / slot upper bound (--grip-open)
GRIP_BACKOFF = 0.002      # m, step back from the open end stop after a stall
GRIP_SLOTS = 4            # numeric width presets, persisted in the cfg file
TEACH_SETTLE_S = 4.0      # s, wait for measured joints after follower switch
ZERO_LOG_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    "joint_zero_log.json"))   # joint readings before every zero-set
JOG_SPEED_PCT = 5         # move_j speed for the fine-alignment jog
JOG_STEPS_DEG = (-1.0, -0.2, 0.2, 1.0)
ZERO_OK_DEG = 0.5         # a zeroed joint must read within this of 0
LIGHT_ON, LIGHT_OFF = "#2e7d32", "#ccc"


def load_cfg():
    try:
        with open(CFG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    planes = cfg.get("planes")
    if not isinstance(planes, dict):
        planes = {}
    fixed = {}
    for i, name in enumerate(PLANE_NAMES):
        try:
            fixed[name] = float(planes[name])
        except (KeyError, TypeError, ValueError):
            # missing (older 2-plane config): stack above the previous one
            prev = fixed[PLANE_NAMES[i - 1]] if i else PLANE_DEFAULT_X[0]
            fixed[name] = round(prev + 0.05, 3) if i else PLANE_DEFAULT_X[0]
    cfg["planes"] = fixed
    cfg.setdefault("targets", {})
    cfg["confirm"] = bool(cfg.get("confirm", True))
    slots = cfg.get("gripper_slots")
    if not isinstance(slots, list) or len(slots) != GRIP_SLOTS:
        slots = [""] * GRIP_SLOTS
    cfg["gripper_slots"] = [str(x) for x in slots]
    return cfg


def save_cfg(cfg):
    os.makedirs(os.path.dirname(CFG_PATH), exist_ok=True)
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


class _LogWriter:
    """A stdout stand-in that forwards whole lines to the UI queue."""

    def __init__(self, ui_q):
        self.ui_q = ui_q
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.ui_q.put(("log", line))

    def flush(self):
        if self._buf:
            self.ui_q.put(("log", self._buf))
            self._buf = ""


class Worker(threading.Thread):
    """Serializes ALL node access: submitted jobs run one at a time with
    their prints redirected into the log pane; silent jobs (polls) skip the
    busy indicator."""

    def __init__(self, ui_q):
        super().__init__(daemon=True)
        self.ui_q = ui_q
        self.jobs = queue.Queue()
        self.busy = False

    def submit(self, fn, silent=False):
        self.jobs.put((fn, silent))

    def idle(self):
        return not self.busy and self.jobs.empty()

    def run(self):
        while True:
            fn, silent = self.jobs.get()
            if fn is None:
                return
            self.busy = True
            if not silent:
                self.ui_q.put(("busy", True))
            out = _LogWriter(self.ui_q)
            try:
                with contextlib.redirect_stdout(out):
                    fn()
            except Exception as exc:  # noqa: BLE001 — surface, don't die
                self.ui_q.put(("log", f"!! {type(exc).__name__}: {exc}"))
            finally:
                out.flush()
                self.busy = False
                if not silent:
                    self.ui_q.put(("busy", False))


class PlaneMap:
    """Top-down canvas view of one work plane: base +Y right, +Z up.

    Layers: shaded reachability region (redrawn on plane change), then
    markers (base origin, live TCP, pending goal, saved targets) redrawn
    on every update. Clicks are handed back in plane coordinates (y, z)."""

    def __init__(self, parent, on_click, on_dblclick):
        self.canvas = tk.Canvas(parent, width=MAP_SIZE, height=MAP_SIZE,
                                bg="white", highlightthickness=1,
                                highlightbackground="#999")
        self.coord_var = tk.StringVar(value="")
        self._on_click = on_click
        self._on_dblclick = on_dblclick
        self.set_bounds((-0.6, 0.6, -0.2, 0.8))
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Double-Button-1>", self._dblclick)
        self.canvas.bind("<Motion>", self._motion)
        self.canvas.bind("<Leave>", lambda _e: self.coord_var.set(""))

    def set_bounds(self, b):
        """(ymin, ymax, zmin, zmax) in m; keeps an equal-scale mapping."""
        ymin, ymax, zmin, zmax = b
        self._s = MAP_SIZE / max(ymax - ymin, zmax - zmin, 1e-6)
        self._oy = MAP_SIZE / 2 - self._s * (ymin + ymax) / 2
        self._oz = MAP_SIZE / 2 + self._s * (zmin + zmax) / 2

    def _px(self, y, z):
        return self._oy + self._s * y, self._oz - self._s * z

    def _plane_yz(self, event):
        return ((event.x - self._oy) / self._s,
                (self._oz - event.y) / self._s)

    def _click(self, event):
        self._on_click(*self._plane_yz(event))

    def _dblclick(self, event):
        self._on_dblclick(*self._plane_yz(event))

    def _motion(self, event):
        y, z = self._plane_yz(event)
        self.coord_var.set(f"y {y:+.3f}  z {z:+.3f} m")

    def draw_region(self, cells_yz, msg=None, color="#cfe5f7"):
        """cells_yz: (N,2) cell centers reachable on this plane, or None
        (then msg — or a default missing-data note — is shown instead)."""
        self.canvas.delete("region")
        if cells_yz is None:
            self.canvas.create_text(
                MAP_SIZE / 2, 32, tags="region", fill="#b00",
                text=msg or "도달영역 데이터 없음 "
                            "(reachable_workspace.py 실행 필요)")
        else:
            r = max(self._s * MAP_CELL / 2, 1.0)
            for y, z in cells_yz:
                px, pz = self._px(y, z)
                self.canvas.create_rectangle(px - r, pz - r, px + r, pz + r,
                                             fill=color, outline="",
                                             tags="region")
        self.canvas.tag_lower("region")

    def draw_label(self, text, color="#333"):
        """Which-plane banner in the top-left corner of the map."""
        self.canvas.delete("plabel")
        self.canvas.create_text(8, 12, text=text, anchor="w", fill=color,
                                tags="plabel",
                                font=("TkDefaultFont", 10, "bold"))

    def draw_markers(self, tcp, goal, targets, path=False):
        """tcp: (y, z, on_plane) or None; goal: (y, z) or None;
        targets: iterable of (name, y, z, on_selected_plane); path: also
        draw the straight TCP -> goal line (plane-change moves)."""
        self.canvas.delete("marker")
        c = self.canvas
        if path and tcp is not None and goal is not None:
            c.create_line(*self._px(tcp[0], tcp[1]), *self._px(*goal),
                          tags="marker", fill="#c62828", dash=(4, 3))
        # base origin crosshair
        by, bz = self._px(0.0, 0.0)
        c.create_line(by - 7, bz, by + 7, bz, tags="marker", fill="#333")
        c.create_line(by, bz - 7, by, bz + 7, tags="marker", fill="#333")
        c.create_text(by + 8, bz + 9, text="base", anchor="w",
                      tags="marker", fill="#333", font=("TkDefaultFont", 8))
        for name, y, z, on_plane in targets:
            py, pz = self._px(y, z)
            col = "#e65100" if on_plane else "#bbb"
            c.create_oval(py - 4, pz - 4, py + 4, pz + 4, tags="marker",
                          outline=col, width=2)
            c.create_text(py + 6, pz - 6, text=name, anchor="w", fill=col,
                          tags="marker", font=("TkDefaultFont", 8))
        if tcp is not None:
            y, z, on_plane = tcp
            py, pz = self._px(y, z)
            col = "#2e7d32" if on_plane else "#999"
            c.create_oval(py - 5, pz - 5, py + 5, pz + 5, tags="marker",
                          fill=col, outline="")
            c.create_text(py, pz - 11, text="TCP", fill=col, tags="marker",
                          font=("TkDefaultFont", 8))
        if goal is not None:
            py, pz = self._px(*goal)
            c.create_line(py - 8, pz, py + 8, pz, tags="marker",
                          fill="#c62828", width=2)
            c.create_line(py, pz - 8, py, pz + 8, tags="marker",
                          fill="#c62828", width=2)


class LevelGui:
    def __init__(self, root, node, args):
        self.root = root
        self.node = node
        self.args = args
        self.cfg = load_cfg()
        self.ui_q = queue.Queue()
        self.worker = Worker(self.ui_q)
        self.last = None          # (q, t, qd, qdd) of the last plan
        self.last_desc = ""       # one-line description of that plan
        self.last_pos = None      # tcp position from the poll
        self.last_yaw = None      # absolute tcp yaw [rad] from the poll
        self._wit_p = "loading"   # witness tcp positions for the map region
        self._goal_yz = None      # pending goal marker on the map
        self._buttons = []
        self._busy = False
        self._grip = None         # latest gripper status: (dict, wall time)
        self._grip_lock = threading.Lock()
        self._grip_client = None  # lazily created gripper_controller client
        self._robot = None        # direct pyAgxArm CAN connection (teach)
        self._robot_err = None    # last connect failure (str)
        self.last_role = None     # linkage role from the poll (LEADER/...)
        if args.real:
            if GripperStatus is not None:
                node.create_subscription(GripperStatus, GRIP_STATUS_TOPIC,
                                         self._grip_cb, 10)
        else:                     # SIM: the mock joint is the only status
            node.create_subscription(JointState, GRIP_MOCK_TOPIC,
                                     self._mock_grip_cb, 1)

        mode = "REAL" if args.real else "SIM"
        root.title(f"Nero Level Move — {mode}")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build(mode)
        self.worker.start()
        self.worker.submit(self._job_poll, silent=True)
        self.worker.submit(self._job_load_witnesses, silent=True)
        root.after(PUMP_MS, self._pump)
        root.after(POLL_MS, self._schedule_poll)

    # ---------------------------------------------------------------- build
    def _btn(self, parent, text, command, danger=False, **kw):
        if danger:
            b = tk.Button(parent, text=text, command=command,
                          bg=DANGER_BG, fg="white",
                          activebackground=DANGER_ACTIVE,
                          activeforeground="white", **kw)
        else:
            b = ttk.Button(parent, text=text, command=command, **kw)
        self._buttons.append(b)
        return b

    def _build(self, mode):
        pad = {"padx": 6, "pady": 2}

        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)
        ttk.Label(top, text=f"모드: {mode}   (플랜은 항상 현재 자세에서 시작)")\
            .pack(side="left")
        self.busy_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.busy_var, foreground="#b00")\
            .pack(side="right")
        # warning toggle: off = no "진행할까요?" dialog before robot moves and
        # the inline safety hints are hidden (the log still gets one line)
        self._confirm = self.cfg["confirm"]      # worker-thread copy
        self.confirm_var = tk.BooleanVar(value=self._confirm)
        ttk.Checkbutton(top, text="경고문 표시 (실행 전 확인 창)",
                        variable=self.confirm_var,
                        command=self._toggle_confirm)\
            .pack(side="right", padx=12)

        cur = ttk.LabelFrame(self.root, text="현재 TCP (자동 갱신)")
        cur.pack(fill="x", **pad)
        self.cur_var = tk.StringVar(value="—")
        ttk.Label(cur, textvariable=self.cur_var,
                  font=("TkFixedFont", 10)).pack(side="left", padx=6, pady=3)
        self._btn(cur, "갱신",
                  lambda: self.worker.submit(self._job_poll, silent=True))\
            .pack(side="right", padx=6, pady=2)

        # plane selection -------------------------------------------------
        pf = ttk.LabelFrame(self.root, text="작업 평면 (world up = base +X → "
                                             "x 값이 평면 높이, plane_1이 가장 "
                                             "아래, 위로 갈수록 x 증가)")
        pf.pack(fill="x", **pad)
        row = ttk.Frame(pf)
        row.pack(fill="x", padx=6, pady=3)
        self.plane_var = tk.StringVar(value="plane_1")
        self.plane_x = {}
        for name in PLANE_NAMES:
            ttk.Radiobutton(row, text=name, value=name,
                            variable=self.plane_var).pack(side="left")
            v = tk.StringVar(value=f"{self.cfg['planes'][name]:g}")
            ttk.Entry(row, textvariable=v, width=7).pack(side="left",
                                                         padx=(2, 12))
            self.plane_x[name] = v
        self._btn(row, "평면 저장", self._save_planes).pack(side="left")

        # goal ------------------------------------------------------------
        gf = ttk.LabelFrame(self.root, text="목표 (선택 평면 위 좌표, m / "
                                             "yaw는 절대각 deg: 0 = +z 자세 "
                                             "방향, 비우면 현재 유지)")
        gf.pack(fill="x", **pad)
        mrow = ttk.Frame(gf)
        mrow.pack(fill="x", padx=6, pady=(3, 0))
        ttk.Label(mrow, text="이동 방식").pack(side="left", padx=(0, 6))
        self.mode_var = tk.StringVar(value="plane")
        for key, label, _ in MOVE_MODES:
            ttk.Radiobutton(mrow, text=label, value=key,
                            variable=self.mode_var).pack(side="left",
                                                         padx=(0, 10))
        self.mode_var.trace_add("write", lambda *_: self._mode_changed())
        row = ttk.Frame(gf)
        row.pack(fill="x", padx=6, pady=3)
        self.goal_vars = {}
        self.goal_entries = {}
        for key, label, width in (("y", "y", 8), ("z", "z", 8),
                                  ("yaw", "yaw°", 6)):
            ttk.Label(row, text=label).pack(side="left")
            v = tk.StringVar()
            e = ttk.Entry(row, textvariable=v, width=width)
            e.pack(side="left", padx=(2, 10))
            self.goal_vars[key] = v
            self.goal_entries[key] = e
        self._btn(row, "현재 위치 채우기", self._fill_current)\
            .pack(side="left", padx=4)
        self.mode_hint = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.mode_hint, foreground="#555")\
            .pack(side="left", padx=8)
        row2 = ttk.Frame(gf)
        row2.pack(fill="x", padx=6, pady=(0, 4))
        self._btn(row2, "플랜", self._do_plan).pack(side="left")
        # planning cancel: the one button that is live ONLY while busy
        # (not in self._buttons); sets level_path.CANCEL — the planner
        # stops at its next waypoint and the job reports "cancelled"
        self.stop_btn = ttk.Button(row2, text="계산 중지",
                                   command=self._do_stop_plan)
        self.stop_btn.pack(side="right", padx=6)
        self.prev_btn = self._btn(row2, "프리뷰", self._do_preview)
        self.prev_btn.pack(side="left", padx=4)
        self.go_btn = self._btn(row2, "실행 (go)", self._do_go, danger=True)
        self.go_btn.pack(side="left")
        self._btn(row2, "Carry (운반 자세)", self._do_carry, danger=True)\
            .pack(side="left", padx=14)
        self._btn(row2, "평면 +z 자세", self._do_sagittal, danger=True)\
            .pack(side="left")
        self.warn_label = ttk.Label(
            row2, text="← 액체 없이! 수평 비보장 셋업 무브 "
                       "(+z 자세: 관절 1/3/5/6=0, 그리퍼 열림 필요)")
        if self.cfg["confirm"]:
            self.warn_label.pack(side="left", padx=4)

        # gripper ---------------------------------------------------------
        rec = gr.load_range(mode="width")
        if rec is None:
            rng = (f"열기 = 최대 {self.args.grip_open * 1000:.0f} mm (스톱에서 "
                   "멈추면 2 mm 후퇴) / 닫기 = 물체까지 (힘 유지)")
        else:
            rng = (f"티칭 범위: 닫힘 {gr.fmt(gr.close_target(rec), 'width')} "
                   f"/ 열림 {gr.fmt(gr.open_target(rec), 'width')}")
        grf = ttk.LabelFrame(self.root, text=f"그리퍼 — {rng}")
        grf.pack(fill="x", **pad)
        row = ttk.Frame(grf)
        row.pack(fill="x", padx=6, pady=3)
        self.grip_var = tk.StringVar(value="상태 없음")
        ttk.Label(row, textvariable=self.grip_var,
                  font=("TkFixedFont", 10)).pack(side="left")
        ttk.Label(row, text="   (닫기는 항상 전송: 닫힘에서 닫기 = 재파지 · "
                            "힘은 launch의 gripper_force)").pack(side="left")
        row2 = ttk.Frame(grf)
        row2.pack(fill="x", padx=6, pady=(0, 4))
        self._btn(row2, "열기", lambda: self._do_grip("open"), danger=True)\
            .pack(side="left", padx=2)
        self._btn(row2, "닫기", lambda: self._do_grip("close"), danger=True)\
            .pack(side="left", padx=2)
        # width presets: "이동 n" button + its mm entry, all on this row;
        # the values persist in the cfg file whenever a slot is used
        ttk.Label(row2, text="슬롯 (mm):").pack(side="left", padx=(14, 4))
        self.grip_slots = []
        for i in range(GRIP_SLOTS):
            self._btn(row2, f"이동 {i + 1}",
                      lambda i=i: self._do_grip("slot", i), danger=True)\
                .pack(side="left", padx=(6, 1))
            v = tk.StringVar(value=self.cfg["gripper_slots"][i])
            ttk.Entry(row2, textvariable=v, width=6, justify="center")\
                .pack(side="left")
            self.grip_slots.append(v)

        # teaching: leader / follower over direct CAN (real arm only) ------
        if mode == "REAL":
            self._build_teach()

        # parameters — packed at the BOTTOM first so the map/log row above
        # is what shrinks when the screen is short, not this row ---------
        self._build_params(pad)

        # map + points + log (side by side: the window is not tall enough
        # for a log pane UNDER the map on a 1080p screen) ------------------
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, **pad)
        mf = ttk.LabelFrame(mid, text="평면 지도 — 클릭 = 목표 지정, "
                                      "더블클릭 = 바로 플랜  (+Y →, +Z ↑)")
        mf.pack(side="left", padx=(0, 6))
        self.map = PlaneMap(mf, self._map_click, self._map_dblclick)
        self.map.canvas.pack(padx=4, pady=(4, 0))
        ttk.Label(mf, textvariable=self.map.coord_var,
                  font=("TkFixedFont", 9)).pack(anchor="e", padx=6)
        self.plane_var.trace_add("write", lambda *_: self._redraw_region())

        tf = ttk.LabelFrame(mid, text="포인트 (평면·x·y·z·yaw 저장 / "
                                      "더블클릭 = 필드로 불러오기)")
        tf.pack(side="left", fill="y", padx=(0, 6))
        left = ttk.Frame(tf)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        self.tgt_list = tk.Listbox(left, height=5, width=52,
                                   exportselection=False,
                                   font=("TkFixedFont", 9))
        self.tgt_list.pack(fill="both", expand=True)
        self.tgt_list.bind("<Double-Button-1>", lambda _e: self._load_target())
        self.tgt_list.bind("<<ListboxSelect>>", lambda _e: self._preview_sel())
        right = ttk.Frame(tf)
        right.pack(side="left", fill="y", padx=6, pady=4)
        ttk.Label(right, text="이름").pack(anchor="w")
        self.tgt_name = tk.StringVar()
        ttk.Entry(right, textvariable=self.tgt_name, width=16).pack(fill="x")
        self._btn(right, "현재 필드를 포인트로 저장", self._save_target)\
            .pack(fill="x", pady=2)
        self._btn(right, "현재 TCP를 포인트로 저장", self._save_current_point)\
            .pack(fill="x")
        self._btn(right, "불러오기", self._load_target).pack(fill="x", pady=2)
        self._btn(right, "삭제", self._del_target).pack(fill="x")
        ttk.Separator(right).pack(fill="x", pady=6)
        ttk.Label(right, text="포인트 간 이동 방식").pack(anchor="w")
        self.pt_mode_var = tk.StringVar(value="plane")
        for key, label, _ in MOVE_MODES:
            ttk.Radiobutton(right, text=label.split(" (")[0], value=key,
                            variable=self.pt_mode_var).pack(anchor="w")
        self._btn(right, "현재 → 포인트로 플랜", self._plan_to_point)\
            .pack(fill="x", pady=(4, 0))

        lf = ttk.LabelFrame(mid, text="로그 (플래너 출력 — 창을 키우면 같이 "
                                      "커짐)")
        lf.pack(side="left", fill="both", expand=True)
        self.log = ScrolledText(lf, height=20, width=60, state="disabled",
                                font=("TkFixedFont", 10))
        self.log.pack(fill="both", expand=True, padx=4, pady=4)
        self._fit_window()

        self._refresh_targets()
        self._update_buttons()
        self._mode_changed()
        self._redraw_region()

    def _build_params(self, pad):
        prm = ttk.LabelFrame(self.root, text="파라미터 (lift/step m, v/a "
                                              "m/s·m/s², qdot/qddot rad/s·rad/s², "
                                              "dwell 꼭짓점 정지 s, 음수=무정지)")
        prm.pack(side="bottom", fill="x", **pad)
        row = ttk.Frame(prm)
        row.pack(fill="x", padx=6, pady=3)
        self.param_vars = {}
        for name in PARAMS:
            ttk.Label(row, text=name).pack(side="left")
            v = tk.StringVar(value=f"{getattr(self.args, name):g}")
            ttk.Entry(row, textvariable=v, width=7)\
                .pack(side="left", padx=(2, 8))
            self.param_vars[name] = v

    def _build_teach(self):
        tf = ttk.LabelFrame(self.root, text="티칭 — leader(무저항 드래그) / "
                                             "follower(제어) 전환, 직접 CAN "
                                             f"({self.args.can})")
        tf.pack(fill="x", padx=6, pady=3)
        row = ttk.Frame(tf)
        row.pack(fill="x", padx=6, pady=3)
        self.mode_lights = {}
        for text, role in (("Follower (제어 복귀)", "FOLLOWER"),
                           ("Leader (손으로 이동)", "LEADER")):
            box = ttk.Frame(row)
            box.pack(side="left", padx=(0, 8))
            cv = tk.Canvas(box, width=16, height=16, highlightthickness=0,
                           bg=self.root.cget("bg"))
            cv.pack(side="left")
            oid = cv.create_oval(3, 3, 13, 13, fill=LIGHT_OFF, outline="#333")
            self.mode_lights[role] = (cv, oid)
            self._btn(box, text,
                      lambda r=role: self._do_mode(r.lower()), danger=True)\
                .pack(side="left")
        self.role_var = tk.StringVar(value="모드: 미확인 (버튼을 누르면 연결)")
        ttk.Label(row, textvariable=self.role_var,
                  font=("TkFixedFont", 10)).pack(side="left", padx=8)
        ttk.Separator(row, orient="vertical").pack(side="left", fill="y",
                                                   padx=10)
        self._btn(row, "Leader 캡처: follower → 포인트 저장 → leader",
                  self._do_leader_capture, danger=True).pack(side="left")
        ttk.Label(row, text="← LEADER에서만: 손으로 잡은 자세를 포인트 패널의 "
                            "이름으로 저장 (가까운 평면, 실측 x·y·z·yaw·관절)")\
            .pack(side="left", padx=6)
        # row 2: joint zero calibration (semi-automatic, see module doc)
        zr = ttk.Frame(tf)
        zr.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(zr, text="관절 영점:").pack(side="left")
        self.zero_target_var = tk.StringVar(value="전체")
        ttk.Combobox(zr, textvariable=self.zero_target_var, width=5,
                     state="readonly",
                     values=["전체"] + [f"J{i}" for i in range(1, 8)])\
            .pack(side="left", padx=(4, 10))
        ttk.Label(zr, text="미세 조정 (FOLLOWER, 선택 관절):").pack(side="left")
        for d in JOG_STEPS_DEG:
            self._btn(zr, f"{d:+g}°", lambda d=d: self._do_jog(d),
                      danger=True).pack(side="left", padx=2)
        self._btn(zr, "관절 읽기", self._do_read_joints).pack(side="left",
                                                          padx=(10, 2))
        self._btn(zr, "영점 설정 (disable → zero → enable)", self._do_zero,
                  danger=True).pack(side="left", padx=(10, 2))
        ttk.Label(zr, text="← 순서: LEADER에서 마킹에 맞춤 → Follower(이동량 "
                           "표시) → 미세 조정 → 영점 설정. 되돌리기용 이전 "
                           "값은 joint_zero_log.json").pack(side="left", padx=6)
        n_teach = 3 + len(JOG_STEPS_DEG) + 2
        if cc is None:
            for b in self._buttons[-n_teach:]:
                b.configure(state="disabled")
            self.role_var.set(f"티칭 사용 불가 — pyAgxArm 없음 "
                              f"({_TEACH_IMPORT_ERR})")

    # ------------------------------------------------------------- teaching
    def _robot_conn(self):
        """Direct pyAgxArm connection, opened on first use (worker thread).
        Raises RuntimeError with a user message when it cannot connect."""
        if self._robot is not None:
            return self._robot
        if cc is None:
            raise RuntimeError("pyAgxArm을 불러올 수 없습니다")
        try:
            self._robot, _ = cc.connect(self.args.can, self.args.fw)
        except Exception as exc:  # noqa: BLE001
            self._robot_err = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                f"'{self.args.can}' 연결 실패 ({self._robot_err}) — CAN "
                "어댑터/can_activate 확인") from None
        self._robot_err = None
        print(f"티칭: '{self.args.can}' 직접 연결됨 (fw {self.args.fw})")
        return self._robot

    def _read_role(self):
        """Current linkage role via the direct connection (worker thread);
        None when not connected."""
        if self._robot is None:
            return None
        try:
            modes = cc.read_modes(self._robot)
        except Exception:  # noqa: BLE001
            return None
        return modes.get("linkage_role")

    def _exec_ok(self):
        """Motion gate for the worker: refuse while the arm is a LEADER (it
        ignores commands and would jump when it stops being one), then the
        sim/real publisher check."""
        role = self._read_role() if self._robot is not None else self.last_role
        if role == "LEADER":
            print("!! 팔이 LEADER(무저항) 모드입니다 — 먼저 [Follower]로 "
                  "복귀하세요")
            return False
        return ml.exec_allowed(self.node, self.args)

    def _do_mode(self, target):
        def job():
            try:
                robot = self._robot_conn()
            except RuntimeError as exc:
                print(f"!! {exc}")
                return
            before = cc.read_modes(robot)
            print(f"모드 전환 → {target}  (현재 "
                  f"{cc.format_mode_oneline(before)})")
            q_lead = None
            if target == "follower" and before.get("linkage_role") == "LEADER":
                q_lead = self._joints_now(robot, want="leader")
            ok, _b, after = mc.set_mode(robot, target, self._ask_from_worker)
            if not ok:
                print("모드 전환 취소/실패 (leader는 모터 enable이 안 되면 "
                      "실패합니다)")
                return
            role = (after or {}).get("linkage_role")
            print(f"모드: {cc.format_mode_oneline(after or {})}")
            if q_lead is not None and role == "FOLLOWER":
                q_meas = self._joints_now(robot, want="measured",
                                          timeout=TEACH_SETTLE_S)
                if q_meas is not None:
                    self._report_shift(q_lead, q_meas)
            if target == "leader" and role != "LEADER":
                print("!! 아직 LEADER가 아닙니다 — 팔을 받친 상태로 다시 "
                      "시도하세요")
            elif target == "follower" and role == "LEADER":
                print("!! 아직 LEADER입니다 — 다시 시도하세요")
            elif target == "leader":
                print("팔을 손으로 움직일 수 있습니다 (처짐 주의). 자세를 "
                      "잡은 뒤 [Leader 캡처]로 포인트 저장")
            self.ui_q.put(("role", role))
            self.ui_q.put(("plan", None, ""))
            self.worker.submit(self._job_poll, silent=True)
        self.worker.submit(job)

    def _joints_now(self, robot, want="measured", timeout=1.0):
        """Current joints [rad, 7] from the direct connection: "measured"
        (0x2A5 feedback, FOLLOWER) or "leader" (leader broadcast). None when
        that source is not available within `timeout`."""
        end = time.time() + timeout
        while True:
            ja, src = sj.read_joints(robot, timeout=0.3)
            if ja is not None and src.startswith(want) \
                    and len(ja.msg) >= 7:
                return np.array(list(ja.msg)[:7], float)
            if time.time() >= end:
                return None
            time.sleep(0.1)

    def _report_shift(self, q_a, q_b, what="리더값 → 제어 전환 후 실측"):
        d = np.rad2deg(q_b - q_a)
        print(f"{what} 관절 이동량 [deg]: "
              f"{np.round(d, 2).tolist()}  (최대 {np.abs(d).max():.2f}°)")
        if np.abs(d).max() > 0.2:
            print("   → 영점을 잡으려면 이 이동량만큼 마킹에서 벗어났습니다. "
                  "[미세 조정]으로 마킹에 다시 맞춘 뒤 [영점 설정]")

    def _zero_index(self):
        t = self.zero_target_var.get()
        return 255 if t == "전체" else int(t[1:])

    def _do_read_joints(self):
        def job():
            try:
                robot = self._robot_conn()
            except RuntimeError as exc:
                print(f"!! {exc}")
                return
            role = self._read_role()
            q = self._joints_now(robot, "leader" if role == "LEADER"
                                 else "measured", timeout=1.5)
            if q is None:
                print(f"!! 관절값을 읽지 못함 (모드 {role})")
                return
            print(f"관절 [{'LEADER 방송값' if role == 'LEADER' else '실측'}] "
                  f"deg: {np.round(np.rad2deg(q), 2).tolist()}")
        self.worker.submit(job)

    def _do_jog(self, delta_deg):
        """Fine alignment for zeroing: move ONE joint by delta_deg with
        move_j at JOG_SPEED_PCT (direct CAN, FOLLOWER only)."""
        idx = self._zero_index()
        if idx == 255:
            self.log_line("!! 미세 조정은 관절 하나를 고르세요 (J1~J7)")
            return

        def job():
            try:
                robot = self._robot_conn()
            except RuntimeError as exc:
                print(f"!! {exc}")
                return
            if self._read_role() != "FOLLOWER":
                print("!! 미세 조정은 FOLLOWER(제어) 상태에서만")
                return
            q = self._joints_now(robot, "measured", timeout=1.5)
            if q is None:
                print("!! 실측 관절값을 읽지 못함")
                return
            import nero_kinematics as nk
            lo, hi = nk.JOINT_LIMITS[idx - 1]
            tgt = q.copy()
            tgt[idx - 1] = float(np.clip(q[idx - 1] + np.deg2rad(delta_deg),
                                         lo, hi))
            if not self._ask_from_worker(
                    f"J{idx}를 {delta_deg:+g}° 움직입니다 "
                    f"({np.rad2deg(q[idx - 1]):+.2f}° → "
                    f"{np.rad2deg(tgt[idx - 1]):+.2f}°, 속도 {JOG_SPEED_PCT}%). "
                    "관절 이동 — 주변을 비우세요."):
                print("미세 조정 취소됨")
                return
            if not mp.enable_with_timeout(robot):
                print("!! 모터 enable 실패")
                return
            robot.set_speed_percent(JOG_SPEED_PCT)
            robot.move_j([float(v) for v in tgt])
            outcome, _st = mp.watch_motion(robot, timeout=10.0)
            q2 = self._joints_now(robot, "measured", timeout=1.5)
            now = "?" if q2 is None else f"{np.rad2deg(q2[idx - 1]):+.2f}°"
            print(f"미세 조정 J{idx} {delta_deg:+g}°: {outcome}, 현재 {now}")
            self.ui_q.put(("plan", None, ""))
            self.worker.submit(self._job_poll, silent=True)
        self.worker.submit(job)

    def _do_zero(self):
        """Semi-automatic joint zero: the selected joint (or all) must already
        sit at its mechanical zero mark. disable -> calibrate_joint (0x475
        0xAE) -> enable, readings logged before/after and the before-values
        appended to ZERO_LOG_PATH for undo."""
        idx = self._zero_index()
        what = "전체 관절 (255)" if idx == 255 else f"J{idx}"

        def read():
            return self._joints_now(robot, "measured", timeout=1.5)

        def fmt(q):
            return "?" if q is None else np.round(np.rad2deg(q), 2).tolist()

        def job():
            nonlocal robot
            try:
                robot = self._robot_conn()
            except RuntimeError as exc:
                print(f"!! {exc}")
                return
            role = self._read_role()
            if role != "FOLLOWER":
                print(f"!! 영점 설정은 FOLLOWER(제어) 상태에서만 (현재 {role})"
                      " — LEADER에서 맞춘 뒤 [Follower]로 전환하세요")
                return
            q_before = read()
            if q_before is None:
                print("!! 실측 관절값을 읽지 못해 중단")
                return
            if not self._ask_from_worker(
                    f"관절 영점 설정: {what}\n현재 실측 [deg] "
                    f"{fmt(q_before)}\n\n지금 위치가 새 영점이 됩니다 "
                    "(펌웨어에 저장, 되돌리려면 로그의 이전 값으로 재설정). "
                    "선택 관절이 기계적 영점 마킹에 정확히 맞아 있습니까?\n"
                    "순서: disable → 영점 → enable (잠깐 브레이크 유지)",
                    force=True):
                print("영점 설정 취소됨")
                return
            # undo record first, then act
            try:
                try:
                    with open(ZERO_LOG_PATH, encoding="utf-8") as f:
                        log = json.load(f)
                except (OSError, ValueError):
                    log = []
                log.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "target": what,
                            "joints_deg_before": fmt(q_before)})
                os.makedirs(os.path.dirname(ZERO_LOG_PATH), exist_ok=True)
                with open(ZERO_LOG_PATH, "w", encoding="utf-8") as f:
                    json.dump(log, f, indent=2, ensure_ascii=False)
                print(f"이전 관절값 기록: {ZERO_LOG_PATH}")
            except OSError as exc:
                print(f"!! 기록 실패 ({exc}) — 중단")
                return
            ok_dis = False
            try:
                ok_dis = bool(robot.disable(idx))
            except Exception as exc:  # noqa: BLE001
                print(f"!! disable 예외: {exc}")
            time.sleep(0.5)
            q_dis = read()
            print(f"disable({idx}) → {'OK' if ok_dis else '미확인'}, 실측 "
                  f"{fmt(q_dis)}")
            try:
                robot.calibrate_joint(idx)
            except Exception as exc:  # noqa: BLE001
                print(f"!! calibrate_joint 예외: {exc}")
            time.sleep(0.7)
            q_zero = read()
            print(f"calibrate_joint({idx}) 전송 → 실측 {fmt(q_zero)}")
            ok_en = mc._enable_with_timeout(robot)
            time.sleep(0.3)
            q_en = read()
            print(f"enable → {'OK' if ok_en else '!! 실패'}, 실측 {fmt(q_en)}")
            # verdict: the zeroed joints must now read ~0
            if q_zero is not None:
                sel = (list(range(7)) if idx == 255 else [idx - 1])
                off = np.rad2deg(np.abs(q_zero[sel])).max()
                if off <= ZERO_OK_DEG:
                    print(f"영점 적용됨: {what} 이 {off:.2f}° 이내로 0을 "
                          "읽습니다")
                else:
                    print(f"!! 영점이 반영되지 않은 듯: {what} 최대 "
                          f"{off:.2f}° (펌웨어가 0x475 영점을 무시했거나 "
                          "지연). 다시 [관절 읽기]로 확인")
            if q_before is not None and q_dis is not None:
                self._report_shift(q_before, q_dis, "disable 시 브레이크 "
                                                     "체결 이동량")
            self.ui_q.put(("plan", None, ""))
            self.worker.submit(self._job_poll, silent=True)
        robot = None
        self.worker.submit(job)

    def _do_leader_capture(self):
        """nero_gui's one-button teach flow, for level points: hold the arm
        (follower), read the measured joints, FK -> point, release back into
        leader. One confirm for the whole sequence."""
        name = self.tgt_name.get().strip()
        if not name:
            self.log_line("!! 포인트 이름을 입력하세요 (포인트 패널)")
            return
        if name in self.cfg["targets"] and not messagebox.askyesno(
                "Nero — 이름 중복", f"포인트 '{name}'을(를) 덮어쓸까요?",
                parent=self.root):
            return

        def job():
            try:
                robot = self._robot_conn()
            except RuntimeError as exc:
                print(f"!! {exc}")
                return
            role = self._read_role()
            if role != "LEADER":
                print(f"!! 현재 LEADER 모드가 아닙니다 (현재 {role}) — 이 "
                      "버튼은 leader에서 손으로 잡은 자세를 저장할 때 씁니다 "
                      "(제어 상태에서는 [현재 TCP를 포인트로 저장])")
                return
            if not self._ask_from_worker(
                    f"Leader 캡처 '{name}':\n1) FOLLOWER로 전환 — 팔이 제어/"
                    "홀드로 돌아갑니다. 팔을 받치세요!\n2) 실측 관절로 포인트 "
                    "저장\n3) 다시 LEADER(무저항)로 — 팔이 손에서 풀립니다.\n"
                    "진행할까요?"):
                print("Leader 캡처 취소됨")
                return
            auto = lambda m: (print(" ".join(m.split())), True)[1]  # noqa: E731
            ok, _b, after = mc.set_mode(robot, "follower", auto)
            if not ok or (after or {}).get("linkage_role") == "LEADER":
                print("!! FOLLOWER 전환 실패 — 팔은 그대로입니다. 다시 시도")
                self.ui_q.put(("role", self._read_role()))
                return
            # wait for MEASURED joint feedback (the leader broadcast stops,
            # the 0x2A5 feedback starts) — up to TEACH_SETTLE_S
            ja, src = None, None
            end = time.time() + TEACH_SETTLE_S
            while time.time() < end:
                ja, src = sj.read_joints(robot, timeout=0.5)
                if ja is not None and src.startswith("measured"):
                    break
                ja = None
            saved = False
            if ja is None:
                print("!! follower 전환 후 실측 관절 피드백이 오지 않습니다 — "
                      "저장하지 않음")
            else:
                q = np.array(list(ja.msg)[:7], float)
                if len(q) != 7:
                    print(f"!! 관절 7개가 아닙니다: {len(q)}")
                else:
                    saved = self._save_taught_point(name, q)
            ok, _b, after = mc.set_mode(robot, "leader", auto)
            role = (after or {}).get("linkage_role")
            if not ok or role != "LEADER":
                print("!! LEADER 복귀 실패 — 팔은 FOLLOWER(제어) 상태입니다. "
                      "[Leader] 버튼으로 다시 시도")
            elif saved:
                print(f"완료 — '{name}' 저장됨. 팔은 다시 LEADER(드래그) "
                      "상태입니다")
            self.ui_q.put(("role", role))
            self.worker.submit(self._job_poll, silent=True)
        self.worker.submit(job)

    def _save_taught_point(self, name, q):
        """FK the taught joints -> point on the nearest plane (measured x
        kept; the plane's height is what gets planned). Worker thread; the
        cfg write + list refresh happen on the UI thread."""
        R_t, t_t = lp.tcp_transform(self.args.tcp_xyz, self.args.tcp_rpy)
        R, p, _, _ = lp.fk_tcp(q[None], R_t, t_t)
        R, p = R[0], p[0]
        tilt = float(np.arccos(np.clip(
            (R @ np.array([-1.0, 0.0, 0.0])) @ lp.UP, -1.0, 1.0)))
        yaw = float(np.rad2deg(lp.abs_yaw(R)))
        planes = self.cfg["planes"]
        near = min(planes, key=lambda n: abs(planes[n] - p[0]))
        dx = p[0] - planes[near]
        print(f"실측: TCP ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) yaw "
              f"{yaw:+.1f}° 운반축 기울기 {np.rad2deg(tilt):.1f}°, 관절 "
              f"{np.round(np.rad2deg(q), 1).tolist()} deg")
        if abs(dx) > MAP_SLAB:
            print(f"!! 주의: 가장 가까운 {near}(x={planes[near]:g})와 "
                  f"{dx * 1000:+.0f} mm 차이 — 그래도 저장합니다. 플랜은 평면 "
                  "높이로 가니 평면 x를 조정하거나 포인트를 확인하세요")
        if np.rad2deg(tilt) > 5.0:
            print("!! 주의: 손으로 잡은 자세가 수평이 아닙니다 (컨테이너 축 "
                  f"{np.rad2deg(tilt):.0f}° 기울어짐) — 수평 이동 시작점으로 "
                  "쓰려면 [Carry]로 정렬 필요")
        rec = {"plane": near, "x": round(float(p[0]), 4),
               "y": round(float(p[1]), 4), "z": round(float(p[2]), 4),
               "yaw": round(yaw, 2),
               "joints_deg": [round(float(v), 2) for v in np.rad2deg(q)],
               "taught": True}
        self.ui_q.put(("point", name, rec))
        return True

    def _fit_window(self):
        """Size the window so the log pane is actually on screen: as tall as
        the widgets want (log at its full height), capped to the screen, and
        never so short that the log gets squeezed out."""
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(max(self.root.winfo_reqwidth(), 1600), sw - 40)
        h = min(max(self.root.winfo_reqheight(), 900), sh - 80)
        self.root.geometry(f"{w}x{h}+{max((sw - w) // 2, 0)}+10")
        self.root.minsize(1100, 700)

    # ------------------------------------------------------------------ map
    def _map_click(self, y, z):
        if self.mode_var.get() == "vertical":
            self.log_line("수직 이동 모드에서는 y·z가 현재 TCP로 고정됩니다 — "
                          "지도 클릭 무시 (다른 이동 방식을 고르세요)")
            return
        self.goal_vars["y"].set(f"{y:.3f}")
        self.goal_vars["z"].set(f"{z:.3f}")
        self._goal_yz = (y, z)
        self._draw_markers()

    def _map_dblclick(self, y, z):
        if self._busy:
            self.log_line("!! 실행 중입니다 — 끝난 뒤 다시 시도하세요")
            return
        if self.mode_var.get() == "vertical":
            self._map_click(y, z)          # just the hint
            return
        self._map_click(y, z)
        self._do_plan()

    def _mode_changed(self):
        mode = self.mode_var.get()
        vertical = mode == "vertical"
        for key in ("y", "z"):
            self.goal_entries[key].configure(
                state="disabled" if vertical else "normal")
        self.mode_hint.set({
            "plane": "선택 평면 위에서 들어올려 옮김 (TCP가 그 평면에 있어야 함)",
            "line": "현재 TCP → 선택 평면의 (y, z)까지 리프트 없이 한 직선",
            "vertical": "현재 y·z 그대로 선택 평면 높이로 (직선, 리프트 없음)",
        }[mode])
        self._sync_goal_marker()

    def _sync_goal_marker(self):
        """Move the goal crosshair to whatever the y/z fields hold."""
        try:
            self._goal_yz = (self._float(self.goal_vars["y"], "y"),
                             self._float(self.goal_vars["z"], "z"))
        except ValueError:
            self._goal_yz = None
        self._draw_markers()

    def _redraw_region(self):
        name = self.plane_var.get()
        fill, accent = PLANE_STYLE[name]
        try:
            x = self._float(self.plane_x[name], f"{name} x")
        except ValueError:
            x = None
        rank = PLANE_NAMES.index(name)
        side = ("가장 아래" if rank == 0 else "가장 위"
                if rank == N_PLANES - 1 else f"아래서 {rank + 1}번째")
        self.map.draw_label(
            f"{name} ({side})" + (f"   x = {x:g} m" if x is not None else ""),
            accent)
        if isinstance(self._wit_p, str):          # still loading
            self.map.draw_region(None, msg="도달영역 계산 중…")
        elif self._wit_p is None:                 # npz missing
            self.map.draw_region(None)
        elif x is None:
            self.map.draw_region(None, msg="평면 x 값이 숫자가 아닙니다")
        else:
            m = np.abs(self._wit_p[:, 0] - x) < MAP_SLAB
            yz = self._wit_p[m, 1:3]
            cells = (np.unique(np.round(yz / MAP_CELL).astype(int), axis=0)
                     * MAP_CELL if len(yz) else np.empty((0, 2)))
            self.map.draw_region(cells, color=fill)
        self._draw_markers()

    def _draw_markers(self):
        try:
            _, x = self._plane_height()
        except ValueError:
            x = None
        tcp = None
        if self.last_pos is not None:
            p = self.last_pos
            on_plane = x is not None and abs(p[0] - x) < MAP_SLAB
            tcp = (p[1], p[2], on_plane)
        sel = self.plane_var.get()
        targets = [(n, t["y"], t["z"], t["plane"] == sel)
                   for n, t in sorted(self.cfg["targets"].items())]
        mode = self.mode_var.get() if hasattr(self, "mode_var") else "plane"
        goal = self._goal_yz
        if mode == "vertical":            # goal sits under the TCP
            goal = None if tcp is None else (tcp[0], tcp[1])
        self.map.draw_markers(tcp, goal, targets, path=(mode == "line"))

    def _job_load_witnesses(self):
        R_t, t_t = lp.tcp_transform(self.args.tcp_xyz, self.args.tcp_rpy)
        wit = lp._witnesses(R_t, t_t)
        self.ui_q.put(("wit", None if wit is None else wit[1]))

    # ------------------------------------------------------------- helpers
    def log_line(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _float(self, var, name, optional=False):
        s = var.get().strip()
        if not s:
            if optional:
                return None
            raise ValueError(f"'{name}' 값을 입력하세요")
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"'{name}'이(가) 숫자가 아닙니다: {s}") from None

    def _plane_height(self):
        name = self.plane_var.get()
        return name, self._float(self.plane_x[name], f"{name} x")

    def _sync_params(self):
        """Parse the parameter entries into self.args (raises on bad input)."""
        for name in PARAMS:
            setattr(self.args, name, self._float(self.param_vars[name], name))

    def _ask_from_worker(self, message, force=False):
        """Yes/no confirmation shown on the UI thread. With the warning
        toggle off it is skipped (auto-yes) and the message goes to the log
        as a single line instead — unless force=True (irreversible steps
        always ask)."""
        if not self._confirm and not force:
            print("(경고 창 꺼짐) " + " ".join(message.split()))
            return True
        reply = queue.Queue()
        self.ui_q.put(("ask", message, reply))
        return reply.get()

    def _toggle_confirm(self):
        on = bool(self.confirm_var.get())
        self._confirm = on
        self.cfg["confirm"] = on
        save_cfg(self.cfg)
        if on:
            self.warn_label.pack(side="left", padx=4)
        else:
            self.warn_label.pack_forget()
        self.log_line("경고문 표시: " + ("켜짐 — 실행 전 확인 창을 띄웁니다"
                                    if on else
                                    "꺼짐 — 확인 없이 바로 실행합니다 (주의)"))

    # ------------------------------------------------------------- actions
    def _save_planes(self):
        try:
            for name in self.plane_x:
                self.cfg["planes"][name] = self._float(self.plane_x[name],
                                                       f"{name} x")
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        xs = [self.cfg["planes"][n] for n in PLANE_NAMES]
        for lo, hi, x_lo, x_hi in zip(PLANE_NAMES[:-1], PLANE_NAMES[1:],
                                      xs[:-1], xs[1:]):
            if x_lo >= x_hi:
                self.log_line(f"!! {lo}(x={x_lo:g})은 {hi}(x={x_hi:g})보다 "
                              "아래(더 작은 x)여야 합니다 — 높이는 plane_1 "
                              "→ plane_5 순으로 커져야 합니다")
                return
        save_cfg(self.cfg)
        self.log_line("평면 저장됨: " + ", ".join(
            f"{n}={self.cfg['planes'][n]:g}" for n in PLANE_NAMES))
        self._redraw_region()
        self._refresh_targets()

    def _fill_current(self):
        if self.last_pos is None:
            self.log_line("!! 아직 TCP 위치를 읽지 못했습니다")
            return
        x, y, z = self.last_pos
        self.goal_vars["y"].set(f"{y:.3f}")
        self.goal_vars["z"].set(f"{z:.3f}")
        self.goal_vars["yaw"].set("" if self.last_yaw is None
                                  else f"{np.rad2deg(self.last_yaw):.1f}")
        # pick whichever plane is closer to the current height
        planes = self.cfg["planes"]
        near = min(planes, key=lambda n: abs(planes[n] - x))
        self.plane_var.set(near)
        self.log_line(f"현재 위치 (x={x:.3f}) → {near} 선택됨 "
                      f"(평면과의 높이차 {abs(planes[near] - x)*1000:.0f} mm)")
        self._sync_goal_marker()

    def _goal_from_fields(self):
        """(plane, [x, y, z], yaw) from the plane choice and the y/z/yaw
        fields — what a saved target records, whatever the move mode."""
        plane, x = self._plane_height()
        y = self._float(self.goal_vars["y"], "y")
        z = self._float(self.goal_vars["z"], "z")
        yaw = self._float(self.goal_vars["yaw"], "yaw", optional=True)
        return plane, np.array([x, y, z]), yaw

    def _move_spec(self):
        """What the selected move mode needs: (mode, plane, x, yz, yaw);
        yz is None in vertical mode (taken from the live TCP at plan time)."""
        mode = self.mode_var.get()
        plane, x = self._plane_height()
        yaw = self._float(self.goal_vars["yaw"], "yaw", optional=True)
        if mode == "vertical":
            return mode, plane, x, None, yaw
        y = self._float(self.goal_vars["y"], "y")
        z = self._float(self.goal_vars["z"], "z")
        return mode, plane, x, (y, z), yaw

    def _do_plan(self):
        try:
            self._sync_params()
            mode, plane, x, yz, yaw = self._move_spec()
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        a = self.args
        profile = MODE_PROFILE[mode]

        def job():
            q_start = ml.read_current_joints(self.node, a.joint_states)
            if q_start is None:
                print(f"!! '{a.joint_states}'에서 조인트 상태를 못 읽음 — "
                      "시뮬/드라이버 실행 중인지 확인")
                return
            R_t, t_t = lp.tcp_transform(a.tcp_xyz, a.tcp_rpy)
            R_cur, p_cur, _, _ = lp.fk_tcp(q_start[None], R_t, t_t)
            R_cur, p_cur = R_cur[0], p_cur[0]
            if yz is None:                       # vertical: keep y, z
                p_goal = np.array([x, p_cur[1], p_cur[2]])
            else:
                p_goal = np.array([x, yz[0], yz[1]])
            dx = float(p_goal[0] - p_cur[0])
            desc = (f"{MODE_LABEL[mode]}: {plane} x={p_goal[0]:g}, "
                    f"y={p_goal[1]:.3f}, z={p_goal[2]:.3f}"
                    + (f", yaw {yaw:g}° (절대)" if yaw is not None else ""))
            print("플랜 — " + desc)
            if mode == "plane":
                if abs(dx) > MAP_SLAB:
                    print(f"!! 주의: 현재 TCP(x={p_cur[0]:.3f})가 {plane}"
                          f"(x={x:g})에 있지 않습니다 (높이차 "
                          f"{dx * 1000:+.0f} mm). 평면 내 이동은 더 높은 쪽 "
                          "위로 들어올려 옮깁니다 — 평면을 바꾸려면 [평면 간 "
                          "직선/수직 이동]을 쓰세요")
                print(f"   경로: 들어올림 {a.lift:g} m → 수평 이동 → 내림")
            else:
                if abs(dx) < 1e-3 and yz is None:
                    print(f"!! 이미 {plane} 높이(x={p_cur[0]:.3f})에 있습니다 "
                          "— 수직 이동할 것이 없음")
                    self.ui_q.put(("plan", None, ""))
                    return
                print(f"   경로: 현재 ({p_cur[0]:.3f}, {p_cur[1]:.3f}, "
                      f"{p_cur[2]:.3f}) → 목표까지 한 직선, 높이 변화 "
                      f"{dx * 1000:+.0f} mm, 리프트 없음 (lift 파라미터 무시). "
                      "직선이 안 풀리면 옆/위/대각 곡선을 자동으로 시도 "
                      "(목표·평면·수평은 그대로)")
            R_goal = None
            if yaw is not None:
                R_goal = lp.orientation_at_yaw(R_cur, np.deg2rad(yaw))
            planned = ml.make_plan(self.node, q_start, p_goal, R_goal, a,
                                   profile=profile)
            self.ui_q.put(("plan", planned, desc))
            if planned is not None:
                print(f"플랜 OK: {planned[1][-1]:.1f}초 — [프리뷰]로 확인 "
                      "후 [실행 (go)]")
        self.worker.submit(job)

    def _do_preview(self):
        if self.last is None:
            self.log_line("!! 플랜이 없습니다 — 먼저 [플랜]을 누르세요")
            return
        planned = self.last
        self.worker.submit(lambda: ml.preview(self.node, *planned))

    def _do_go(self):
        if self.last is None:
            self.log_line("!! 플랜이 없습니다 — 먼저 [플랜]을 누르세요")
            return
        planned = self.last
        desc = self.last_desc
        a = self.args

        def job():
            if not self._exec_ok():
                return
            mode = "REAL 로봇" if a.real else "시뮬"
            if not self._ask_from_worker(
                    f"{mode}에서 마지막 플랜({planned[1][-1]:.1f}초)을 "
                    "실행할까요?" + (f"\n{desc}" if desc else "")):
                print("실행 취소됨")
                return
            ok = ml.execute_moveit(self.node,
                                   ml.to_joint_trajectory(*planned))
            print("실행 완료" if ok else "!! 실행 실패 — 터미널 로그 확인")
            self.ui_q.put(("plan", None))    # robot moved: plan is stale
        self.worker.submit(job)

    def _do_carry(self):
        try:
            self._sync_params()
            y = self._float(self.goal_vars["y"], "y", optional=True)
            z = self._float(self.goal_vars["z"], "z", optional=True)
            if (y is None) != (z is None):
                raise ValueError("carry는 y·z를 둘 다 채우거나 둘 다 "
                                 "비워야 합니다 (비우면 제자리 재정렬)")
            p_tgt = None
            where = "제자리"
            if y is not None:
                plane, x = self._plane_height()
                p_tgt = np.array([x, y, z])
                where = f"{plane} ({x:g}, {y:g}, {z:g})"
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        a = self.args

        def job():
            qj = ml.read_current_joints(self.node, a.joint_states)
            if qj is None:
                print(f"!! '{a.joint_states}'에서 조인트 상태를 못 읽음")
                return
            R_t, t_t = lp.tcp_transform(a.tcp_xyz, a.tcp_rpy)
            carry = ml.plan_carry(qj, p_tgt, a.qdot, R_t, t_t)
            if carry is None:
                print("!! carry 자세의 IK 해가 없습니다 — 베이스에 더 가까운 "
                      "위치로 시도하세요")
                return
            qs, ts, tilt0, dq = carry
            if not self._exec_ok():
                return
            mode = "REAL 로봇" if a.real else "시뮬"
            if not self._ask_from_worker(
                    f"{mode}에서 carry({where})?\n재정렬 "
                    f"{np.rad2deg(tilt0):.1f}°, 최대 관절 이동 "
                    f"{np.rad2deg(dq):.1f}°, {ts[-1]:.1f}초.\n"
                    "수평 비보장 관절 이동입니다 — 액체를 들고 있으면 안 "
                    "됩니다."):
                print("carry 취소됨")
                return
            ok = ml.execute_moveit(self.node, ml.to_joint_trajectory(qs, ts))
            print("carry 완료" if ok else "!! carry 실행 실패")
            self.ui_q.put(("plan", None))
        self.worker.submit(job)

    def _do_sagittal(self):
        """Move to the in-plane carry pose: joints 1/3/5/6 = 0, tcp at the
        selected plane height and the goal z along base +Z, gripper straight
        down. Not level on the way (so only with an OPEN, empty gripper),
        level at the end."""
        try:
            self._sync_params()
            z = self._float(self.goal_vars["z"], "z")
            plane, x = self._plane_height()
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        why = self._gripper_not_open()
        if why:
            self.log_line(f"!! +z 자세는 그리퍼가 열려 있을 때만: {why}")
            return
        a = self.args

        def job():
            qj = ml.read_current_joints(self.node, a.joint_states)
            if qj is None:
                print(f"!! '{a.joint_states}'에서 조인트 상태를 못 읽음")
                return
            R_t, t_t = lp.tcp_transform(a.tcp_xyz, a.tcp_rpy)
            res = ml.plan_sagittal_carry(qj, x, z, a.qdot, R_t, t_t)
            if res is None:
                print(f"!! {plane} (x={x:g})에서 z={z:g}에 닿는 관절 2/4/7 "
                      "해가 없습니다 — z를 줄여 보세요")
                return
            qs, ts, q_goal, p_goal, dq = res
            print(f"+z 자세: 관절 {np.round(np.rad2deg(q_goal), 1).tolist()} "
                  f"deg, TCP ({p_goal[0]:.3f}, {p_goal[1]:.3f}, "
                  f"{p_goal[2]:.3f})")
            why = self._gripper_not_open()
            if why:
                print(f"!! 그리퍼 상태 변경으로 중단: {why}")
                return
            if not self._exec_ok():
                return
            mode = "REAL 로봇" if a.real else "시뮬"
            if not self._ask_from_worker(
                    f"{mode}에서 {plane} +z 자세로 이동?\n"
                    f"TCP → ({p_goal[0]:.3f}, {p_goal[1]:.3f}, "
                    f"{p_goal[2]:.3f}), 최대 관절 이동 "
                    f"{np.rad2deg(dq):.1f}°, {ts[-1]:.1f}초.\n"
                    "수평 비보장 관절 이동입니다 — 그리퍼가 비어 있어야 "
                    "합니다."):
                print("+z 자세 취소됨")
                return
            ok = ml.execute_moveit(self.node, ml.to_joint_trajectory(qs, ts))
            print("+z 자세 완료 (수평)" if ok else "!! +z 자세 실행 실패")
            self.ui_q.put(("plan", None))
        self.worker.submit(job)

    # ------------------------------------------------------------- gripper
    def _gripper_not_open(self, margin=0.005):
        """None when the gripper is known to be open (fresh status, width
        within `margin` of the open target), else a short reason."""
        st = self._grip_state()
        if st is None:
            return f"'{GRIP_STATUS_TOPIC}' 상태 수신 안 됨"
        if st["age"] > GRIP_STALE_S:
            return f"그리퍼 상태가 오래됨 ({st['age']:.0f}s 전)"
        try:
            target = self._grip_target("open")
        except ValueError as exc:
            return str(exc)
        if st["width"] < target - margin:
            return (f"폭 {st['width'] * 1000:.1f} mm < 열림 "
                    f"{target * 1000:.1f} mm — 먼저 [열기]")
        return None

    def _grip_cb(self, msg):
        snap = {
            "width": float(msg.width), "force": float(msg.force),
            "enabled": bool(msg.driver_enable_status),
            "homed": bool(msg.homing_status),
            "errors": [name for flag, name in (
                (msg.voltage_too_low, "저전압"),
                (msg.motor_overheating, "모터과열"),
                (msg.driver_overcurrent, "과전류"),
                (msg.driver_overheating, "드라이버과열"),
                (msg.driver_error_status, "드라이버오류")) if flag],
        }
        with self._grip_lock:
            self._grip = (snap, time.time())

    def _mock_grip_cb(self, msg):
        """SIM: the mock broadcaster's "gripper" joint is the jaw state."""
        if GRIP_JOINT not in msg.name:
            return
        w = float(msg.position[msg.name.index(GRIP_JOINT)])
        snap = {"width": w, "force": 0.0, "enabled": True, "homed": True,
                "errors": [], "sim": True}
        with self._grip_lock:
            self._grip = (snap, time.time())

    def _grip_state(self):
        """Latest gripper status snapshot for the UI (plus its age), or
        None when nothing has been received."""
        with self._grip_lock:
            g = self._grip
        if g is None:
            return None
        snap, t = g
        return dict(snap, age=time.time() - t)

    def _show_grip(self, st):
        if st is None:
            self.grip_var.set(f"상태 없음 — '{GRIP_STATUS_TOPIC}' 수신 안 됨 "
                              "(드라이버/그리퍼 확인)")
            return
        bits = [f"폭 {st['width'] * 1000:5.1f} mm", f"힘 {st['force']:.2f} N",
                "enable" if st["enabled"] else "DISABLED",
                "homed" if st["homed"] else "미호밍"]
        if st["errors"]:
            bits.append("오류: " + ",".join(st["errors"]))
        if st.get("sim"):
            bits.append("(시뮬 mock)")
        if st["age"] > GRIP_STALE_S:
            bits.append(f"(마지막 수신 {st['age']:.0f}s 전 — 오래됨)")
        self.grip_var.set("  ".join(bits))

    def _grip_target(self, spec, slot=None):
        """Resolve 'open' / 'close' / a preset 'slot' to a width in metres.
        A hand-taught range (src/data/gripper.json) is honoured when present;
        otherwise open = --grip-open, close = 0. Raises ValueError with a
        user message."""
        rec = gr.load_range(mode="width")
        if rec is not None:
            lo, hi = gr.limits(rec)
        else:
            lo, hi = 0.0, min(self.args.grip_open, GRIP_WIDTH_MAX)
        if spec == "slot":
            w = self._float(self.grip_slots[slot], f"슬롯 {slot + 1}") / 1000.0
            if not lo <= w <= hi:
                raise ValueError(
                    f"목표 {w * 1000:.1f} mm가 허용 범위 "
                    f"[{lo * 1000:.1f}, {hi * 1000:.1f}] mm 밖입니다")
            return w
        if rec is not None:
            return gr.open_target(rec) if spec == "open" \
                else gr.close_target(rec)
        return hi if spec == "open" else 0.0

    def _do_grip(self, spec, slot=None):
        try:
            target = self._grip_target(spec, slot)
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        if spec == "slot":
            vals = [v.get().strip() for v in self.grip_slots]
            if vals != self.cfg["gripper_slots"]:
                self.cfg["gripper_slots"] = vals
                save_cfg(self.cfg)
        what = {"open": "열기", "close": "닫기",
                "slot": f"슬롯 {slot + 1 if slot is not None else ''} 이동"}[spec]
        a = self.args

        def job():
            if not self._exec_ok():
                return
            mode = "REAL 로봇" if a.real else "시뮬"
            if not self._ask_from_worker(
                    f"{mode} 그리퍼 {what}: 목표 {target * 1000:.1f} mm?\n"
                    "그리퍼가 움직입니다 — 손가락과 물건을 조에서 치우세요."):
                print(f"그리퍼 {what} 취소됨")
                return
            outcome, st = self._move_grip(target)
            if outcome == "holding" and spec == "open" and st is not None:
                # jaws hit the open end stop before the nominal stroke: park
                # the target just inside so the stream stops pushing the stop
                back = max(0.0, st["width"] - GRIP_BACKOFF)
                print(f"그리퍼 열기: {st['width'] * 1000:.1f} mm에서 스톱 — "
                      f"{back * 1000:.1f} mm로 후퇴")
                outcome, st = self._move_grip(back)
                label = "열기 (최대)"
            else:
                label = what
            self.ui_q.put(("grip", st))
            w = "?" if st is None else f"{st['width'] * 1000:.1f} mm"
            if outcome == "reached":
                print(f"그리퍼 {label} 완료: {w}")
            elif outcome == "holding" and spec == "close":
                print(f"그리퍼 닫기: {w}에서 물체 파지 — 힘 유지 중 "
                      "(놓으려면 열기)")
            elif outcome == "holding":
                print(f"그리퍼 {what}: {w}에서 정지 — 걸림 (힘 유지 중)")
            elif outcome == "no_server":
                print(f"!! 그리퍼 {what}: '{GRIP_ACTION}' 액션 서버 없음 — "
                      "sim.launch/real.launch의 gripper_controller 확인")
            elif outcome == "rejected":
                print(f"!! 그리퍼 {what}: gripper_controller가 목표를 거부")
            elif outcome == "no_status":
                src = GRIP_STATUS_TOPIC if a.real else GRIP_MOCK_TOPIC
                print(f"!! 그리퍼 {what}: '{src}' 상태가 안 옵니다 — "
                      + ("드라이버가 그리퍼를 못 잡았거나 CAN 끊김"
                         if a.real else "mock 브로드캐스터 확인"))
            else:
                print(f"!! 그리퍼 {what}: {GRIP_TIMEOUT:.0f}s 안에 "
                      f"{target * 1000:.1f} mm 도달 못 함 (현재 {w}). 드라이버 "
                      "enable/제어 게이트 확인 — 드라이버 로그를 보세요")
        self.worker.submit(job)

    def _move_grip(self, target):
        """Ramp the mock gripper joint to `target` via gripper_controller and
        wait for the (real or mock) jaws to get there, stall, or time out.
        Returns (outcome, last_state); outcome is one of "reached",
        "holding", "timeout", "no_status", "no_server", "rejected"."""
        if self._grip_client is None:
            self._grip_client = ActionClient(self.node, FollowJointTrajectory,
                                             GRIP_ACTION)
        if not self._grip_client.wait_for_server(timeout_sec=3.0):
            return "no_server", self._grip_state()
        st = self._grip_state()
        w0 = st["width"] if st is not None else target
        T = max(GRIP_MIN_T, abs(target - w0) / GRIP_SPEED)
        jt = JointTrajectory()
        jt.joint_names = [GRIP_JOINT]
        pt = JointTrajectoryPoint()
        pt.positions = [float(target)]
        pt.velocities = [0.0]
        pt.time_from_start.sec = int(T)
        pt.time_from_start.nanosec = int((T % 1.0) * 1e9)
        jt.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        fut = self._grip_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
        handle = fut.result()
        if handle is None or not handle.accepted:
            return "rejected", self._grip_state()
        # watch the jaws themselves, not the mock's goal result
        start = time.time()
        last_w = None
        still = 0
        seen = False
        while time.time() - start < GRIP_TIMEOUT:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            st = self._grip_state()
            if st is None or st["age"] > GRIP_STALE_S:
                continue
            seen = True
            w = st["width"]
            if abs(w - target) <= GRIP_TOL and time.time() - start >= T:
                end = time.time() + 0.3          # let the reading settle
                while time.time() < end:
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                return "reached", self._grip_state()
            # stalled: width stopped changing well short of the target
            # (only once the ramp has had time to get ahead of the jaws)
            if last_w is not None and abs(w - last_w) < GRIP_TOL / 20:
                still += 1
                if still >= 12 and time.time() - start > T + 0.5:
                    return "holding", st
            else:
                still = 0
            last_w = w
        return ("timeout" if seen else "no_status"), self._grip_state()

    # ------------------------------------------------------------- targets
    def _point_x(self, t):
        """Height a point is planned at: its plane's CURRENT height. Returns
        (x, note) — note is set when that differs from the x saved with the
        point (planes were re-taught since)."""
        x = self.cfg["planes"].get(t["plane"])
        if x is None:                       # plane name no longer exists
            return t.get("x"), f"평면 '{t['plane']}' 없음 — 저장된 x 사용"
        note = None
        if t.get("x") is not None and abs(t["x"] - x) > 1e-3:
            note = (f"저장 당시 x={t['x']:g} → 현재 {t['plane']} 높이 "
                    f"x={x:g} 적용")
        return x, note

    def _refresh_targets(self):
        self.tgt_list.delete(0, "end")
        for name, t in sorted(self.cfg["targets"].items()):
            yaw = "유지" if t.get("yaw") is None else f"{t['yaw']:g}°"
            x, note = self._point_x(t)
            xs = "x=?" if x is None else f"x={x:+.3f}"
            self.tgt_list.insert(
                "end", f"{name:<12} {t['plane']:<8} {xs} "
                       f"y={t['y']:+.3f} z={t['z']:+.3f} yaw {yaw}"
                       + ("  (*)" if note else "")
                       + ("  [티칭]" if t.get("taught") else ""))
        if hasattr(self, "map"):
            self._draw_markers()

    def _store_point(self, name, plane, x, y, z, yaw):
        if name in self.cfg["targets"] and not messagebox.askyesno(
                "Nero — 이름 중복", f"포인트 '{name}'을(를) 덮어쓸까요?",
                parent=self.root):
            return False
        self.cfg["targets"][name] = {"plane": plane, "x": float(x),
                                     "y": float(y), "z": float(z),
                                     "yaw": None if yaw is None
                                     else float(yaw)}
        save_cfg(self.cfg)
        self._refresh_targets()
        self.log_line(f"포인트 '{name}' 저장됨: {plane} x={x:g} y={y:.3f} "
                      f"z={z:.3f} yaw " + ("유지" if yaw is None
                                          else f"{yaw:g}°"))
        return True

    def _save_current_point(self):
        """Save the live TCP as a point on the nearest plane (yaw included)."""
        name = self.tgt_name.get().strip()
        if not name:
            self.log_line("!! 포인트 이름을 입력하세요")
            return
        if self.last_pos is None:
            self.log_line("!! 아직 TCP 위치를 읽지 못했습니다")
            return
        x, y, z = self.last_pos
        planes = self.cfg["planes"]
        near = min(planes, key=lambda n: abs(planes[n] - x))
        dx = abs(planes[near] - x)
        if dx > MAP_SLAB:
            self.log_line(f"!! 현재 TCP(x={x:.3f})가 어느 평면에도 없습니다 "
                          f"(가장 가까운 {near}와 {dx * 1000:.0f} mm 차이) — "
                          "먼저 평면으로 이동하거나 평면 높이를 조정하세요")
            return
        yaw = None if self.last_yaw is None else float(np.rad2deg(self.last_yaw))
        if dx > 1e-3:
            self.log_line(f"현재 높이 x={x:.3f} → {near}(x={planes[near]:g})"
                          f"로 저장 ({dx * 1000:.0f} mm 차이)")
        self._store_point(name, near, planes[near], y, z, yaw)

    def _preview_sel(self):
        """Selecting a point in the list shows it as the pending goal on the
        map (fields untouched)."""
        sel = self.tgt_list.curselection()
        if not sel:
            return
        t = self.cfg["targets"].get(self.tgt_list.get(sel[0]).split()[0])
        if t is None:
            return
        self._goal_yz = (t["y"], t["z"])
        self._draw_markers()

    def _plan_to_point(self):
        """Plan from the current pose to the selected point with the
        point-move mode: load it into the fields, sync the mode, plan."""
        name = self._sel_target()
        if name is None:
            return
        if self._busy:
            self.log_line("!! 실행 중입니다 — 끝난 뒤 다시 시도하세요")
            return
        self._load_target(name)
        mode = self.pt_mode_var.get()
        self.mode_var.set(mode)
        t = self.cfg["targets"][name]
        if mode == "vertical":
            self.log_line(f"수직 이동: 포인트 '{name}'의 y·z는 쓰지 않고 "
                          f"{t['plane']} 높이로만 갑니다")
        self._do_plan()

    def _save_target(self):
        name = self.tgt_name.get().strip()
        if not name:
            self.log_line("!! 포인트 이름을 입력하세요")
            return
        try:
            plane, p, yaw = self._goal_from_fields()
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        self._store_point(name, plane, p[0], p[1], p[2], yaw)

    def _sel_target(self):
        sel = self.tgt_list.curselection()
        if not sel:
            self.log_line("!! 목록에서 포인트를 먼저 선택하세요")
            return None
        return self.tgt_list.get(sel[0]).split()[0]

    def _load_target(self, name=None):
        if name is None:
            name = self._sel_target()
            if name is None:
                return
        t = self.cfg["targets"][name]
        x, note = self._point_x(t)
        if t["plane"] in self.plane_x:
            self.plane_var.set(t["plane"])
        else:
            self.log_line(f"!! 포인트 '{name}'의 평면 '{t['plane']}'이(가) "
                          "없습니다 — 평면 선택을 그대로 둡니다")
        self.goal_vars["y"].set(f"{t['y']:g}")
        self.goal_vars["z"].set(f"{t['z']:g}")
        self.goal_vars["yaw"].set("" if t.get("yaw") is None
                                  else f"{t['yaw']:g}")
        self.tgt_name.set(name)
        self.log_line(f"포인트 '{name}' → 필드로 불러옴"
                      + (f" ({note})" if note else ""))
        self._sync_goal_marker()

    def _del_target(self):
        name = self._sel_target()
        if name is None:
            return
        del self.cfg["targets"][name]
        save_cfg(self.cfg)
        self._refresh_targets()
        self.log_line(f"포인트 '{name}' 삭제됨")

    # ---------------------------------------------------------------- poll
    def _job_poll(self):
        qj = ml.read_current_joints(self.node, self.args.joint_states,
                                    timeout=0.8)
        # gripper status rides the same loop; give it a short window
        if not self.args.real or GripperStatus is not None:
            end = time.time() + 0.3
            while time.time() < end:
                st = self._grip_state()
                if st is not None and st["age"] < 0.5:
                    break
                rclpy.spin_once(self.node, timeout_sec=0.05)
        self.ui_q.put(("grip", self._grip_state()))
        src = ""
        if self._robot is not None:
            role = self._read_role()
            self.ui_q.put(("role", role))
            if qj is None and role == "LEADER":
                # no measured feedback while a leader: show the leader
                # broadcast so you can see where the hand-guided tcp is
                ja, s_ = sj.read_joints(self._robot, timeout=0.3)
                if ja is not None and len(ja.msg) >= 7:
                    qj = np.array(list(ja.msg)[:7], float)
                    src = "  [LEADER 방송값 — 실행 불가]"
        if qj is None:
            # say WHY: no publisher at all (driver/mock not running) vs. a
            # publisher that is silent (stalled)
            n_pub = self.node.count_publishers(self.args.joint_states)
            self.ui_q.put(("state", n_pub))
            return
        R_t, t_t = lp.tcp_transform(self.args.tcp_xyz, self.args.tcp_rpy)
        R, p, _, _ = lp.fk_tcp(qj[None], R_t, t_t)
        # tilt of the carrying axis (container-up = tcp -x) vs world up
        tilt = float(np.arccos(np.clip(
            (R[0] @ np.array([-1.0, 0.0, 0.0])) @ lp.UP, -1.0, 1.0)))
        self.ui_q.put(("state", (p[0], tilt, lp.abs_yaw(R[0]), src)))

    def _schedule_poll(self):
        if self.worker.idle():
            self.worker.submit(self._job_poll, silent=True)
        self.root.after(POLL_MS, self._schedule_poll)

    # ------------------------------------------------------------- UI pump
    def _pump(self):
        try:
            while True:
                item = self.ui_q.get_nowait()
                kind = item[0]
                if kind == "log":
                    self.log_line(item[1])
                elif kind == "busy":
                    self._busy = item[1]
                    self.busy_var.set("실행 중…" if item[1] else "")
                    self._update_buttons()
                elif kind == "plan":
                    self.last = item[1]
                    self.last_desc = item[2] if len(item) > 2 else ""
                    self._update_buttons()
                elif kind == "state":
                    self._show_state(item[1])
                elif kind == "role":
                    self._show_role(item[1])
                elif kind == "point":
                    self.cfg["targets"][item[1]] = item[2]
                    save_cfg(self.cfg)
                    self._refresh_targets()
                    self.tgt_name.set(item[1])
                elif kind == "grip":
                    self._show_grip(item[1])
                elif kind == "wit":
                    self._wit_p = item[1]
                    if self._wit_p is not None:
                        p = self._wit_p
                        self.map.set_bounds((p[:, 1].min() - 0.05,
                                             p[:, 1].max() + 0.05,
                                             p[:, 2].min() - 0.05,
                                             p[:, 2].max() + 0.05))
                    self._redraw_region()
                elif kind == "ask":
                    item[2].put(messagebox.askyesno(
                        "Nero — 진행할까요?", item[1], parent=self.root))
        except queue.Empty:
            pass
        finally:
            self.root.after(PUMP_MS, self._pump)

    def _show_state(self, state):
        if state is None or isinstance(state, int):
            if state is None or state == 0:
                why = ("퍼블리셔 없음 — 팔 드라이버가 안 떠 있음 (CAN 어댑터 "
                       "연결/can_activate 후 real.launch 재시작)"
                       if self.args.real else
                       "퍼블리셔 없음 — sim.launch(mock) 실행 중인지 확인")
            else:
                why = f"퍼블리셔 {state}개 있으나 메시지 없음 — 드라이버 정지?"
            self.cur_var.set(f"'{self.args.joint_states}' 조인트 없음: {why}")
            self.last_pos = None
            self.last_yaw = None
            return
        p, tilt, yaw, src = state
        self.last_pos = p
        self.last_yaw = yaw
        level = ("수평" if np.rad2deg(tilt) < 2.0
                 else f"기울어짐 {np.rad2deg(tilt):.0f}°")
        self.cur_var.set(f"x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:+.3f} m   "
                         f"yaw {np.rad2deg(yaw):+.1f}°   운반축 {level}{src}")
        self._draw_markers()

    def _show_role(self, role):
        self.last_role = role
        if not hasattr(self, "mode_lights"):
            return
        for name, (cv, oid) in self.mode_lights.items():
            cv.itemconfig(oid, fill=LIGHT_ON if role == name else LIGHT_OFF)
        if role == "LEADER":
            txt = "모드: LEADER — 손으로 이동 가능, 실행 버튼 잠김"
        elif role == "FOLLOWER":
            txt = "모드: FOLLOWER — 제어 상태"
        elif self._robot is None:
            txt = "모드: 미확인 (버튼을 누르면 연결)"
        else:
            txt = "모드: 미확인 — 피드백 없음 (팔 전원/CAN 확인)"
        self.role_var.set(txt)

    def _update_buttons(self):
        state = "disabled" if self._busy else "normal"
        for b in self._buttons:
            b.configure(state=state)
        self.stop_btn.configure(state="normal" if self._busy else "disabled")
        if not self._busy and self.last is None:
            self.prev_btn.configure(state="disabled")
            self.go_btn.configure(state="disabled")

    def _do_stop_plan(self):
        """Cancel the running plan computation (planning only: a trajectory
        already being executed by MoveIt is not touched)."""
        lp.CANCEL.set()
        self.log_line("계산 중지 요청 — 플래너가 다음 웨이포인트에서 멈춥니다 "
                      "(실행 중인 로봇 동작은 이 버튼으로 멈추지 않음)")

    def _on_close(self):
        if self.worker.busy and not messagebox.askyesno(
                "Nero — 종료", "명령이 아직 실행 중입니다. 그래도 종료할까요?",
                parent=self.root):
            return
        self.worker.submit(None)
        # let the worker leave rclpy.spin_once before the node is destroyed
        # (tearing down mid-spin aborts the process)
        self.worker.join(timeout=3.0)
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true",
                    help="work with the REAL arm (default: sim)")
    ap.add_argument("--joint-states", default=None)
    ap.add_argument("--lift", type=float, default=0.08)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--v", type=float, default=0.06)
    ap.add_argument("--a", type=float, default=0.2)
    ap.add_argument("--qdot", type=float, default=1.0)
    ap.add_argument("--qddot", type=float, default=3.0)
    ap.add_argument("--dwell", type=float, default=0.1,
                    help="hold at each path corner [s]; negative = no stop")
    ap.add_argument("--tcp-xyz", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--tcp-rpy", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--can", default="can0",
                    help="CAN channel for the direct teach connection "
                         "(leader/follower switching; default can0)")
    ap.add_argument("--fw", default="v112",
                    choices=["auto", "default", "v111", "v112", "v120"],
                    help="arm firmware parser for the teach connection "
                         "(default v112; 'auto' probes, slower)")
    ap.add_argument("--grip-open", type=float, default=GRIP_OPEN_DEFAULT,
                    help="'open' target / slot upper bound [m] when no taught "
                         "range exists (default 0.1 = 100 mm)")
    ap.add_argument("--save", default=os.path.join(ml.OUT, "level_move.npz"))
    args = ap.parse_args()
    if args.joint_states is None:
        args.joint_states = ("feedback/joint_states" if args.real
                             else "control/joint_states")

    rclpy.init()
    node = Node("nero_level_gui")
    root = tk.Tk()
    gui = LevelGui(root, node, args)
    gui.log_line(f"[level_gui] {'REAL' if args.real else 'SIM'} 모드, "
                 f"조인트 토픽 '{args.joint_states}', 설정 {CFG_PATH}")
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
