#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tkinter JOG pendant for the Nero TCP — step moves in position and
orientation, over the ROS2 / MoveIt stack (sim or real).

Every jog button click is ONE fixed-size step: the TCP (link7 + the fixed
tcp_offset, default 0 = the flange) moves a straight line by the selected
distance along a world axis, or turns by the selected angle about a world
axis through the TCP point. The step is planned right then from the live
joint state (jog_path.plan_line: straight-line waypoints, 6-DoF IK each,
TOPP-RA timing under the speed caps) and executed through MoveIt's
/execute_trajectory, exactly like move_level / level_gui.

World frame = base_link's axes. The arm is mounted sideways, so +X is UP
(against gravity); Y and Z are horizontal. Rotations: Rx turns about the
vertical (yaw), Ry / Rz tilt about the two horizontal axes.

    source /opt/ros/humble/setup.bash
    # sim:  ros2 launch ../orientation_constrained_nero_ctrl/sim.launch.py
    # real: ros2 launch ../orientation_constrained_nero_ctrl/real.launch.py
    python3 jog_gui.py            # SIM (control/joint_states)
    python3 jog_gui.py --real     # REAL (feedback/joint_states)

현재 TCP sits above the tabs and 로그 below them, so the live pose and the last
command stay visible whichever tab is open. 활성화 / 프리뷰만 are on the 조그
tab and govern every red button in the window, the 경로 tab included.

  [조그]
  현재 TCP     live position [m], roll/pitch/yaw [deg] (base_link, URDF
               R = Rz·Ry·Rx — the convention show_pose.py prints) and joints
  조그         활성화 = unlock the red buttons (asks once — the arm moves on
               every click after that); 프리뷰만 = send each move to RViz's
               Planned Path display instead of executing (works locked);
               step size radios; the ± buttons per axis
  목표 자세    type an absolute pose and go there in one straight line
  저장된 자세   name + save the live posture as src/data/waypoints/<name>.json
               (the same format save_waypoint.py / goto_waypoint.py use, so a
               pose taught here is replayable from the CLI and vice versa);
               a selected one can be loaded into 목표 자세, driven to in a
               straight line, replayed in joint space, or deleted
  그리퍼       live status, 열기 / 닫기 (gripper_controller, as level_gui_2)

  [경로]  (path_tab.py, stored by nero_ctrl/paths.py)
               build an ordered program over saved poses: 이동 (a straight TCP
               line to a pose), 그리퍼 열기/닫기, 대기 N초, and 신호 (block
               until the operator presses 계속; the schema reserves a ROS topic
               source for when one exists). 계획 + 저장 solves the WHOLE chain
               once — each move starts from the joints the previous move ended
               on — and stores the joint waypoints in src/data/paths/. 실행
               replays them with no IK at all, retiming only, and refuses a
               move whose stored start is more than 5° from where the arm
               actually is. 정지 stops between steps and cancels a move already
               running on the controller.

  [파라미터]   v [m/s] and w [deg/s] speed caps, accel, joint limits, waypoint
               spacing. They are read live, so the same stored path can be
               replayed fast or slow without re-planning.

Sim mode refuses to execute while a real driver publishes
feedback/joint_states (move_level.exec_allowed); use --real then.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEVEL_DIR = os.path.join(os.path.dirname(_HERE),
                          "orientation_constrained_nero_ctrl")
_CTRL_DIR = os.path.join(os.path.dirname(_HERE), "nero_ctrl")
for _d in (_HERE, _LEVEL_DIR, _CTRL_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import jog_path as jp  # noqa: E402
import level_path as lp  # noqa: E402
import move_level as ml  # noqa: E402
import waypoints as wp  # noqa: E402  (saved poses, src/data/waypoints/*.json)
import paths as pp  # noqa: E402  (solved paths, src/data/paths/*.json)
import path_tab as pt  # noqa: E402  (the 경로 tab: editor + runner)
try:                          # gripper helpers + hand-taught jaw range
    import level_gui_2 as lg  # noqa: E402
except ImportError:
    import level_gui as lg  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from control_msgs.action import FollowJointTrajectory  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

POLL_MS = 1000
PUMP_MS = 50
DANGER_BG = "#c62828"
DANGER_ACTIVE = "#8e0000"
PARAMS = ("v", "w", "a", "qdot", "qddot", "step", "ang_step")
POS_STEPS_MM = (1.0, 5.0, 10.0, 50.0)
ROT_STEPS_DEG = (1.0, 5.0, 15.0, 45.0)
POS_LABEL = {"x": "X  위(+) / 아래(−)", "y": "Y", "z": "Z"}
ROT_LABEL = {"x": "Rx  수직축 회전 (yaw)", "y": "Ry  Y축 기울임",
             "z": "Rz  Z축 기울임"}


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
    """Serializes all node access; job prints go to the log pane."""

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
            except Exception as exc:  # noqa: BLE001
                self.ui_q.put(("log", f"!! {type(exc).__name__}: {exc}"))
            finally:
                out.flush()
                self.busy = False
                if not silent:
                    self.ui_q.put(("busy", False))


class JogGui:
    def __init__(self, root, node, args):
        self.root = root
        self.node = node
        self.args = args
        self.ui_q = queue.Queue()
        self.worker = Worker(self.ui_q)
        self.last_state = None      # (p, rpy, q) of the live TCP
        self._buttons = []          # everything disabled while a job runs
        self._jog_buttons = []      # additionally need 활성화 or 프리뷰만
        self._busy = False
        self._grip = None           # latest gripper status: (dict, wall time)
        self._grip_lock = threading.Lock()
        self._grip_client = None
        if args.real:
            if lg.GripperStatus is not None:
                node.create_subscription(lg.GripperStatus, lg.GRIP_STATUS_TOPIC,
                                         self._grip_cb, 10)
        else:
            node.create_subscription(JointState, lg.GRIP_MOCK_TOPIC,
                                     self._mock_grip_cb, 1)
        mode = "REAL" if args.real else "SIM"
        root.title(f"Nero JOG — {mode}")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build(mode)
        self.worker.start()
        self.worker.submit(self._job_poll, silent=True)
        root.after(PUMP_MS, self._pump)
        root.after(POLL_MS, self._schedule_poll)
        self._fit_window()

    # ---------------------------------------------------------------- build
    def _btn(self, parent, text, command, danger=False, jog=False, **kw):
        if danger:
            b = tk.Button(parent, text=text, command=command, bg=DANGER_BG,
                          fg="white", activebackground=DANGER_ACTIVE,
                          activeforeground="white", **kw)
        else:
            b = ttk.Button(parent, text=text, command=command, **kw)
        (self._jog_buttons if jog else self._buttons).append(b)
        return b

    def _build(self, mode):
        pad = dict(padx=6, pady=3)
        a = self.args
        # 현재 TCP and 로그 sit outside the notebook: whichever tab is open, you
        # always want to see where the arm is and what the last command said.
        head = ttk.Frame(self.root)
        head.pack(fill="x")
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=4, pady=2)
        tab_jog = ttk.Frame(nb)
        tab_path = ttk.Frame(nb)
        tab_param = ttk.Frame(nb)
        nb.add(tab_jog, text=" 조그 ")
        nb.add(tab_path, text=" 경로 ")
        nb.add(tab_param, text=" 파라미터 ")
        foot = ttk.Frame(self.root)
        foot.pack(fill="both", expand=True)
        body = ttk.Frame(tab_jog)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", anchor="n")
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, anchor="n")

        # --- live state
        cur = ttk.LabelFrame(head, text=f"현재 TCP (자동 갱신) — {mode} · "
                                        "base_link 좌표, 위 = +X")
        cur.pack(fill="x", **pad)
        self.state_var = tk.StringVar(value="읽는 중…")
        ttk.Label(cur, textvariable=self.state_var, font=("TkFixedFont", 10),
                  justify="left").pack(side="left", padx=6, pady=3)
        self._btn(cur, "갱신",
                  lambda: self.worker.submit(self._job_poll, silent=True)
                  ).pack(side="right", padx=6, pady=2)

        # --- jog
        jf = ttk.LabelFrame(left, text="조그 (클릭 1회 = 스텝 1회 이동)")
        jf.pack(fill="x", **pad)
        row = ttk.Frame(jf)
        row.pack(fill="x", padx=6, pady=3)
        self.arm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="활성화 (버튼 클릭 시 즉시 이동)",
                        variable=self.arm_var, command=self._on_arm
                        ).pack(side="left")
        self.preview_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="프리뷰만 (RViz, 실행 안 함)",
                        variable=self.preview_var,
                        command=self._update_buttons).pack(side="left",
                                                           padx=(12, 0))
        self.confirm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="매번 확인 창", variable=self.confirm_var
                        ).pack(side="left", padx=(12, 0))

        srow = ttk.Frame(jf)
        srow.pack(fill="x", padx=6, pady=(2, 0))
        ttk.Label(srow, text="위치 스텝 [mm]").pack(side="left")
        self.pos_step = tk.StringVar(value=f"{POS_STEPS_MM[2]:g}")
        for s in POS_STEPS_MM:
            ttk.Radiobutton(srow, text=f"{s:g}", value=f"{s:g}",
                            variable=self.pos_step).pack(side="left", padx=2)
        ttk.Radiobutton(srow, text="직접", value="custom",
                        variable=self.pos_step).pack(side="left", padx=(6, 0))
        self.pos_custom = tk.StringVar(value="20")
        ttk.Entry(srow, textvariable=self.pos_custom, width=6).pack(side="left")
        srow = ttk.Frame(jf)
        srow.pack(fill="x", padx=6, pady=(2, 4))
        ttk.Label(srow, text="회전 스텝 [°]  ").pack(side="left")
        self.rot_step = tk.StringVar(value=f"{ROT_STEPS_DEG[1]:g}")
        for s in ROT_STEPS_DEG:
            ttk.Radiobutton(srow, text=f"{s:g}", value=f"{s:g}",
                            variable=self.rot_step).pack(side="left", padx=2)
        ttk.Radiobutton(srow, text="직접", value="custom",
                        variable=self.rot_step).pack(side="left", padx=(6, 0))
        self.rot_custom = tk.StringVar(value="30")
        ttk.Entry(srow, textvariable=self.rot_custom, width=6).pack(side="left")

        grid = ttk.Frame(jf)
        grid.pack(fill="x", padx=6, pady=4)
        ttk.Label(grid, text="위치 (world 축, 직선)", font=("TkDefaultFont", 10,
                                                        "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 2))
        for i, ax in enumerate(("x", "y", "z")):
            ttk.Label(grid, text=POS_LABEL[ax]).grid(row=1 + i, column=0,
                                                     sticky="w", padx=(0, 8))
            self._btn(grid, "−", lambda ax=ax: self._do_jog("pos", ax, -1),
                      danger=True, jog=True, width=5).grid(row=1 + i, column=1,
                                                            padx=2, pady=1)
            self._btn(grid, "+", lambda ax=ax: self._do_jog("pos", ax, +1),
                      danger=True, jog=True, width=5).grid(row=1 + i, column=2,
                                                            padx=2, pady=1)
        ttk.Label(grid, text="회전 (world 축, TCP 점 중심)",
                  font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=4, columnspan=3, sticky="w", padx=(24, 0), pady=(0, 2))
        for i, ax in enumerate(("x", "y", "z")):
            ttk.Label(grid, text=ROT_LABEL[ax]).grid(row=1 + i, column=4,
                                                     sticky="w", padx=(24, 8))
            self._btn(grid, "−", lambda ax=ax: self._do_jog("rot", ax, -1),
                      danger=True, jog=True, width=5).grid(row=1 + i, column=5,
                                                            padx=2, pady=1)
            self._btn(grid, "+", lambda ax=ax: self._do_jog("rot", ax, +1),
                      danger=True, jog=True, width=5).grid(row=1 + i, column=6,
                                                            padx=2, pady=1)
        ttk.Label(jf, text="빨간 버튼 = 팔이 움직임 · 회전은 오른손 법칙 "
                           "(+ = 축을 보며 반시계)", foreground="#555"
                  ).pack(anchor="w", padx=6, pady=(0, 3))

        # --- absolute goal
        gf = ttk.LabelFrame(left, text="목표 자세로 이동 (base_link · x y z [m] "
                                       "· roll pitch yaw [°] · 직선 1회)")
        gf.pack(fill="x", **pad)
        row = ttk.Frame(gf)
        row.pack(fill="x", padx=6, pady=3)
        self.goal_vars = {}
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            ttk.Label(row, text=key).pack(side="left")
            v = tk.StringVar()
            ttk.Entry(row, textvariable=v, width=7).pack(side="left",
                                                        padx=(2, 8))
            self.goal_vars[key] = v
        row = ttk.Frame(gf)
        row.pack(fill="x", padx=6, pady=(0, 4))
        self._btn(row, "현재 값 채우기", self._fill_current).pack(side="left",
                                                               padx=2)
        self._btn(row, "이동", self._do_goto, danger=True, jog=True
                  ).pack(side="left", padx=(12, 2))
        ttk.Label(row, text="(조그와 같은 활성화/프리뷰 설정을 따름)",
                  foreground="#555").pack(side="left", padx=8)

        # --- gripper
        rec = lg.gr.load_range(mode="width")
        if rec is None:
            rng = (f"열기 = 최대 {a.grip_open * 1000:.0f} mm / 닫기 = 물체까지 "
                   "(힘 유지)")
        else:
            rng = (f"티칭 범위: 닫힘 {lg.gr.fmt(lg.gr.close_target(rec), 'width')}"
                   f" / 열림 {lg.gr.fmt(lg.gr.open_target(rec), 'width')}")
        grf = ttk.LabelFrame(right, text=f"그리퍼 — {rng}")
        grf.pack(fill="x", **pad)
        row = ttk.Frame(grf)
        row.pack(fill="x", padx=6, pady=3)
        self.grip_var = tk.StringVar(value="상태 없음")
        ttk.Label(row, textvariable=self.grip_var, font=("TkFixedFont", 10)
                  ).pack(side="left")
        row = ttk.Frame(grf)
        row.pack(fill="x", padx=6, pady=(0, 4))
        self._btn(row, "열기", lambda: self._do_grip("open"), danger=True
                  ).pack(side="left", padx=2)
        self._btn(row, "닫기", lambda: self._do_grip("close"), danger=True
                  ).pack(side="left", padx=2)
        ttk.Label(row, text="힘은 launch의 gripper_force · 닫힘에서 닫기 = 재파지",
                  foreground="#555").pack(side="left", padx=10)

        # --- saved poses
        self._build_waypoints(right, pad)

        # --- params (own tab: they apply to jogging and to path replay alike)
        pf = ttk.LabelFrame(tab_param, text="파라미터")
        pf.pack(fill="x", **pad)
        self.param_vars = {}
        grid = ttk.Frame(pf)
        grid.pack(fill="x", padx=6, pady=2)
        labels = dict(v="v 최대 속도 [m/s]", w="w 최대 각속도 [°/s]",
                      a="a 가속도 [m/s²]", qdot="qdot [rad/s]",
                      qddot="qddot [rad/s²]", step="경로점 간격 [m]",
                      ang_step="경로점 각도 간격 [°]")
        for i, name in enumerate(PARAMS):
            ttk.Label(grid, text=labels[name]).grid(row=i // 2, column=2 * (i % 2),
                                                    sticky="e", padx=(0, 4),
                                                    pady=1)
            v = tk.StringVar(value=f"{getattr(a, name):g}")
            ttk.Entry(grid, textvariable=v, width=8).grid(
                row=i // 2, column=2 * (i % 2) + 1, sticky="w", padx=(0, 14))
            self.param_vars[name] = v
        ttk.Label(pf, text="각 이동은 v·w·a·qdot·qddot 아래에서 TOPP-RA로 타이밍 "
                           "(toppra 없으면 사다리꼴)", foreground="#555"
                  ).pack(anchor="w", padx=6, pady=(0, 3))

        # --- log
        lf = ttk.LabelFrame(foot, text="로그")
        lf.pack(fill="both", expand=True, **pad)
        self.log = ScrolledText(lf, height=10, width=100, state="disabled",
                                font=("TkFixedFont", 9))
        self.log.pack(fill="both", expand=True, padx=6, pady=4)

        # --- 경로 tab, last: it registers buttons and wants self.log to exist
        self.path_tab = pt.PathTab(tab_path, self)
        self._wp_refresh()          # after self.log: a bad file logs a warning
        self.path_tab.refresh()
        self._update_buttons()

    def _build_waypoints(self, parent, pad):
        """저장된 자세 panel: save the live posture, list / inspect / replay."""
        wf = ttk.LabelFrame(parent, text="저장된 자세 (src/data/waypoints/*.json)")
        wf.pack(fill="x", **pad)

        row = ttk.Frame(wf)
        row.pack(fill="x", padx=6, pady=(3, 1))
        ttk.Label(row, text="이름").pack(side="left")
        self.wp_name = tk.StringVar()
        ttk.Entry(row, textvariable=self.wp_name, width=12).pack(side="left",
                                                                padx=(2, 8))
        ttk.Label(row, text="설명").pack(side="left")
        self.wp_desc = tk.StringVar()
        ttk.Entry(row, textvariable=self.wp_desc, width=24).pack(
            side="left", padx=(2, 8), fill="x", expand=True)
        self._btn(row, "현재 자세 저장", self._do_save_wp).pack(side="left", padx=2)

        lrow = ttk.Frame(wf)
        lrow.pack(fill="x", padx=6, pady=2)
        self.wp_list = tk.Listbox(lrow, height=7, width=52, exportselection=False,
                                  font=("TkFixedFont", 9),
                                  activestyle="dotbox")
        self.wp_list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lrow, orient="vertical", command=self.wp_list.yview)
        sb.pack(side="left", fill="y")
        self.wp_list.configure(yscrollcommand=sb.set)
        self.wp_list.bind("<<ListboxSelect>>", lambda _e: self._wp_show())
        self.wp_list.bind("<Double-Button-1>", lambda _e: self._do_load_wp())

        self.wp_detail = tk.StringVar(value="목록에서 하나를 선택하세요\n\n")
        ttk.Label(wf, textvariable=self.wp_detail, font=("TkFixedFont", 9),
                  justify="left", foreground="#333").pack(anchor="w", padx=6,
                                                          pady=(1, 2))

        row = ttk.Frame(wf)
        row.pack(fill="x", padx=6, pady=(0, 4))
        self._btn(row, "목표란에 불러오기", self._do_load_wp).pack(side="left", padx=2)
        self._btn(row, "이동 (직선)", self._do_wp_goto, danger=True, jog=True
                  ).pack(side="left", padx=(10, 2))
        self._btn(row, "이동 (관절)", self._do_wp_joint, danger=True, jog=True
                  ).pack(side="left", padx=2)
        self._btn(row, "새로고침", self._wp_refresh).pack(side="right", padx=2)
        self._btn(row, "삭제", self._do_delete_wp).pack(side="right", padx=2)
        ttk.Label(wf, text="직선 = TCP 직선 보간 (IK 필요) · 관절 = 저장된 관절각으로 "
                           "보간 (IK 없이 확실하지만 경로는 직선이 아님)",
                  foreground="#555", wraplength=580, justify="left"
                  ).pack(anchor="w", padx=6, pady=(0, 3))
        self._wp_cache = {}

    def _fit_window(self):
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(self.root.winfo_reqwidth() + 20, sw - 40)
        h = min(self.root.winfo_reqheight() + 20, sh - 80)
        self.root.geometry(f"{w}x{h}+{max((sw - w) // 2, 0)}+10")

    # ------------------------------------------------------------ helpers
    def _update_buttons(self):
        state = "disabled" if self._busy else "normal"
        for b in self._buttons:
            b.configure(state=state)
        jog_ok = (not self._busy
                  and (self.arm_var.get() or self.preview_var.get()))
        for b in self._jog_buttons:
            b.configure(state="normal" if jog_ok else "disabled")

    def log_line(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _float(self, var, name):
        s = var.get().strip()
        if not s:
            raise ValueError(f"'{name}' 값을 입력하세요")
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"'{name}'이(가) 숫자가 아닙니다: {s}") from None

    def _sync_params(self):
        for name in PARAMS:
            setattr(self.args, name, self._float(self.param_vars[name], name))
        if self.args.v <= 0 or self.args.w <= 0 or self.args.step <= 0 \
                or self.args.ang_step <= 0:
            raise ValueError("v, w, 경로점 간격은 0보다 커야 합니다")

    def _step_value(self, kind):
        """Selected jog step: metres for 'pos', radians for 'rot'."""
        if kind == "pos":
            sel, custom, unit = self.pos_step, self.pos_custom, "mm"
        else:
            sel, custom, unit = self.rot_step, self.rot_custom, "°"
        raw = sel.get()
        val = self._float(custom, f"직접 입력 스텝 [{unit}]") \
            if raw == "custom" else float(raw)
        if val <= 0:
            raise ValueError("스텝은 0보다 커야 합니다")
        return val / 1000.0 if kind == "pos" else float(np.deg2rad(val))

    def _ask_from_worker(self, message):
        reply = queue.Queue()
        self.ui_q.put(("ask", message, reply))
        return reply.get()

    def _read_joints(self):
        qj = ml.read_current_joints(self.node, self.args.joint_states,
                                    timeout=2.0)
        if qj is None:
            print(f"!! '{self.args.joint_states}'에서 조인트 상태를 못 읽음 — "
                  "시뮬/드라이버 실행 중인지 확인")
        return qj

    def _tcp(self):
        return lp.tcp_transform(self.args.tcp_xyz, self.args.tcp_rpy)

    # ------------------------------------------------------------ arming
    def _on_arm(self):
        if self.arm_var.get():
            mode = "REAL 로봇" if self.args.real else "시뮬"
            if not messagebox.askyesno(
                    "Nero — 조그 활성화",
                    f"{mode}에서 조그를 활성화할까요?\n\n활성화 후에는 빨간 버튼을 "
                    "누를 때마다 팔이 바로 움직입니다 (확인 창 없음 — 필요하면 "
                    "'매번 확인 창'을 켜세요).\n주변을 비우고 비상정지를 준비하세요.",
                    parent=self.root):
                self.arm_var.set(False)
            else:
                self.log_line(f"조그 활성화 ({mode})")
        else:
            self.log_line("조그 비활성화")
        self._update_buttons()

    # ------------------------------------------------------------ moves
    def _run_move(self, what, target_fn):
        """Plan a straight line from the live pose to target_fn(R0, p0) and
        preview or execute it. Runs on the worker."""
        try:
            self._sync_params()
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        a = self.args
        preview_only = bool(self.preview_var.get())
        confirm_each = bool(self.confirm_var.get())
        if not preview_only and not self.arm_var.get():
            self.log_line("!! 조그가 비활성화 상태입니다 — '활성화' 또는 '프리뷰만'")
            return

        def job():
            qj = self._read_joints()
            if qj is None:
                return
            R_t, t_t = self._tcp()
            R0, p0 = jp.pose_now(qj, R_t, t_t)
            R1, p1 = target_fn(R0, p0)
            t0 = time.time()
            plan = jp.plan_line(qj, R1, p1, step=a.step,
                                ang_step=np.deg2rad(a.ang_step),
                                R_t=R_t, t_t=t_t)
            if not plan["ok"]:
                print(f"!! {what}: {plan['msg']}")
                return
            t, q, qd, qdd = jp.retime(plan["q"], plan["s"], plan["ang"],
                                      v=a.v, w=a.w, a=a.a, qdot=a.qdot,
                                      qddot=a.qddot)
            rpy1 = np.rad2deg(jp.rpy_of(R1))
            print(f"{what}: {len(plan['q'])} 경로점, {t[-1]:.2f}초"
                  f"{'' if qd is not None else ' (사다리꼴)'}, "
                  f"계산 {time.time() - t0:.2f}초 → x {p1[0]:+.3f} y {p1[1]:+.3f} "
                  f"z {p1[2]:+.3f}  rpy {rpy1[0]:+.1f} {rpy1[1]:+.1f} "
                  f"{rpy1[2]:+.1f}°")
            if preview_only:
                ml.preview(self.node, q, t, qd, qdd)
                return
            if not ml.exec_allowed(self.node, a):
                return
            if confirm_each and not self._ask_from_worker(
                    f"{'REAL 로봇' if a.real else '시뮬'}: {what} "
                    f"({t[-1]:.2f}초) 실행할까요?"):
                print(f"{what} 취소됨")
                return
            ok = ml.execute_moveit(self.node,
                                   ml.to_joint_trajectory(q, t, qd, qdd))
            print(f"{what} 완료" if ok else f"!! {what} 실행 실패 — 터미널 로그 확인")
            self._job_poll()
        self.worker.submit(job)

    def _do_jog(self, kind, axis, sign):
        try:
            step = self._step_value(kind)
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        if kind == "pos":
            d = np.zeros(3)
            d["xyz".index(axis)] = sign * step
            what = f"조그 {axis.upper()} {sign * step * 1000:+g} mm"
            fn = lambda R0, p0: jp.jog_target(R0, p0, dpos=d)  # noqa: E731
        else:
            what = f"조그 R{axis} {np.rad2deg(sign * step):+g}°"
            fn = lambda R0, p0: jp.jog_target(  # noqa: E731
                R0, p0, axis=axis, dang=sign * step)
        self._run_move(what, fn)

    def _fill_current(self):
        if self.last_state is None:
            self.log_line("!! 현재 TCP를 아직 못 읽었습니다")
            return
        p, rpy, _ = self.last_state
        for key, val in zip(("x", "y", "z"), p):
            self.goal_vars[key].set(f"{val:.4f}")
        for key, val in zip(("roll", "pitch", "yaw"), np.rad2deg(rpy)):
            self.goal_vars[key].set(f"{val:.1f}")

    def _do_goto(self):
        try:
            xyz = [self._float(self.goal_vars[k], k) for k in ("x", "y", "z")]
            rpy = [np.deg2rad(self._float(self.goal_vars[k], k))
                   for k in ("roll", "pitch", "yaw")]
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        R1, p1 = jp.pose_from_rpy(xyz, rpy)
        self._run_move("목표 자세 이동", lambda R0, p0: (R1, p1))

    # ------------------------------------------------------- saved poses
    def _wp_line(self, name, rec):
        """One fixed-width list row: name + the saved TCP position."""
        c = rec.get("cartesian")
        tail = "  +관절" if rec.get("joints_deg") else ""
        if not c:
            return f"{name:<12} (관절만){tail}"
        return (f"{name:<12} x{c['x_mm'] / 1000:+.3f} y{c['y_mm'] / 1000:+.3f} "
                f"z{c['z_mm'] / 1000:+.3f}{tail}")

    def _wp_refresh(self):
        """Reload src/data/waypoints/ into the list, keeping the selection."""
        keep = self._wp_selected_name()
        self._wp_cache = {}
        bad = []
        for name in wp.list_waypoints():
            try:
                self._wp_cache[name] = wp.load_waypoint(name)
            except (OSError, ValueError) as exc:
                bad.append(f"{name} ({type(exc).__name__})")
        self.wp_list.delete(0, "end")
        for name in self._wp_cache:
            self.wp_list.insert("end", self._wp_line(name, self._wp_cache[name]))
        if bad:
            self.log_line("!! 읽지 못한 웨이포인트: " + ", ".join(bad))
        names = list(self._wp_cache)
        if getattr(self, "path_tab", None) is not None:
            self.path_tab.waypoint_names(names)
        if keep in names:
            i = names.index(keep)
            self.wp_list.selection_set(i)
            self.wp_list.see(i)
        self._wp_show()

    def _wp_selected_name(self):
        sel = self.wp_list.curselection()
        if not sel:
            return None
        names = list(self._wp_cache)
        i = sel[0]
        return names[i] if i < len(names) else None

    def _wp_selected(self, need_cartesian=False, need_joints=False):
        """(name, record) for the selected row, or None after logging why."""
        name = self._wp_selected_name()
        if name is None:
            self.log_line("!! 목록에서 자세를 먼저 선택하세요")
            return None
        rec = self._wp_cache[name]
        if need_cartesian and not rec.get("cartesian"):
            self.log_line(f"!! '{name}'에는 직교 좌표가 없습니다 — 관절 이동을 쓰세요")
            return None
        if need_joints and not rec.get("joints_deg"):
            self.log_line(f"!! '{name}'에는 관절각이 없습니다 — 직선 이동을 쓰세요")
            return None
        return name, rec

    def _wp_show(self):
        name = self._wp_selected_name()
        if name is None:
            self.wp_detail.set(("목록에서 하나를 선택하세요"
                                if self._wp_cache else
                                f"저장된 자세 없음 — {wp.WAYPOINTS_DIR}") + "\n\n")
            return
        rec = self._wp_cache[name]
        c = rec.get("cartesian")
        lines = []
        if c:
            lines.append(f"x {c['x_mm'] / 1000:+.4f}  y {c['y_mm'] / 1000:+.4f}  "
                         f"z {c['z_mm'] / 1000:+.4f} [m]   "
                         f"rpy {c['roll_deg']:+.2f} {c['pitch_deg']:+.2f} "
                         f"{c['yaw_deg']:+.2f} [°]")
        j = rec.get("joints_deg")
        if j:
            lines.append("joints [°] " + " ".join(f"{d:+6.1f}" for d in j))
        meta = [f"저장 {rec.get('saved_at', '?')}"]
        if rec.get("source"):
            meta.append(str(rec["source"]))
        if rec.get("description"):
            meta.append(str(rec["description"]))
        lines.append(" · ".join(meta))
        while len(lines) < 3:                 # keep the panel from resizing
            lines.append("")
        self.wp_detail.set("\n".join(lines[:3]))

    def _do_save_wp(self):
        name = self.wp_name.get().strip()
        err = wp.check_name(name)
        if err:
            self.log_line(f"!! 이름 오류: {err} (공백/'/'/'..' 없이, 숫자로 시작 금지)")
            return
        if self.last_state is None:
            self.log_line("!! 현재 TCP를 아직 못 읽었습니다 — 저장할 자세가 없습니다")
            return
        p, rpy, qj = self.last_state
        path = wp.waypoint_path(name)
        if os.path.exists(path) and not messagebox.askyesno(
                "Nero — 덮어쓸까요?",
                f"'{name}' 자세가 이미 있습니다.\n\n{path}\n\n현재 자세로 "
                "덮어쓸까요?", parent=self.root):
            self.log_line(f"'{name}' 저장 취소됨")
            return
        a = self.args
        rec = wp.make_waypoint(
            name, self.wp_desc.get().strip(),
            cartesian=wp.cartesian_from_pose([*p, *rpy]),
            joints_deg=wp.joints_deg_from_rad(qj),
            extra={
                "frame": "base_link",
                "rpy_convention": "Rz*Ry*Rx",
                "source": f"jog_gui {'real' if a.real else 'sim'}",
                "tcp_offset": {"xyz_m": [float(v) for v in a.tcp_xyz],
                               "rpy_rad": [float(v) for v in a.tcp_rpy]},
            })
        try:
            path, existed = wp.save_waypoint(rec, overwrite=True)
        except (OSError, ValueError) as exc:
            self.log_line(f"!! 저장 실패: {type(exc).__name__}: {exc}")
            return
        self.log_line(f"자세 '{name}' {'덮어씀' if existed else '저장'} → {path}")
        self.log_line("   " + wp.format_waypoint(rec))
        self.wp_desc.set("")
        self._wp_refresh()
        names = list(self._wp_cache)
        if name in names:
            self.wp_list.selection_clear(0, "end")
            self.wp_list.selection_set(names.index(name))
            self.wp_list.see(names.index(name))
            self._wp_show()

    def _do_delete_wp(self):
        got = self._wp_selected()
        if got is None:
            return
        name, _rec = got
        if not messagebox.askyesno(
                "Nero — 자세 삭제",
                f"저장된 자세 '{name}'을 삭제할까요?\n\n"
                f"{wp.waypoint_path(name)}\n\n되돌릴 수 없습니다.",
                parent=self.root):
            return
        try:
            path = wp.delete_waypoint(name)
        except (OSError, ValueError) as exc:
            self.log_line(f"!! 삭제 실패: {type(exc).__name__}: {exc}")
            return
        self.log_line(f"자세 '{name}' 삭제됨 ({path})")
        self._wp_refresh()

    def _do_load_wp(self):
        """Copy the selected pose into 목표 자세 — types only, never moves."""
        got = self._wp_selected(need_cartesian=True)
        if got is None:
            return
        name, rec = got
        c = rec["cartesian"]
        for key, val in (("x", c["x_mm"] / 1000), ("y", c["y_mm"] / 1000),
                         ("z", c["z_mm"] / 1000)):
            self.goal_vars[key].set(f"{val:.4f}")
        for key, val in (("roll", c["roll_deg"]), ("pitch", c["pitch_deg"]),
                         ("yaw", c["yaw_deg"])):
            self.goal_vars[key].set(f"{val:.1f}")
        self.log_line(f"'{name}'을 목표 자세 칸에 불러왔습니다 (이동은 별도 버튼)")

    def _do_wp_goto(self):
        got = self._wp_selected(need_cartesian=True)
        if got is None:
            return
        name, rec = got
        pose6 = wp.to_cartesian_target(rec)
        R1, p1 = jp.pose_from_rpy(pose6[:3], pose6[3:])
        self._run_move(f"자세 '{name}' 이동 (직선)", lambda R0, p0: (R1, p1))

    def _do_wp_joint(self):
        got = self._wp_selected(need_joints=True)
        if got is None:
            return
        name, rec = got
        self._run_joint_move(f"자세 '{name}' 이동 (관절)",
                             np.asarray(wp.to_joints_rad(rec), float))

    def _run_joint_move(self, what, q_goal):
        """Joint-space interpolation from the live joints to q_goal. No IK, so it
        reaches poses plan_line rejects — but the TCP path is not a line, which
        is why this one always asks before it executes."""
        try:
            self._sync_params()
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        a = self.args
        preview_only = bool(self.preview_var.get())
        if not preview_only and not self.arm_var.get():
            self.log_line("!! 조그가 비활성화 상태입니다 — '활성화' 또는 '프리뷰만'")
            return

        def job():
            qj = self._read_joints()
            if qj is None:
                return
            qj = np.asarray(qj, float)
            if q_goal.shape != qj.shape:
                print(f"!! {what}: 저장된 관절 {q_goal.size}개 ≠ 현재 {qj.size}개")
                return
            q, t, dq = ml._joint_interp(qj, q_goal, a.qdot)
            print(f"{what}: {len(q)} 경로점, {t[-1]:.2f}초, 최대 관절 변화 "
                  f"{np.rad2deg(dq):.1f}°")
            if preview_only:
                ml.preview(self.node, q, t)
                return
            if not ml.exec_allowed(self.node, a):
                return
            if not self._ask_from_worker(
                    f"{'REAL 로봇' if a.real else '시뮬'}: {what}\n\n"
                    f"관절 보간 {t[-1]:.2f}초, 최대 관절 변화 "
                    f"{np.rad2deg(dq):.1f}°.\nTCP는 직선으로 가지 않습니다 — "
                    "경로가 비어 있는지 확인하세요."):
                print(f"{what} 취소됨")
                return
            ok = ml.execute_moveit(self.node, ml.to_joint_trajectory(q, t))
            print(f"{what} 완료" if ok else f"!! {what} 실행 실패 — 터미널 로그 확인")
            self._job_poll()
        self.worker.submit(job)

    # ------------------------------------------------------------ gripper
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
        if lg.GRIP_JOINT not in msg.name:
            return
        w = float(msg.position[msg.name.index(lg.GRIP_JOINT)])
        snap = {"width": w, "force": 0.0, "enabled": True, "homed": True,
                "errors": [], "sim": True}
        with self._grip_lock:
            self._grip = (snap, time.time())

    def _grip_state(self):
        with self._grip_lock:
            g = self._grip
        if g is None:
            return None
        snap, t = g
        return dict(snap, age=time.time() - t)

    def _show_grip(self, st):
        if st is None:
            src = lg.GRIP_STATUS_TOPIC if self.args.real else lg.GRIP_MOCK_TOPIC
            self.grip_var.set(f"상태 없음 — '{src}' 수신 안 됨")
            return
        bits = [f"폭 {st['width'] * 1000:5.1f} mm", f"힘 {st['force']:.2f} N",
                "enable" if st["enabled"] else "DISABLED",
                "homed" if st["homed"] else "미호밍"]
        if st["errors"]:
            bits.append("오류: " + ",".join(st["errors"]))
        if st.get("sim"):
            bits.append("(시뮬 mock)")
        if st["age"] > lg.GRIP_STALE_S:
            bits.append(f"(마지막 수신 {st['age']:.0f}s 전 — 오래됨)")
        self.grip_var.set("  ".join(bits))

    def _grip_target(self, spec):
        rec = lg.gr.load_range(mode="width")
        if rec is not None:
            return (lg.gr.open_target(rec) if spec == "open"
                    else lg.gr.close_target(rec))
        hi = min(self.args.grip_open, lg.GRIP_WIDTH_MAX)
        return hi if spec == "open" else 0.0

    def _do_grip(self, spec):
        target = self._grip_target(spec)
        what = {"open": "열기", "close": "닫기"}[spec]
        a = self.args

        def job():
            if not ml.exec_allowed(self.node, a):
                return
            mode = "REAL 로봇" if a.real else "시뮬"
            if not self._ask_from_worker(
                    f"{mode} 그리퍼 {what}: 목표 {target * 1000:.1f} mm?\n"
                    "그리퍼가 움직입니다 — 손가락과 물건을 조에서 치우세요."):
                print(f"그리퍼 {what} 취소됨")
                return
            outcome, st = self._move_grip(target)
            label = what
            if outcome == "holding" and spec == "open" and st is not None:
                back = max(0.0, st["width"] - lg.GRIP_BACKOFF)
                print(f"그리퍼 열기: {st['width'] * 1000:.1f} mm에서 스톱 — "
                      f"{back * 1000:.1f} mm로 후퇴")
                outcome, st = self._move_grip(back)
                label = "열기 (최대)"
            self.ui_q.put(("grip", st))
            w = "?" if st is None else f"{st['width'] * 1000:.1f} mm"
            if outcome == "reached":
                print(f"그리퍼 {label} 완료: {w}")
            elif outcome == "holding" and spec == "close":
                print(f"그리퍼 닫기: {w}에서 물체 파지 — 힘 유지 중 (놓으려면 열기)")
            elif outcome == "holding":
                print(f"그리퍼 {what}: {w}에서 정지 — 걸림 (힘 유지 중)")
            elif outcome == "no_server":
                print(f"!! 그리퍼 {what}: '{lg.GRIP_ACTION}' 액션 서버 없음 — "
                      "sim.launch/real.launch의 gripper_controller 확인")
            elif outcome == "rejected":
                print(f"!! 그리퍼 {what}: gripper_controller가 목표를 거부")
            elif outcome == "no_status":
                src = lg.GRIP_STATUS_TOPIC if a.real else lg.GRIP_MOCK_TOPIC
                print(f"!! 그리퍼 {what}: '{src}' 상태가 안 옵니다 — "
                      + ("드라이버가 그리퍼를 못 잡았거나 CAN 끊김"
                         if a.real else "mock 브로드캐스터 확인"))
            else:
                print(f"!! 그리퍼 {what}: {lg.GRIP_TIMEOUT:.0f}s 안에 "
                      f"{target * 1000:.1f} mm 도달 못 함 (현재 {w})")
        self.worker.submit(job)

    def _move_grip(self, target):
        """Ramp the gripper joint to `target` via gripper_controller and wait
        for the jaws (same logic as level_gui_2). Returns (outcome, state)."""
        if self._grip_client is None:
            self._grip_client = ActionClient(self.node, FollowJointTrajectory,
                                             lg.GRIP_ACTION)
        if not self._grip_client.wait_for_server(timeout_sec=3.0):
            return "no_server", self._grip_state()
        st = self._grip_state()
        w0 = st["width"] if st is not None else target
        T = max(lg.GRIP_MIN_T, abs(target - w0) / lg.GRIP_SPEED)
        jt = JointTrajectory()
        jt.joint_names = [lg.GRIP_JOINT]
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
        start = time.time()
        last_w = None
        still = 0
        seen = False
        while time.time() - start < lg.GRIP_TIMEOUT:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            st = self._grip_state()
            if st is None or st["age"] > lg.GRIP_STALE_S:
                continue
            seen = True
            w = st["width"]
            if abs(w - target) <= lg.GRIP_TOL and time.time() - start >= T:
                end = time.time() + 0.3
                while time.time() < end:
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                return "reached", self._grip_state()
            if last_w is not None and abs(w - last_w) < lg.GRIP_TOL / 20:
                still += 1
                if still >= 12 and time.time() - start > T + 0.5:
                    return "holding", st
            else:
                still = 0
            last_w = w
        return ("timeout" if seen else "no_status"), self._grip_state()

    # --------------------------------------------------------------- poll
    def _job_poll(self):
        qj = ml.read_current_joints(self.node, self.args.joint_states,
                                    timeout=0.8)
        end = time.time() + 0.3          # gripper status rides the same loop
        while time.time() < end:
            st = self._grip_state()
            if st is not None and st["age"] < 0.5:
                break
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.ui_q.put(("grip", self._grip_state()))
        if qj is None:
            self.ui_q.put(("state", self.node.count_publishers(
                self.args.joint_states)))
            return
        R_t, t_t = self._tcp()
        R, p = jp.pose_now(qj, R_t, t_t)
        self.ui_q.put(("state", (p, np.array(jp.rpy_of(R)), qj)))

    def _schedule_poll(self):
        if self.worker.idle():
            self.worker.submit(self._job_poll, silent=True)
        self.root.after(POLL_MS, self._schedule_poll)

    def _show_state(self, state):
        if isinstance(state, int):
            self.last_state = None
            if state == 0:
                why = ("퍼블리셔 없음 — 드라이버가 안 떠 있음 (CAN 연결/"
                       "can_activate 후 real.launch 재시작)" if self.args.real
                       else "퍼블리셔 없음 — sim.launch 실행 중인지 확인")
            else:
                why = "퍼블리셔는 있으나 메시지 없음"
            self.state_var.set(f"'{self.args.joint_states}': {why}")
            return
        p, rpy, qj = state
        self.last_state = state
        d = np.rad2deg(rpy)
        joints = " ".join(f"{x:+6.1f}" for x in np.rad2deg(qj))
        self.state_var.set(
            f"x {p[0]:+.4f}  y {p[1]:+.4f}  z {p[2]:+.4f}  [m]\n"
            f"roll {d[0]:+7.2f}  pitch {d[1]:+7.2f}  yaw {d[2]:+7.2f}  [°]\n"
            f"joints [°] {joints}")

    def _pump(self):
        try:
            while True:
                item = self.ui_q.get_nowait()
                kind = item[0]
                if kind == "log":
                    self.log_line(item[1])
                elif kind == "busy":
                    self._busy = item[1]
                    self._update_buttons()
                elif kind == "state":
                    self._show_state(item[1])
                elif kind == "grip":
                    self._show_grip(item[1])
                elif kind == "pathsaved":
                    self.path_tab.adopt(item[1])
                elif kind == "pathstatus":
                    self.path_tab.status.set(item[1])
                elif kind == "pathrun":
                    self.path_tab._set_running(item[1])
                elif kind == "pathsignal":
                    self.path_tab._set_signal_wait(item[1])
                elif kind == "ask":
                    item[2].put(messagebox.askyesno(
                        "Nero — 진행할까요?", item[1], parent=self.root))
        except queue.Empty:
            pass
        finally:
            self.root.after(PUMP_MS, self._pump)

    def _on_close(self):
        if self.worker.busy and not messagebox.askyesno(
                "Nero — 종료", "명령이 아직 실행 중입니다. 그래도 종료할까요?",
                parent=self.root):
            return
        if getattr(self, "path_tab", None) is not None:
            self.path_tab._do_stop()      # cancel a running path before exiting
        self.worker.submit(None)
        self.worker.join(timeout=3.0)
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true",
                    help="work with the REAL arm (feedback/joint_states; "
                         "without it: sim, control/joint_states)")
    ap.add_argument("--joint-states", default=None,
                    help="current-state topic (overrides the --real default)")
    ap.add_argument("--tcp-xyz", type=float, nargs=3, default=(0, 0, 0),
                    help="tcp_offset xyz given to the MoveIt launch [m]")
    ap.add_argument("--tcp-rpy", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--v", type=float, default=0.05, help="max tcp speed [m/s]")
    ap.add_argument("--w", type=float, default=15.0,
                    help="max tcp angular speed [deg/s]")
    ap.add_argument("--a", type=float, default=0.2, help="max tcp accel [m/s^2]")
    ap.add_argument("--qdot", type=float, default=1.0,
                    help="max joint rate [rad/s]")
    ap.add_argument("--qddot", type=float, default=3.0,
                    help="max joint acceleration [rad/s^2]")
    ap.add_argument("--step", type=float, default=0.005,
                    help="waypoint spacing along the line [m]")
    ap.add_argument("--ang-step", type=float, default=2.0, dest="ang_step",
                    help="waypoint spacing in orientation [deg]")
    ap.add_argument("--grip-open", type=float, default=lg.GRIP_OPEN_DEFAULT,
                    help="'열기' 목표 폭 [m] (티칭 범위가 없을 때; 기본 0.1)")
    args = ap.parse_args()
    if args.joint_states is None:
        args.joint_states = ("feedback/joint_states" if args.real
                             else "control/joint_states")

    rclpy.init()
    node = Node("nero_jog_gui")
    root = tk.Tk()
    gui = JogGui(root, node, args)
    gui.log_line(f"[jog_gui] {'REAL' if args.real else 'SIM'} 모드, 조인트 토픽 "
                 f"'{args.joint_states}' · 조그는 '활성화'를 켜야 움직입니다")
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
