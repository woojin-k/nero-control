#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tkinter GUI for carrying the container at a fixed angle to gravity — the
level_gui_2 map layout over tilt_control / move_level.

The tilt angle is given at launch (--tilt, degrees from level) and the
container is carried at that angle; the side it leans toward is --lean
(default front), so the lean direction turns with the yaw you command,
exactly like the container itself. There is no separate direction input:
yaw IS the direction.

LEFT (map side):
  현재 TCP      live position, absolute yaw, tilt angle and lean direction
  작업 평면     the work planes shared with level_gui_2 (src/data/
                level_targets.json: names + heights x; edit them there)
  지도          top-down map of the selected plane (base +Y right, +Z up):
                reachable region computed for the TILTED attitude (once per
                tilt/lean, cached in out/), base origin, live TCP, saved
                points and the pending goal. Click = set the goal (y, z),
                double-click = plan the move right away. Arrows show the
                lean direction: gray at the TCP (current), orange at the
                goal (at the goal yaw, or the current yaw if none given).
  목표          y, z on the plane (+ optional absolute yaw [deg]) and the
                move shape: 자동 (in-plane lift move when the TCP is on the
                plane, else one straight line), 평면 내, 직선
  포인트        the saved points (same file as level_gui_2): click to load,
                double-click to plan, save the current goal, delete
RIGHT:
  자세          the launch tilt/lean, and the setup buttons: Carry (setup
                joint move to the tilted carry pose — no liquid), 기울임
                적용 (in-place rotation from level/another tilt to --tilt,
                yaw held; a move plan prepends this by itself when the arm
                is not at --tilt yet — every move travels at --tilt)
  플랜/실행     플랜 (move to the goal at --tilt) / 프리뷰 (RViz
                Planned Path) / 실행 (asks first) / 계산 중지
  그리퍼        live status, 열기 / 닫기 (gripper_controller, as level_gui_2)
  파라미터, 로그

The in-place tilt rotates about tcp_link (--tcp-xyz/--tcp-rpy).
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
for _d in (_HERE, _LEVEL_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import level_path as lp  # noqa: E402
import move_level as ml  # noqa: E402
import tilt_path as tp  # noqa: E402
import tilt_control as tc  # noqa: E402
try:                          # the plane map + shared planes/points file
    import level_gui_2 as lg  # noqa: E402
except ImportError:
    import level_gui as lg  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from control_msgs.action import FollowJointTrajectory  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

POLL_MS = 2000
PUMP_MS = 50
DANGER_BG = "#c62828"
DANGER_ACTIVE = "#8e0000"
TILT_PARAMS = ("omega", "t_min", "dt", "qdot")
MOVE_PARAMS = ("lift", "step", "v", "a", "qddot", "dwell")
CUR_COLOR = "#757575"
TGT_COLOR = "#ef6c00"
ARROW_MIN = 0.03      # m, lean arrow length at 0 deg
ARROW_MAX = 0.15      # m, lean arrow length at 90 deg
MOVE_MODES = (("auto", "자동"), ("plane", "평면 내 (들어올림→수평→내림)"),
              ("line", "직선 (현재→목표)"))


class _LogWriter:
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


class TiltGui:
    def __init__(self, root, node, args):
        self.root = root
        self.node = node
        self.args = args
        self.cfg = lg.load_cfg()
        self.ui_q = queue.Queue()
        self.worker = Worker(self.ui_q)
        self.last = None
        self.last_desc = ""
        self.last_state = None      # (p, yaw, theta, lean dir or None)
        self._wit_p = "loading"
        self._goal_yz = None
        self._buttons = []
        self._busy = False
        self._confirm = True
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
        root.title(f"Nero Tilt Carry {args.tilt:g}° — {mode}")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build(mode)
        self.worker.start()
        self.worker.submit(self._job_poll, silent=True)
        self.worker.submit(self._job_load_witnesses, silent=True)
        root.after(PUMP_MS, self._pump)
        root.after(POLL_MS, self._schedule_poll)
        self._fit_window()

    # ---------------------------------------------------------------- build
    def _btn(self, parent, text, command, danger=False, **kw):
        if danger:
            b = tk.Button(parent, text=text, command=command, bg=DANGER_BG,
                          fg="white", activebackground=DANGER_ACTIVE,
                          activeforeground="white", **kw)
        else:
            b = ttk.Button(parent, text=text, command=command, **kw)
        self._buttons.append(b)
        return b

    def _pname(self, key):
        return self.cfg.get("plane_names", {}).get(key, key)

    def _build(self, mode):
        pad = dict(padx=6, pady=3)
        a = self.args
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", anchor="n")
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, anchor="n")

        # --- live state
        cur = ttk.LabelFrame(left, text=f"현재 TCP (자동 갱신) — {mode}")
        cur.pack(fill="x", **pad)
        self.state_var = tk.StringVar(value="읽는 중…")
        ttk.Label(cur, textvariable=self.state_var, font=("TkFixedFont", 10)
                  ).pack(side="left", padx=6, pady=3)
        self._btn(cur, "갱신",
                  lambda: self.worker.submit(self._job_poll, silent=True)
                  ).pack(side="right", padx=6, pady=2)

        # --- planes
        pf = ttk.LabelFrame(left, text="작업 평면 (x = 높이; 이름·높이 편집은 "
                                       "level_gui_2에서)")
        pf.pack(fill="x", **pad)
        row = ttk.Frame(pf)
        row.pack(fill="x", padx=6, pady=3)
        self.plane_var = tk.StringVar()
        order = sorted(lg.PLANE_NAMES, key=lambda k: self.cfg["planes"][k])
        for key in order:
            ttk.Radiobutton(
                row, text=f"{self._pname(key)} ({self.cfg['planes'][key]:g})",
                value=key, variable=self.plane_var).pack(side="left",
                                                         padx=(0, 8))
        self.plane_var.set(order[0])
        self.plane_var.trace_add("write", lambda *_: self._redraw_region())

        # --- map
        mf = ttk.LabelFrame(left, text="지도 (base +Y →, +Z ↑ · 클릭 = 목표, "
                                       "더블클릭 = 플랜 · 화살표 = 기울어진 방향)")
        mf.pack(fill="x", **pad)
        self.map = lg.PlaneMap(mf, self._map_click, self._map_dblclick)
        self.map.canvas.pack(padx=6, pady=(4, 0))
        ttk.Label(mf, textvariable=self.map.coord_var, foreground="#555"
                  ).pack(anchor="e", padx=8)

        # --- goal
        gf = ttk.LabelFrame(left, text="목표 (선택 평면 위 y z [m] · yaw 절대각 "
                                       "[°]: 0 = 정면 +Z, 비우면 현재 유지)")
        gf.pack(fill="x", **pad)
        row = ttk.Frame(gf)
        row.pack(fill="x", padx=6, pady=3)
        self.goal_vars = {}
        for key, label, width in (("y", "y", 8), ("z", "z", 8),
                                  ("yaw", "yaw°", 6)):
            ttk.Label(row, text=label).pack(side="left")
            v = tk.StringVar()
            e = ttk.Entry(row, textvariable=v, width=width)
            e.pack(side="left", padx=(2, 10))
            e.bind("<Return>", lambda _e: self._sync_goal_marker())
            e.bind("<FocusOut>", lambda _e: self._sync_goal_marker())
            self.goal_vars[key] = v
        self._btn(row, "현재 위치 채우기", self._fill_current).pack(side="left")
        row = ttk.Frame(gf)
        row.pack(fill="x", padx=6, pady=(0, 3))
        ttk.Label(row, text="이동 방식").pack(side="left", padx=(0, 6))
        self.mode_var = tk.StringVar(value="auto")
        for key, label in MOVE_MODES:
            ttk.Radiobutton(row, text=label, value=key, variable=self.mode_var
                            ).pack(side="left", padx=(0, 8))
        self.mode_var.trace_add("write", lambda *_: self._draw_markers())

        # --- points
        tf = ttk.LabelFrame(left, text="포인트 (level_gui_2와 공유 · 클릭 = "
                                       "불러오기, 더블클릭 = 플랜)")
        tf.pack(fill="x", **pad)
        row = ttk.Frame(tf)
        row.pack(fill="x", padx=6, pady=3)
        self.tgt_list = tk.Listbox(row, height=5, width=44, exportselection=False)
        self.tgt_list.pack(side="left", fill="x", expand=True)
        self.tgt_list.bind("<<ListboxSelect>>", lambda _e: self._load_point())
        self.tgt_list.bind("<Double-Button-1>", lambda _e: self._do_plan())
        col = ttk.Frame(row)
        col.pack(side="left", padx=6, fill="y")
        self.tgt_name = tk.StringVar()
        ttk.Entry(col, textvariable=self.tgt_name, width=12).pack(anchor="w")
        self._btn(col, "목표를 포인트로 저장", self._save_point).pack(anchor="w",
                                                                pady=2)
        self._btn(col, "포인트 삭제", self._delete_point).pack(anchor="w")
        self._refresh_targets()

        # --- attitude (launch-fixed) + setup moves
        at = ttk.LabelFrame(right, text="자세 (실행 인자로 고정)")
        at.pack(fill="x", **pad)
        ttk.Label(at, text=f"기울기 {a.tilt:g}°  (중력 방향 기준, --tilt)   ·   "
                           f"기울어지는 쪽: 컨테이너 {tp.LEAN_KO[a.lean]} "
                           f"(--lean {a.lean})   ·   방향은 yaw로 정합니다",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=6,
                                                           pady=(4, 2))
        row = ttk.Frame(at)
        row.pack(fill="x", padx=6, pady=(0, 4))
        self._btn(row, f"Carry (운반 자세, {a.tilt:g}° 포함)", self._do_carry,
                  danger=True).pack(side="left", padx=2)
        self._btn(row, f"기울임 적용 ({a.tilt:g}°)", self._do_tilt
                  ).pack(side="left", padx=(12, 2))
        ttk.Label(at, text="Carry = 액체 없이 하는 관절 셋업 무브 · 기울임 적용 = "
                           "제자리에서 tcp 점 기준 회전, yaw 유지 (이동 플랜은 "
                           "필요하면 이것을 자동으로 앞에 붙임)",
                  foreground="#555").pack(anchor="w", padx=6, pady=(0, 3))

        # --- plan / run
        rf = ttk.LabelFrame(right, text=f"플랜 / 실행 (모든 이동은 기울기 "
                                        f"{a.tilt:g}° 유지)")
        rf.pack(fill="x", **pad)
        br = ttk.Frame(rf)
        br.pack(fill="x", padx=6, pady=4)
        self._btn(br, "플랜", self._do_plan).pack(side="left", padx=2)
        self._btn(br, "프리뷰 (RViz)", self._do_preview).pack(side="left", padx=2)
        self.go_btn = self._btn(br, "실행 (go)", self._do_go, danger=True)
        self.go_btn.pack(side="left", padx=2)
        ttk.Button(br, text="계산 중지", command=self._do_stop_plan
                   ).pack(side="right", padx=2)
        self.confirm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rf, text="실행 전 확인 창", variable=self.confirm_var,
                        command=lambda: setattr(self, "_confirm",
                                                bool(self.confirm_var.get()))
                        ).pack(anchor="w", padx=6)
        self.plan_var = tk.StringVar(value="플랜 없음")
        ttk.Label(rf, textvariable=self.plan_var, wraplength=560
                  ).pack(anchor="w", padx=6, pady=2)

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

        # --- params
        pf = ttk.LabelFrame(right, text="파라미터")
        pf.pack(fill="x", **pad)
        self.param_vars = {}
        grid = ttk.Frame(pf)
        grid.pack(fill="x", padx=6, pady=2)
        labels = dict(omega="omega [°/s]", t_min="t_min [s]", dt="dt [s]",
                      qdot="qdot [rad/s]", lift="lift [m]", step="step [m]",
                      v="v [m/s]", a="a [m/s²]", qddot="qddot [rad/s²]",
                      dwell="dwell [s]")
        for i, name in enumerate(TILT_PARAMS + MOVE_PARAMS):
            r, c = divmod(i, 5)
            ttk.Label(grid, text=labels[name]).grid(row=r, column=2 * c,
                                                   sticky="e", padx=(6, 2))
            v = tk.StringVar(value=f"{getattr(a, name):g}")
            ttk.Entry(grid, textvariable=v, width=7).grid(row=r,
                                                          column=2 * c + 1)
            self.param_vars[name] = v

        # --- log
        lf = ttk.LabelFrame(right, text="로그")
        lf.pack(fill="both", expand=True, **pad)
        self.log = ScrolledText(lf, height=18, width=72, state="disabled",
                                font=("TkFixedFont", 9))
        self.log.pack(fill="both", expand=True, padx=4, pady=4)
        self._update_buttons()
        self._redraw_region()

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
        if not self._busy:
            self.go_btn.configure(
                state="normal" if self.last is not None else "disabled")

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

    def _sync_params(self):
        for name in TILT_PARAMS + MOVE_PARAMS:
            setattr(self.args, name, self._float(self.param_vars[name], name))

    def _ask_from_worker(self, message):
        if not self._confirm:
            print("(확인 창 꺼짐) " + " ".join(message.split()))
            return True
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

    def _goal_yaw(self):
        """Yaw [rad] the goal arrow is drawn at: the field, else current."""
        try:
            yaw = self._float(self.goal_vars["yaw"], "yaw", optional=True)
        except ValueError:
            yaw = None
        if yaw is not None:
            return np.deg2rad(yaw)
        return self.last_state[1] if self.last_state is not None else 0.0

    # ---------------------------------------------------------------- map
    def _plane_x(self):
        return float(self.cfg["planes"][self.plane_var.get()])

    def _map_click(self, y, z):
        self.goal_vars["y"].set(f"{y:.3f}")
        self.goal_vars["z"].set(f"{z:.3f}")
        self._goal_yz = (y, z)
        self._draw_markers()

    def _map_dblclick(self, y, z):
        if self._busy:
            self.log_line("!! 실행 중입니다 — 끝난 뒤 다시 시도하세요")
            return
        self._map_click(y, z)
        self._do_plan()

    def _sync_goal_marker(self):
        try:
            self._goal_yz = (self._float(self.goal_vars["y"], "y"),
                             self._float(self.goal_vars["z"], "z"))
        except ValueError:
            self._goal_yz = None
        self._draw_markers()

    def _redraw_region(self):
        key = self.plane_var.get()
        fill, accent = lg.PLANE_STYLE[key]
        x = self._plane_x()
        self.map.draw_label(f"{self._pname(key)}   x = {x:g} m   "
                            f"(도달영역: 기울기 {self.args.tilt:g}° 기준)", accent)
        if isinstance(self._wit_p, str):
            self.map.draw_region(None, msg="도달영역 계산 중…")
        elif self._wit_p is None:
            self.map.draw_region(None)
        else:
            m = np.abs(self._wit_p[:, 0] - x) < lg.MAP_SLAB
            yz = self._wit_p[m, 1:3]
            cells = (np.unique(np.round(yz / lg.MAP_CELL).astype(int), axis=0)
                     * lg.MAP_CELL if len(yz) else np.empty((0, 2)))
            self.map.draw_region(cells, color=fill)
        self._draw_markers()

    def _arrow(self, y, z, d, theta, color, width, tag):
        """Lean arrow on the map from (y, z) along horizontal direction d,
        length scaled with the tilt angle."""
        L = ARROW_MIN + (ARROW_MAX - ARROW_MIN) * min(theta, np.pi / 2) / (
            np.pi / 2)
        c = self.map.canvas
        x0, y0 = self.map._px(y, z)
        x1, y1 = self.map._px(y + L * d[1], z + L * d[2])
        c.create_line(x0, y0, x1, y1, fill=color, width=width, arrow="last",
                      arrowshape=(10, 12, 5), tags=tag)
        c.create_text(x1, y1 - 8, text=f"{np.rad2deg(theta):.0f}°", fill=color,
                      font=("TkDefaultFont", 8, "bold"), tags=tag)

    def _draw_markers(self):
        if not hasattr(self, "map"):
            return
        x = self._plane_x()
        tcp = None
        cur_theta = cur_d = None
        if self.last_state is not None:
            p, _, cur_theta, cur_d = self.last_state
            tcp = (p[1], p[2], abs(p[0] - x) < lg.MAP_SLAB)
        sel = self.plane_var.get()
        targets = [(n, t["y"], t["z"], t["plane"] == sel)
                   for n, t in sorted(self.cfg["targets"].items())]
        mode = self.mode_var.get() if hasattr(self, "mode_var") else "auto"
        self.map.draw_markers(tcp, self._goal_yz, targets,
                              path=(mode == "line"))
        c = self.map.canvas
        c.delete("tilt")
        if tcp is not None and cur_d is not None:
            self._arrow(tcp[0], tcp[1], cur_d, cur_theta, CUR_COLOR, 3, "tilt")
        if self._goal_yz is not None:
            d = tp.lean_world(self._goal_yaw(), self.args.lean)
            self._arrow(self._goal_yz[0], self._goal_yz[1], d,
                        np.deg2rad(self.args.tilt), TGT_COLOR, 3, "tilt")

    def _job_load_witnesses(self):
        """Reachable region for the TILTED attitude (computed once per
        tilt/lean and cached in out/)."""
        wit = tp.tilt_workspace(np.deg2rad(self.args.tilt), self.args.lean,
                                tcp_xyz=tuple(self.args.tcp_xyz),
                                tcp_rpy=tuple(self.args.tcp_rpy))
        self.ui_q.put(("wit", None if wit is None else wit[0]))

    # ------------------------------------------------------------- points
    def _refresh_targets(self):
        self.tgt_list.delete(0, "end")
        for n, t in sorted(self.cfg["targets"].items()):
            self.tgt_list.insert(
                "end", f"{n}  {self._pname(t['plane'])}  y {t['y']:+.3f}  "
                       f"z {t['z']:+.3f}  yaw {np.rad2deg(t.get('yaw', 0)):+.0f}°")

    def _selected_point(self):
        sel = self.tgt_list.curselection()
        if not sel:
            return None
        return self.tgt_list.get(sel[0]).split()[0]

    def _load_point(self):
        name = self._selected_point()
        if name is None or name not in self.cfg["targets"]:
            return
        t = self.cfg["targets"][name]
        self.tgt_name.set(name)
        if t["plane"] in lg.PLANE_NAMES:
            self.plane_var.set(t["plane"])
        self.goal_vars["y"].set(f"{t['y']:.3f}")
        self.goal_vars["z"].set(f"{t['z']:.3f}")
        self.goal_vars["yaw"].set(f"{np.rad2deg(t.get('yaw', 0)):.1f}")
        self._sync_goal_marker()
        self.log_line(f"포인트 '{name}' 불러옴 ({self._pname(t['plane'])}, "
                      f"y {t['y']:.3f}, z {t['z']:.3f})")

    def _save_point(self):
        name = self.tgt_name.get().strip()
        if not name:
            self.log_line("!! 포인트 이름을 입력하세요")
            return
        try:
            y = self._float(self.goal_vars["y"], "y")
            z = self._float(self.goal_vars["z"], "z")
            yaw = self._float(self.goal_vars["yaw"], "yaw", optional=True)
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        key = self.plane_var.get()
        if yaw is None:
            yaw = np.rad2deg(self.last_state[1]) if self.last_state else 0.0
        self.cfg["targets"][name] = dict(plane=key, x=self._plane_x(), y=y,
                                         z=z, yaw=float(np.deg2rad(yaw)))
        lg.save_cfg(self.cfg)
        self._refresh_targets()
        self._draw_markers()
        self.log_line(f"포인트 '{name}' 저장 ({self._pname(key)}, y {y:.3f}, "
                      f"z {z:.3f}, yaw {yaw:.1f}°)")

    def _delete_point(self):
        name = self._selected_point() or self.tgt_name.get().strip()
        if name not in self.cfg["targets"]:
            self.log_line("!! 삭제할 포인트를 목록에서 고르세요")
            return
        if not messagebox.askyesno("Nero — 포인트 삭제",
                                   f"포인트 '{name}'을(를) 삭제할까요?",
                                   parent=self.root):
            return
        del self.cfg["targets"][name]
        lg.save_cfg(self.cfg)
        self._refresh_targets()
        self._draw_markers()
        self.log_line(f"포인트 '{name}' 삭제")

    # ------------------------------------------------------------ actions
    def _fill_current(self):
        if self.last_state is None:
            self.log_line("!! 현재 TCP를 아직 못 읽었습니다")
            return
        p, yaw, _, _ = self.last_state
        self.goal_vars["y"].set(f"{p[1]:.3f}")
        self.goal_vars["z"].set(f"{p[2]:.3f}")
        self.goal_vars["yaw"].set(f"{np.rad2deg(yaw):.1f}")
        planes = self.cfg["planes"]
        near = min(planes, key=lambda k: abs(planes[k] - p[0]))
        if abs(planes[near] - p[0]) < lg.MAP_SLAB:
            self.plane_var.set(near)
        self._sync_goal_marker()

    def _do_carry(self):
        try:
            self._sync_params()
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        a = self.args

        def job():
            qj = self._read_joints()
            if qj is None:
                return
            desc = (f"Carry 운반 자세 (기울기 {a.tilt:g}° {tp.LEAN_KO[a.lean]}) "
                    "— 관절 무브, 자세 비보장, 액체 없이")
            print("플랜 — " + desc)
            planned = tc.make_carry(self.node, qj, a)
            self.ui_q.put(("plan", planned, desc))
            if planned is not None:
                print(f"플랜 OK: {planned[1][-1]:.1f}초 — [실행 (go)]")
        self.worker.submit(job)

    def _do_tilt(self, theta_deg=None):
        try:
            self._sync_params()
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        a = self.args
        theta = a.tilt if theta_deg is None else theta_deg

        def job():
            qj = self._read_joints()
            if qj is None:
                return
            desc = f"기울임 적용 {theta:g}° {tp.LEAN_KO[a.lean]} (제자리, yaw 유지)"
            print("플랜 — " + desc)
            planned = tc.make_tilt(self.node, qj, theta, a)
            self.ui_q.put(("plan", planned, desc))
            if planned is not None:
                print(f"플랜 OK: {planned[1][-1]:.1f}초 — [프리뷰]로 확인 후 "
                      "[실행 (go)]")
        self.worker.submit(job)

    def _do_plan(self):
        try:
            self._sync_params()
            y = self._float(self.goal_vars["y"], "y")
            z = self._float(self.goal_vars["z"], "z")
            yaw = self._float(self.goal_vars["yaw"], "yaw", optional=True)
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        key = self.plane_var.get()
        p_goal = np.array([self._plane_x(), y, z])
        mode = self.mode_var.get()
        a = self.args
        pname = self._pname(key)

        def job():
            qj = self._read_joints()
            if qj is None:
                return
            R_t, t_t = lp.tcp_transform(a.tcp_xyz, a.tcp_rpy)
            R_cur, p_cur, _, _ = lp.fk_tcp(qj[None], R_t, t_t)
            R_cur, p_cur = R_cur[0], p_cur[0]
            on_plane = abs(p_cur[0] - p_goal[0]) < lg.MAP_SLAB
            prof = mode
            if prof == "auto":
                prof = "plane" if on_plane else "line"
                print("이동 방식 자동: " + ("평면 내 (들어올림→수평→내림)"
                                     if on_plane else "직선 (평면 높이가 다름)"))
            elif prof == "plane" and not on_plane:
                print(f"!! 주의: 현재 TCP(x={p_cur[0]:.3f})가 {pname}"
                      f"(x={p_goal[0]:g})에 있지 않습니다 — 더 높은 쪽 위로 "
                      "들어올려 옮깁니다")
            a.profile = "lift" if prof == "plane" else "straight"
            desc = (f"이동 {pname} x={p_goal[0]:g} y={y:.3f} z={z:.3f}"
                    + (f" yaw {yaw:g}°" if yaw is not None else "")
                    + f" (기울기 {a.tilt:g}°)")
            print("플랜 — " + desc)
            planned = tc.make_move(self.node, qj, p_goal, yaw, a)
            self.ui_q.put(("plan", planned, desc))
            if planned is not None:
                print(f"플랜 OK: {planned[1][-1]:.1f}초 — [프리뷰]로 확인 후 "
                      "[실행 (go)]")
        self.worker.submit(job)

    def _do_preview(self):
        if self.last is None:
            self.log_line("!! 플랜이 없습니다")
            return
        planned = self.last
        self.worker.submit(lambda: ml.preview(self.node, *planned))

    def _do_go(self):
        if self.last is None:
            self.log_line("!! 플랜이 없습니다")
            return
        planned, desc, a = self.last, self.last_desc, self.args

        def job():
            if not ml.exec_allowed(self.node, a):
                return
            mode = "REAL 로봇" if a.real else "시뮬"
            if not self._ask_from_worker(
                    f"{mode}에서 마지막 플랜({planned[1][-1]:.1f}초)을 "
                    f"실행할까요?\n{desc}"):
                print("실행 취소됨")
                return
            ok = ml.execute_moveit(self.node, ml.to_joint_trajectory(*planned))
            print("실행 완료" if ok else "!! 실행 실패 — 터미널 로그 확인")
            self.ui_q.put(("plan", None, ""))
        self.worker.submit(job)

    def _do_stop_plan(self):
        lp.CANCEL.set()
        self.log_line("계산 중지 요청 (실행 중인 로봇 동작은 멈추지 않음)")

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
        R_t, t_t = lp.tcp_transform(self.args.tcp_xyz, self.args.tcp_rpy)
        R, p, _, _ = lp.fk_tcp(qj[None], R_t, t_t)
        theta, d = tp.tilt_state(R[0])
        self.ui_q.put(("state", (p[0], lp.abs_yaw(R[0]), theta, d)))

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
        else:
            p, yaw, theta, d = state
            self.last_state = state
            lean = "-" if d is None else f"방위 {tp.azimuth_of(d):+.0f}°"
            mark = ""
            if abs(theta - np.deg2rad(self.args.tilt)) > tc.TILT_TOL:
                mark = f"   ≠ {self.args.tilt:g}° (이동 시 자동 적용)"
            self.state_var.set(
                f"x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:+.3f}  "
                f"yaw {np.rad2deg(yaw):+.1f}°\n"
                f"기울기 {np.rad2deg(theta):.2f}°  {lean}{mark}")
        self._draw_markers()

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
                elif kind == "plan":
                    self.last = item[1]
                    self.last_desc = item[2] if len(item) > 2 else ""
                    if self.last is None:
                        self.plan_var.set("플랜 없음")
                    else:
                        self.plan_var.set(f"플랜: {self.last_desc} — "
                                          f"{self.last[1][-1]:.1f}초")
                    self._update_buttons()
                elif kind == "state":
                    self._show_state(item[1])
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

    def _on_close(self):
        if self.worker.busy and not messagebox.askyesno(
                "Nero — 종료", "명령이 아직 실행 중입니다. 그래도 종료할까요?",
                parent=self.root):
            return
        self.worker.submit(None)
        self.worker.join(timeout=3.0)
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.add_common_args(ap)
    ap.add_argument("--grip-open", type=float, default=lg.GRIP_OPEN_DEFAULT,
                    help="'열기' 목표 폭 [m] (티칭 범위가 없을 때; 기본 0.1)")
    args = ap.parse_args()
    if args.joint_states is None:
        args.joint_states = ("feedback/joint_states" if args.real
                             else "control/joint_states")

    rclpy.init()
    node = Node("nero_tilt_gui")
    root = tk.Tk()
    gui = TiltGui(root, node, args)
    gui.log_line(f"[tilt_gui] {'REAL' if args.real else 'SIM'} 모드, 기울기 "
                 f"{args.tilt:g}° ({args.lean}), 조인트 토픽 "
                 f"'{args.joint_states}', 평면/포인트 {lg.CFG_PATH}")
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
