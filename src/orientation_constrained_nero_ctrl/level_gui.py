#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tkinter GUI for level-constrained container moves — a shell over move_level.

Two horizontal work planes (world up = base_link +X, so a plane is a fixed x;
plane_1 is the LOWER one). Pick a plane, then either type the in-plane
coordinates (y, z) or click them on the top-down plane map (double-click =
plan immediately); the map shades the reachable region of the selected plane
from the Monte-Carlo workspace study (out/workspace_B_link7.npz) and shows
the base origin, live TCP, saved targets and the pending goal. Yaw change is
optional. Named targets and the plane heights persist in
src/data/level_targets.json.

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
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import level_path as lp  # noqa: E402
import move_level as ml  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402

CFG_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    "level_targets.json"))
POLL_MS = 2000        # live TCP pose refresh period (read-only)
PUMP_MS = 50          # UI queue pump period
DANGER_BG = "#c62828"
DANGER_ACTIVE = "#8e0000"
PARAMS = ("lift", "step", "v", "a", "qdot", "qddot")
MAP_SIZE = 400        # px, square top-down map of the work plane
MAP_SLAB = 0.025      # m, half-thickness of the plane slab for reachability
MAP_CELL = 0.01       # m, reachability shading cell size


def load_cfg():
    try:
        with open(CFG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    cfg.setdefault("planes", {"plane_1": -0.36, "plane_2": -0.31})
    cfg.setdefault("targets", {})
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

    def draw_markers(self, tcp, goal, targets):
        """tcp: (y, z, on_plane) or None; goal: (y, z) or None;
        targets: iterable of (name, y, z, on_selected_plane)."""
        self.canvas.delete("marker")
        c = self.canvas
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
        self.last_pos = None      # tcp position from the poll
        self._wit_p = "loading"   # witness tcp positions for the map region
        self._goal_yz = None      # pending goal marker on the map
        self._buttons = []
        self._busy = False

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
        pad = {"padx": 6, "pady": 3}

        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)
        ttk.Label(top, text=f"모드: {mode}   (플랜은 항상 현재 자세에서 시작)")\
            .pack(side="left")
        self.busy_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.busy_var, foreground="#b00")\
            .pack(side="right")

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
                                             "x 값이 평면 높이)")
        pf.pack(fill="x", **pad)
        row = ttk.Frame(pf)
        row.pack(fill="x", padx=6, pady=3)
        self.plane_var = tk.StringVar(value="plane_1")
        self.plane_x = {}
        for name, label in (("plane_1", "plane_1 (아래)"),
                            ("plane_2", "plane_2 (위)")):
            ttk.Radiobutton(row, text=label, value=name,
                            variable=self.plane_var).pack(side="left")
            v = tk.StringVar(value=f"{self.cfg['planes'][name]:g}")
            ttk.Entry(row, textvariable=v, width=8).pack(side="left",
                                                         padx=(2, 14))
            self.plane_x[name] = v
        self._btn(row, "평면 저장", self._save_planes).pack(side="left")

        # goal ------------------------------------------------------------
        gf = ttk.LabelFrame(self.root, text="목표 (평면 내 좌표, m / yaw는 "
                                             "시작 대비 변화량 deg, 비우면 유지)")
        gf.pack(fill="x", **pad)
        row = ttk.Frame(gf)
        row.pack(fill="x", padx=6, pady=3)
        self.goal_vars = {}
        for key, label, width in (("y", "y", 8), ("z", "z", 8),
                                  ("yaw", "yaw°", 6)):
            ttk.Label(row, text=label).pack(side="left")
            v = tk.StringVar()
            ttk.Entry(row, textvariable=v, width=width)\
                .pack(side="left", padx=(2, 10))
            self.goal_vars[key] = v
        self._btn(row, "현재 위치 채우기", self._fill_current)\
            .pack(side="left", padx=4)
        row2 = ttk.Frame(gf)
        row2.pack(fill="x", padx=6, pady=(0, 4))
        self._btn(row2, "플랜", self._do_plan).pack(side="left")
        self.prev_btn = self._btn(row2, "프리뷰", self._do_preview)
        self.prev_btn.pack(side="left", padx=4)
        self.go_btn = self._btn(row2, "실행 (go)", self._do_go, danger=True)
        self.go_btn.pack(side="left")
        self._btn(row2, "Carry (운반 자세)", self._do_carry, danger=True)\
            .pack(side="left", padx=14)
        ttk.Label(row2, text="← 액체 없이! 수평 비보장 셋업 무브")\
            .pack(side="left")

        # map + saved targets ---------------------------------------------
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

        tf = ttk.LabelFrame(mid, text="저장된 목표 (더블클릭 = 필드로 "
                                      "불러오기)")
        tf.pack(side="left", fill="both", expand=True)
        left = ttk.Frame(tf)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        self.tgt_list = tk.Listbox(left, height=5, exportselection=False,
                                   font=("TkFixedFont", 9))
        self.tgt_list.pack(fill="both", expand=True)
        self.tgt_list.bind("<Double-Button-1>", lambda _e: self._load_target())
        right = ttk.Frame(tf)
        right.pack(side="left", fill="y", padx=6, pady=4)
        ttk.Label(right, text="이름").pack(anchor="w")
        self.tgt_name = tk.StringVar()
        ttk.Entry(right, textvariable=self.tgt_name, width=14).pack(fill="x")
        self._btn(right, "현재 필드 저장", self._save_target)\
            .pack(fill="x", pady=2)
        self._btn(right, "불러오기", self._load_target).pack(fill="x")
        self._btn(right, "삭제", self._del_target).pack(fill="x", pady=2)

        # parameters ------------------------------------------------------
        prm = ttk.LabelFrame(self.root, text="파라미터 (lift/step m, v/a "
                                              "m/s·m/s², qdot/qddot rad/s·rad/s²)")
        prm.pack(fill="x", **pad)
        row = ttk.Frame(prm)
        row.pack(fill="x", padx=6, pady=3)
        self.param_vars = {}
        for name in PARAMS:
            ttk.Label(row, text=name).pack(side="left")
            v = tk.StringVar(value=f"{getattr(self.args, name):g}")
            ttk.Entry(row, textvariable=v, width=7)\
                .pack(side="left", padx=(2, 8))
            self.param_vars[name] = v

        self.log = ScrolledText(self.root, height=10, state="disabled",
                                font=("TkFixedFont", 9))
        self.log.pack(fill="both", expand=True, **pad)

        self._refresh_targets()
        self._update_buttons()
        self._redraw_region()

    # ------------------------------------------------------------------ map
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
        """Move the goal crosshair to whatever the y/z fields hold."""
        try:
            self._goal_yz = (self._float(self.goal_vars["y"], "y"),
                             self._float(self.goal_vars["z"], "z"))
        except ValueError:
            self._goal_yz = None
        self._draw_markers()

    _PLANE_STYLE = {          # region fill / label color per plane
        "plane_1": ("#cfe5f7", "#1565c0", "아래"),
        "plane_2": ("#d7ecd9", "#2e7d32", "위"),
    }

    def _redraw_region(self):
        name = self.plane_var.get()
        fill, accent, side = self._PLANE_STYLE[name]
        try:
            x = self._float(self.plane_x[name], f"{name} x")
        except ValueError:
            x = None
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
        self.map.draw_markers(tcp, self._goal_yz, targets)

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

    def _ask_from_worker(self, message):
        reply = queue.Queue()
        self.ui_q.put(("ask", message, reply))
        return reply.get()

    # ------------------------------------------------------------- actions
    def _save_planes(self):
        try:
            for name in self.plane_x:
                self.cfg["planes"][name] = self._float(self.plane_x[name],
                                                       f"{name} x")
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        if self.cfg["planes"]["plane_1"] >= self.cfg["planes"]["plane_2"]:
            self.log_line("!! plane_1은 plane_2보다 아래(더 작은 x)여야 합니다")
            return
        save_cfg(self.cfg)
        self.log_line(f"평면 저장됨: {self.cfg['planes']}")
        self._redraw_region()

    def _fill_current(self):
        if self.last_pos is None:
            self.log_line("!! 아직 TCP 위치를 읽지 못했습니다")
            return
        x, y, z = self.last_pos
        self.goal_vars["y"].set(f"{y:.3f}")
        self.goal_vars["z"].set(f"{z:.3f}")
        self.goal_vars["yaw"].set("")
        # pick whichever plane is closer to the current height
        planes = self.cfg["planes"]
        near = min(planes, key=lambda n: abs(planes[n] - x))
        self.plane_var.set(near)
        self.log_line(f"현재 위치 (x={x:.3f}) → {near} 선택됨 "
                      f"(평면과의 높이차 {abs(planes[near] - x)*1000:.0f} mm)")
        self._sync_goal_marker()

    def _goal_from_fields(self):
        plane, x = self._plane_height()
        y = self._float(self.goal_vars["y"], "y")
        z = self._float(self.goal_vars["z"], "z")
        yaw = self._float(self.goal_vars["yaw"], "yaw", optional=True)
        return plane, np.array([x, y, z]), yaw

    def _do_plan(self):
        try:
            self._sync_params()
            plane, p_goal, yaw = self._goal_from_fields()
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        a = self.args

        def job():
            print(f"플랜: {plane} x={p_goal[0]:g}, y={p_goal[1]:g}, "
                  f"z={p_goal[2]:g}"
                  + (f", yaw {yaw:+g}°" if yaw is not None else ""))
            q_start = ml.read_current_joints(self.node, a.joint_states)
            if q_start is None:
                print(f"!! '{a.joint_states}'에서 조인트 상태를 못 읽음 — "
                      "시뮬/드라이버 실행 중인지 확인")
                return
            R_goal = None
            if yaw is not None:
                R_t, t_t = lp.tcp_transform(a.tcp_xyz, a.tcp_rpy)
                R_cur, _, _, _ = lp.fk_tcp(q_start[None], R_t, t_t)
                R_goal = lp.rot_axis(lp.UP, np.deg2rad(yaw)) @ R_cur[0]
            planned = ml.make_plan(self.node, q_start, p_goal, R_goal, a)
            self.ui_q.put(("plan", planned))
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
        a = self.args

        def job():
            if not ml.exec_allowed(self.node, a):
                return
            mode = "REAL 로봇" if a.real else "시뮬"
            if not self._ask_from_worker(
                    f"{mode}에서 마지막 플랜({planned[1][-1]:.1f}초)을 "
                    "실행할까요?"):
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
            if not ml.exec_allowed(self.node, a):
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

    # ------------------------------------------------------------- targets
    def _refresh_targets(self):
        self.tgt_list.delete(0, "end")
        for name, t in sorted(self.cfg["targets"].items()):
            yaw = "유지" if t.get("yaw") is None else f"{t['yaw']:+g}°"
            self.tgt_list.insert(
                "end", f"{name:<12} {t['plane']:<8} "
                       f"y={t['y']:+.3f} z={t['z']:+.3f} yaw {yaw}")
        if hasattr(self, "map"):
            self._draw_markers()

    def _save_target(self):
        name = self.tgt_name.get().strip()
        if not name:
            self.log_line("!! 목표 이름을 입력하세요")
            return
        try:
            plane, p, yaw = self._goal_from_fields()
        except ValueError as exc:
            self.log_line(f"!! {exc}")
            return
        if name in self.cfg["targets"] and not messagebox.askyesno(
                "Nero — 이름 중복", f"목표 '{name}'을(를) 덮어쓸까요?",
                parent=self.root):
            return
        self.cfg["targets"][name] = {"plane": plane, "y": float(p[1]),
                                     "z": float(p[2]), "yaw": yaw}
        save_cfg(self.cfg)
        self._refresh_targets()
        self.log_line(f"목표 '{name}' 저장됨")

    def _sel_target(self):
        sel = self.tgt_list.curselection()
        if not sel:
            self.log_line("!! 목록에서 목표를 먼저 선택하세요")
            return None
        return self.tgt_list.get(sel[0]).split()[0]

    def _load_target(self):
        name = self._sel_target()
        if name is None:
            return
        t = self.cfg["targets"][name]
        self.plane_var.set(t["plane"])
        self.goal_vars["y"].set(f"{t['y']:g}")
        self.goal_vars["z"].set(f"{t['z']:g}")
        self.goal_vars["yaw"].set("" if t.get("yaw") is None
                                  else f"{t['yaw']:g}")
        self.tgt_name.set(name)
        self.log_line(f"목표 '{name}' → 필드로 불러옴 (플랜은 직접 누르세요)")
        self._sync_goal_marker()

    def _del_target(self):
        name = self._sel_target()
        if name is None:
            return
        del self.cfg["targets"][name]
        save_cfg(self.cfg)
        self._refresh_targets()
        self.log_line(f"목표 '{name}' 삭제됨")

    # ---------------------------------------------------------------- poll
    def _job_poll(self):
        qj = ml.read_current_joints(self.node, self.args.joint_states,
                                    timeout=0.8)
        if qj is None:
            self.ui_q.put(("state", None))
            return
        R_t, t_t = lp.tcp_transform(self.args.tcp_xyz, self.args.tcp_rpy)
        R, p, _, _ = lp.fk_tcp(qj[None], R_t, t_t)
        # tilt of the carrying axis (container-up = tcp -x) vs world up
        tilt = float(np.arccos(np.clip(
            (R[0] @ np.array([-1.0, 0.0, 0.0])) @ lp.UP, -1.0, 1.0)))
        self.ui_q.put(("state", (p[0], tilt)))

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
                    self._update_buttons()
                elif kind == "state":
                    self._show_state(item[1])
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
        if state is None:
            self.cur_var.set(f"'{self.args.joint_states}' 조인트 없음 — "
                             "시뮬/드라이버 확인")
            self.last_pos = None
            return
        p, tilt = state
        self.last_pos = p
        level = ("수평" if np.rad2deg(tilt) < 2.0
                 else f"기울어짐 {np.rad2deg(tilt):.0f}°")
        self.cur_var.set(f"x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:+.3f} m   "
                         f"운반축 {level}")
        self._draw_markers()

    def _update_buttons(self):
        state = "disabled" if self._busy else "normal"
        for b in self._buttons:
            b.configure(state=state)
        if not self._busy and self.last is None:
            self.prev_btn.configure(state="disabled")
            self.go_btn.configure(state="disabled")

    def _on_close(self):
        if self.worker.busy and not messagebox.askyesno(
                "Nero — 종료", "명령이 아직 실행 중입니다. 그래도 종료할까요?",
                parent=self.root):
            return
        self.worker.submit(None)
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
    ap.add_argument("--tcp-xyz", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--tcp-rpy", type=float, nargs=3, default=(0, 0, 0))
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
