#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tkinter GUI for the Nero arm — a thin shell over the SAME cores as the CLI.

Every button builds the exact command line the CLI would take and feeds it to
nero_cli.dispatch() on a single worker thread (one shared CAN connection, one
command at a time). Actuating commands pop the same y/N warning as the console,
as a popup. Nothing here reimplements arm logic; sequences recorded in the GUI
replay identically in the CLI and vice versa.

Running this does NOT enable motors or move anything on its own; actuation only
happens when you press a moving button and confirm the popup.

Usage:
    ./.venv/bin/python src/nero_ctrl/nero_gui.py --can can0 --fw v112
"""
import argparse
import contextlib
import math
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# Reusable cores + the CLI dispatcher (sibling modules in src/nero_ctrl/).
import connect_check as cc
import show_pose as sp
import show_joints as sj
import waypoints as wp
import sequences as sq
import nero_cli as cli


POLL_MS = 1500        # live status/pose/joints refresh period (read-only)
PUMP_MS = 50          # UI queue pump period

# Red = this button can make the arm/gripper move (it always pops a confirm).
DANGER_BG = "#c62828"
DANGER_ACTIVE = "#8e0000"
LIGHT_ON = "#00c853"   # linkage-mode lamp: current mode
LIGHT_OFF = "#777777"  # linkage-mode lamp: inactive / unknown


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


class RobotWorker(threading.Thread):
    """Serializes ALL robot access: dispatched commands and silent polls."""

    def __init__(self, robot, state, ui_q):
        super().__init__(daemon=True)
        self.robot = robot
        self.state = state
        self.ui_q = ui_q
        self.jobs = queue.Queue()
        self.busy = False

    def submit(self, kind, payload=None):
        self.jobs.put((kind, payload))

    def idle(self):
        return not self.busy and self.jobs.empty()

    def run(self):
        while True:
            kind, payload = self.jobs.get()
            if kind == "quit":
                return
            self.busy = True
            try:
                if kind == "cmd":
                    self._run_cmd(payload)
                elif kind == "multi":
                    self._run_multi(payload)
                elif kind == "poll":
                    self._run_poll()
            finally:
                self.busy = False

    def _run_cmd(self, payload):
        line, auto = payload
        self.ui_q.put(("echo", line))
        self.ui_q.put(("busy", True))
        out = _LogWriter(self.ui_q)
        if auto:                      # frontend decided: no confirm popup
            self.state["auto_yes"] = True
        try:
            with contextlib.redirect_stdout(out):
                cli.dispatch(self.robot, self.state, line)
        except Exception as exc:  # noqa: BLE001 — surface, don't kill the GUI
            self.ui_q.put(("log", f"!! {type(exc).__name__}: {exc}"))
        finally:
            if auto:
                self.state["auto_yes"] = False
            out.flush()
            self.ui_q.put(("busy", False))

    def _run_multi(self, payload):
        """Run several command lines as ONE gesture: an optional upfront
        confirm popup ('warn' in the payload), then every step auto-confirmed
        (like 'seq run'); stops on the first failed step."""
        tag, lines = payload["tag"], payload["lines"]
        self.ui_q.put(("echo", "  ->  ".join(lines)))
        self.ui_q.put(("busy", True))
        out = _LogWriter(self.ui_q)
        try:
            with contextlib.redirect_stdout(out):
                confirm = self.state.get("confirm") or (lambda _m: True)
                if payload.get("warn") and not confirm(payload["warn"]):
                    print(f"[{tag}] cancelled. 취소되었습니다.")
                    return
                self.state["auto_yes"] = True
                try:
                    for i, line in enumerate(lines, start=1):
                        print(f"[{tag}] --- step {i}/{len(lines)}: {line}")
                        if not cli.dispatch(self.robot, self.state, line):
                            print(f"[{tag}] step {i} FAILED — 중단합니다. "
                                  "(팔 상태를 확인하세요)")
                            return
                finally:
                    self.state["auto_yes"] = False
                print(f"[{tag}] {payload['done']}")
        except Exception as exc:  # noqa: BLE001
            self.ui_q.put(("log", f"!! {type(exc).__name__}: {exc}"))
        finally:
            out.flush()
            self.ui_q.put(("busy", False))

    def _run_poll(self):
        try:
            snap = {
                "modes": cc.read_modes(self.robot),
                "ok": self.robot.is_ok(),
                "fps": self.robot.get_fps(),
            }
            pose, psrc = sp.read_pose(self.robot, timeout=0.3)
            ja, jsrc = sj.read_joints(self.robot, timeout=0.3)
            snap["pose"], snap["pose_src"] = pose, psrc
            snap["joints"] = list(ja.msg) if ja is not None else None
            snap["joints_src"] = jsrc
            self.ui_q.put(("state", snap))
        except Exception as exc:  # noqa: BLE001
            self.ui_q.put(("log", f"!! poll: {type(exc).__name__}: {exc}"))


class NeroGui:
    def __init__(self, root, robot, state, can_name):
        self.root = root
        self.state = state
        self.ui_q = queue.Queue()
        self.worker = RobotWorker(robot, state, self.ui_q)
        self.last_pose = None       # native m/rad, from the poll
        self.last_joints = None     # rad, from the poll
        self.last_role = None       # linkage role from the poll
        self._buttons = []          # action buttons disabled while busy
        self._move_buttons = []     # also disabled while in LEADER mode
        self._busy = False

        # Cores call these from the WORKER thread; hand the question to the
        # UI thread and block until the popup answers.
        state["confirm"] = self._confirm_from_worker
        state["interact"] = self._interact_from_worker

        root.title(f"Nero Control — {can_name}")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self.worker.start()
        self.worker.submit("poll")
        root.after(PUMP_MS, self._pump)
        root.after(POLL_MS, self._schedule_poll)

    # ---------------------------------------------------------------- build
    def _btn(self, parent, text, command, **kw):
        b = ttk.Button(parent, text=text, command=command, **kw)
        self._buttons.append(b)
        return b

    def _danger_btn(self, parent, text, command, disable_in_leader=True):
        """A red button for anything that can move the arm/gripper.

        Motion commands are refused by the cores in LEADER mode anyway, so
        those buttons also gray out while the arm is in LEADER; mode-switch /
        leader-capture buttons pass disable_in_leader=False (they are how you
        LEAVE leader)."""
        b = tk.Button(parent, text=text, command=command,
                      bg=DANGER_BG, fg="white",
                      activebackground=DANGER_ACTIVE,
                      activeforeground="white")
        self._buttons.append(b)
        if disable_in_leader:
            self._move_buttons.append(b)
        return b

    def _build(self):
        pad = {"padx": 4, "pady": 3}
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)
        self.status_var = tk.StringVar(value="상태 읽는 중…")
        ttk.Label(top, textvariable=self.status_var).pack(side="left")
        self.busy_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.busy_var, foreground="#b00")\
            .pack(side="right")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, **pad)

        # Left: live pose + joints -----------------------------------------
        live = ttk.LabelFrame(body, text="현재 상태 (자동 갱신)")
        live.pack(side="left", fill="y", padx=(0, 6))
        self.pose_vars = [tk.StringVar(value="—") for _ in range(6)]
        for i, name in enumerate(("X", "Y", "Z", "Roll", "Pitch", "Yaw")):
            row = ttk.Frame(live)
            row.pack(fill="x", padx=6)
            ttk.Label(row, text=name, width=6).pack(side="left")
            ttk.Label(row, textvariable=self.pose_vars[i], width=13,
                      anchor="e", font=("TkFixedFont", 9)).pack(side="right")
        ttk.Separator(live).pack(fill="x", pady=4)
        self.joint_vars = [tk.StringVar(value="—") for _ in range(7)]
        for i in range(7):
            row = ttk.Frame(live)
            row.pack(fill="x", padx=6)
            ttk.Label(row, text=f"J{i + 1}", width=6).pack(side="left")
            ttk.Label(row, textvariable=self.joint_vars[i], width=13,
                      anchor="e", font=("TkFixedFont", 9)).pack(side="right")
        ttk.Separator(live).pack(fill="x", pady=4)
        self._btn(live, "지금 갱신", lambda: self.worker.submit("poll"))\
            .pack(padx=6, pady=(0, 6), fill="x")

        # Right: tabs ------------------------------------------------------
        nb = ttk.Notebook(body)
        nb.pack(side="left", fill="both", expand=True)
        nb.add(self._tab_move(nb), text=" 이동 ")
        nb.add(self._tab_waypoints(nb), text=" 웨이포인트 ")
        nb.add(self._tab_gripper(nb), text=" 그리퍼 ")
        nb.add(self._tab_sequences(nb), text=" 시퀀스 ")

        # Mode + speed row --------------------------------------------------
        modes = ttk.Frame(self.root)
        modes.pack(fill="x", **pad)
        self.mode_lights = {}
        for text, line, role in (("Follower", "mode follower", "FOLLOWER"),
                                 ("Leader", "mode leader", "LEADER"),
                                 ("티칭 (teach)", "teach", None),
                                 ("CAN 복귀", "canmode", None)):
            box = ttk.Frame(modes)
            box.pack(side="left", padx=2)
            cv = tk.Canvas(box, width=16, height=14, highlightthickness=0,
                           bg=self.root.cget("bg"))
            cv.pack()
            if role:   # round lamp: green when this is the current linkage mode
                oid = cv.create_oval(3, 2, 13, 12, fill=LIGHT_OFF,
                                     outline="#333")
                self.mode_lights[role] = (cv, oid)
            # follower/leader switch without a confirm popup (auto=True);
            # these buttons must stay usable in LEADER — they leave it.
            self._danger_btn(box, text,
                             lambda l=line, a=bool(role): self.run_cmd(l, a),
                             disable_in_leader=False).pack()
        # Default ON: moving commands pop a confirm. Off = run immediately.
        self.confirm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(modes, text="이동 경고문",
                        variable=self.confirm_var).pack(side="left", padx=8)
        ttk.Label(modes, text="   속도 %").pack(side="left")
        self.speed_var = tk.StringVar(value=str(self.state["speed"]))
        ttk.Spinbox(modes, from_=1, to=100, width=5,
                    textvariable=self.speed_var).pack(side="left")
        self._btn(modes, "적용",
                  lambda: self.run_cmd(f"speed {self.speed_var.get().strip()}"))\
            .pack(side="left", padx=2)

        # Log + command line ------------------------------------------------
        self.log = ScrolledText(self.root, height=12, state="disabled",
                                font=("TkFixedFont", 9))
        self.log.pack(fill="both", expand=True, **pad)
        cmdrow = ttk.Frame(self.root)
        cmdrow.pack(fill="x", **pad)
        ttk.Label(cmdrow, text="nero>").pack(side="left")
        self.cmd_var = tk.StringVar()
        entry = ttk.Entry(cmdrow, textvariable=self.cmd_var)
        entry.pack(side="left", fill="x", expand=True, padx=4)
        entry.bind("<Return>", lambda _e: self._run_typed())
        ttk.Button(cmdrow, text="실행", command=self._run_typed)\
            .pack(side="left")

    def _entries(self, parent, labels, width=7, defaults=None):
        """A row of labeled Entry widgets; returns their StringVars."""
        svars = []
        for i, lab in enumerate(labels):
            ttk.Label(parent, text=lab).grid(row=0, column=2 * i, sticky="e")
            v = tk.StringVar(value=(defaults[i] if defaults else ""))
            ttk.Entry(parent, textvariable=v, width=width)\
                .grid(row=0, column=2 * i + 1, padx=(0, 4))
            svars.append(v)
        return svars

    def _tab_move(self, nb):
        tab = ttk.Frame(nb)
        pl = ("X mm", "Y mm", "Z mm", "R°", "P°", "Y°")
        jl = tuple(f"J{i + 1}°" for i in range(7))

        f1 = ttk.LabelFrame(tab, text="movep — 절대 위치 (mm / deg)")
        f1.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(f1)
        row.pack(fill="x", padx=4, pady=2)
        self.movep_vars = self._entries(row, pl)
        btns = ttk.Frame(f1)
        btns.pack(fill="x", padx=4, pady=(0, 4))
        self._btn(btns, "현재 자세 채우기", self._fill_pose).pack(side="left")
        self._danger_btn(btns, "이동", self._do_movep).pack(side="left", padx=4)

        f2 = ttk.LabelFrame(tab, text="movep rel — 상대 이동 (비우면/0이면 유지)")
        f2.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(f2)
        row.pack(fill="x", padx=4, pady=2)
        self.movep_rel_vars = self._entries(row, pl, defaults=["0"] * 6)
        b = ttk.Frame(f2)
        b.pack(fill="x", padx=4, pady=(0, 4))
        self._danger_btn(b, "상대 이동", self._do_movep_rel).pack(side="left")

        f3 = ttk.LabelFrame(tab, text="movej — 절대 관절 (deg)")
        f3.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(f3)
        row.pack(fill="x", padx=4, pady=2)
        self.movej_vars = self._entries(row, jl, width=6)
        btns = ttk.Frame(f3)
        btns.pack(fill="x", padx=4, pady=(0, 4))
        self._btn(btns, "현재 관절 채우기", self._fill_joints).pack(side="left")
        self._danger_btn(btns, "이동", self._do_movej).pack(side="left", padx=4)

        f4 = ttk.LabelFrame(tab, text="movej rel — 상대 관절 (deg, 0 = 유지)")
        f4.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(f4)
        row.pack(fill="x", padx=4, pady=2)
        self.movej_rel_vars = self._entries(row, jl, width=6,
                                            defaults=["0"] * 7)
        b = ttk.Frame(f4)
        b.pack(fill="x", padx=4, pady=(0, 4))
        self._danger_btn(b, "상대 이동", self._do_movej_rel).pack(side="left")
        return tab

    def _tab_waypoints(self, nb):
        tab = ttk.Frame(nb)
        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        self.wp_list = tk.Listbox(left, height=10, exportselection=False)
        self.wp_list.pack(fill="both", expand=True)
        right = ttk.Frame(tab)
        right.pack(side="left", fill="y", padx=6, pady=4)

        self._btn(right, "새로고침", self.refresh_lists).pack(fill="x")
        ttk.Separator(right).pack(fill="x", pady=4)
        ttk.Label(right, text="이동 방식").pack(anchor="w")
        self.goto_method = ttk.Combobox(
            right, values=("auto", "joints", "cartesian"), width=10,
            state="readonly")
        self.goto_method.set("auto")
        self.goto_method.pack(fill="x")
        self._danger_btn(right, "선택 지점으로 이동", self._do_goto)\
            .pack(fill="x", pady=2)
        ttk.Separator(right).pack(fill="x", pady=4)
        ttk.Label(right, text="현재 자세 저장").pack(anchor="w")
        self.wp_name = tk.StringVar()
        self.wp_desc = tk.StringVar()
        ttk.Entry(right, textvariable=self.wp_name, width=14).pack(fill="x")
        ttk.Entry(right, textvariable=self.wp_desc, width=14).pack(fill="x",
                                                                   pady=2)
        self._btn(right, "저장 (save)", self._do_save).pack(fill="x")
        ttk.Separator(right).pack(fill="x", pady=4)
        # Only usable IN leader mode (it captures a hand-guided posture).
        self.capture_btn = self._danger_btn(
            right, "Leader 캡처\nfollower→저장→leader",
            self._do_leader_capture, disable_in_leader=False)
        self.capture_btn.pack(fill="x")
        return tab

    def _tab_gripper(self, nb):
        tab = ttk.Frame(nb)
        r1 = ttk.Frame(tab)
        r1.pack(fill="x", padx=6, pady=4)
        self._btn(r1, "연결 확인", lambda: self.run_cmd("gripper"))\
            .pack(side="left", padx=2)
        self._btn(r1, "정보", lambda: self.run_cmd("gripper info"))\
            .pack(side="left", padx=2)
        self._btn(r1, "리셋", lambda: self.run_cmd("gripper reset"))\
            .pack(side="left", padx=2)

        r2 = ttk.LabelFrame(tab, text="동작 (모든 모터 enable — 팔은 자세 유지)")
        r2.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(r2)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="힘 N").pack(side="left")
        self.grip_force = tk.StringVar(value="1")
        ttk.Entry(row, textvariable=self.grip_force, width=5)\
            .pack(side="left", padx=(2, 8))
        self._danger_btn(row, "열기", lambda: self._grip("open")).pack(side="left",
                                                                padx=2)
        self._danger_btn(row, "닫기", lambda: self._grip("close")).pack(side="left",
                                                                 padx=2)
        row2 = ttk.Frame(r2)
        row2.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(row2, text="목표 (mm 폭 / deg 각)").pack(side="left")
        self.grip_to = tk.StringVar()
        ttk.Entry(row2, textvariable=self.grip_to, width=7)\
            .pack(side="left", padx=2)
        self._danger_btn(row2, "이동", self._grip_to).pack(side="left", padx=2)

        r3 = ttk.LabelFrame(tab, text="영점 캘리브레이션 (조를 손으로 닫고 진행)")
        r3.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(r3)
        row.pack(fill="x", padx=4, pady=4)
        ttk.Label(row, text="스트로크 mm").pack(side="left")
        self.calib_range = ttk.Combobox(row, values=("(유지)", "70", "100"),
                                        width=6, state="readonly")
        self.calib_range.set("(유지)")
        self.calib_range.pack(side="left", padx=4)
        self._danger_btn(row, "캘리브레이션", self._do_calib).pack(side="left",
                                                            padx=2)
        return tab

    def _tab_sequences(self, nb):
        tab = ttk.Frame(nb)
        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        self.seq_list = tk.Listbox(left, height=10, exportselection=False)
        self.seq_list.pack(fill="both", expand=True)
        right = ttk.Frame(tab)
        right.pack(side="left", fill="y", padx=6, pady=4)

        self._btn(right, "새로고침", self.refresh_lists).pack(fill="x")
        self._btn(right, "스텝 보기",
                  lambda: self._seq_sel("seq show")).pack(fill="x", pady=2)
        self._danger_btn(right, "실행 (seq run)",
                         lambda: self._seq_sel("seq run")).pack(fill="x")
        ttk.Separator(right).pack(fill="x", pady=4)
        self.rec_var = tk.StringVar(value="녹화 안 함")
        ttk.Label(right, textvariable=self.rec_var).pack(anchor="w")
        self.rec_name = tk.StringVar()
        ttk.Entry(right, textvariable=self.rec_name, width=14).pack(fill="x")
        self._btn(right, "녹화 시작", self._rec_start).pack(fill="x", pady=2)
        self._btn(right, "녹화 저장", lambda: self.run_cmd("seq stop"))\
            .pack(fill="x")
        self._btn(right, "녹화 취소", lambda: self.run_cmd("seq cancel"))\
            .pack(fill="x", pady=2)
        return tab

    # ------------------------------------------------------------- commands
    def run_cmd(self, line, auto=False):
        line = line.strip()
        if not line:
            return
        if not self.confirm_var.get():   # warnings toggled off -> no popups
            auto = True
        self.worker.submit("cmd", (line, auto))

    def _run_typed(self):
        line = self.cmd_var.get().strip()
        if not line:
            return
        if line.split()[0].lower() in ("quit", "exit", "q"):
            self._on_close()
            return
        self.cmd_var.set("")
        self.run_cmd(line)

    def _vals(self, svars):
        return [v.get().strip() for v in svars]

    def _do_movep(self):
        self.run_cmd("movep " + " ".join(self._vals(self.movep_vars)))

    def _do_movep_rel(self):
        vals = [v or "0" for v in self._vals(self.movep_rel_vars)]
        if vals[3:] == ["0", "0", "0"]:
            vals = vals[:3]           # keep current orientation
        self.run_cmd("movep rel " + " ".join(vals))

    def _do_movej(self):
        self.run_cmd("movej " + " ".join(self._vals(self.movej_vars)))

    def _do_movej_rel(self):
        vals = [v or "0" for v in self._vals(self.movej_rel_vars)]
        self.run_cmd("movej rel " + " ".join(vals))

    def _fill_pose(self):
        if self.last_pose is None:
            self.log_line("!! 아직 자세를 읽지 못했습니다.")
            return
        x, y, z, r, p, yw = self.last_pose
        for var, val in zip(self.movep_vars,
                            (x * 1000, y * 1000, z * 1000,
                             math.degrees(r), math.degrees(p),
                             math.degrees(yw))):
            var.set(f"{val:.1f}")

    def _fill_joints(self):
        if self.last_joints is None:
            self.log_line("!! 아직 관절값을 읽지 못했습니다.")
            return
        for var, rad in zip(self.movej_vars, self.last_joints):
            var.set(f"{math.degrees(rad):.1f}")

    def _sel(self, listbox):
        sel = listbox.curselection()
        if not sel:
            self.log_line("!! 목록에서 항목을 먼저 선택하세요.")
            return None
        return listbox.get(sel[0]).split()[0]

    def _do_goto(self):
        name = self._sel(self.wp_list)
        if name:
            method = self.goto_method.get()
            self.run_cmd(f"goto {name}"
                         + (f" {method}" if method != "auto" else ""))

    def _dup_name(self, name):
        """True (and a popup) when a waypoint with this name already exists —
        the GUI never overwrites silently."""
        if name in wp.list_waypoints():
            messagebox.showwarning(
                "Nero — 이름 중복",
                f"웨이포인트 '{name}'이(가) 이미 있습니다.\n"
                "덮어쓰지 않았습니다 — 다른 이름을 입력하세요.",
                parent=self.root)
            return True
        return False

    def _do_save(self):
        name = self.wp_name.get().strip()
        if not name:
            self.log_line("!! 웨이포인트 이름을 입력하세요.")
            return
        if self._dup_name(name):
            return
        desc = self.wp_desc.get().strip()
        self.run_cmd(f"save {name}" + (f" {desc}" if desc else ""))

    def _do_leader_capture(self):
        """From LEADER: hold the arm (follower), save the posture, release
        back into leader — one button, one confirm."""
        name = self.wp_name.get().strip()
        if not name:
            self.log_line("!! 웨이포인트 이름을 입력하세요.")
            return
        if self.last_role != "LEADER":
            self.log_line("!! 현재 LEADER 모드가 아닙니다 — 이 버튼은 leader에서 "
                          "손으로 잡은 자세를 저장할 때 씁니다. "
                          "(일반 저장은 '저장 (save)')")
            return
        if self._dup_name(name):
            return
        desc = self.wp_desc.get().strip()
        save = f"save {name}" + (f" {desc}" if desc else "")
        self.worker.submit("multi", {
            "tag": "캡처",
            "lines": ["mode follower", save, "mode leader"],
            "done": f"완료 — '{name}' 저장됨. 팔은 다시 LEADER(드래그) "
                    "상태입니다."})

    def _grip(self, spec):
        force = self.grip_force.get().strip()
        self.run_cmd(f"gripper {spec}" + (f" {force}" if force else ""))

    def _grip_to(self):
        val = self.grip_to.get().strip()
        if not val:
            self.log_line("!! 그리퍼 목표값을 입력하세요.")
            return
        force = self.grip_force.get().strip()
        self.run_cmd(f"gripper to {val}" + (f" {force}" if force else ""))

    def _do_calib(self):
        rng = self.calib_range.get()
        self.run_cmd("gripper calib" + (f" {rng}" if rng in ("70", "100")
                                        else ""))

    def _seq_sel(self, prefix):
        name = self._sel(self.seq_list)
        if name:
            self.run_cmd(f"{prefix} {name}")

    def _rec_start(self):
        name = self.rec_name.get().strip()
        if not name:
            self.log_line("!! 시퀀스 이름을 입력하세요.")
            return
        self.run_cmd(f"seq record {name}")

    # ------------------------------------------------- worker-side callbacks
    def _confirm_from_worker(self, message):
        reply = queue.Queue()
        self.ui_q.put(("ask", message, reply))
        return reply.get()

    def _interact_from_worker(self, message):
        reply = queue.Queue()
        self.ui_q.put(("pause", message, reply))
        return reply.get()

    # ------------------------------------------------------------- UI pump
    def log_line(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _pump(self):
        try:
            while True:
                item = self.ui_q.get_nowait()
                kind = item[0]
                if kind == "log":
                    self.log_line(item[1])
                elif kind == "echo":
                    self.log_line(f"nero> {item[1]}")
                elif kind == "busy":
                    self._set_busy(item[1])
                elif kind == "state":
                    self._show_state(item[1])
                elif kind == "ask":
                    item[2].put(messagebox.askyesno(
                        "Nero — 진행할까요?", item[1], parent=self.root))
                elif kind == "pause":
                    item[2].put(messagebox.askokcancel(
                        "Nero — 준비되면 확인", item[1], parent=self.root))
        except queue.Empty:
            pass
        finally:
            self.root.after(PUMP_MS, self._pump)

    def _set_busy(self, busy):
        self._busy = busy
        self.busy_var.set("실행 중…" if busy else "")
        self._update_buttons()
        if not busy:
            self.refresh_lists()
            self._show_rec()
            self.speed_var.set(str(self.state["speed"]))
            self.worker.submit("poll")   # snappy lamp/button refresh

    def _update_buttons(self):
        state = "disabled" if self._busy else "normal"
        for b in self._buttons:
            b.configure(state=state)
        if not self._busy and self.last_role == "LEADER":
            # The cores refuse motion in LEADER anyway; gray the buttons out
            # so that is visible before clicking.
            for b in self._move_buttons:
                b.configure(state="disabled")
        if not self._busy and self.last_role != "LEADER":
            # Leader capture only makes sense while hand-guiding in LEADER.
            self.capture_btn.configure(state="disabled")

    def _show_rec(self):
        cap = self.state.get("seq_rec")
        if cap is None:
            self.rec_var.set("녹화 안 함")
        else:
            self.rec_var.set(f"녹화 중: {cap['name']} "
                             f"({len(cap['steps'])} 스텝)")

    def _show_state(self, snap):
        m = snap["modes"]
        role = m.get("linkage_role")
        self.last_role = role
        self._update_buttons()
        for name, (cv, oid) in self.mode_lights.items():
            cv.itemconfig(oid, fill=LIGHT_ON if role == name else LIGHT_OFF)
        self.status_var.set(
            f"linkage: {m.get('linkage_role') or '?'}   "
            f"ctrl: {m.get('ctrl_mode') or '?'}   "
            f"comm: {'OK' if snap['ok'] else 'FAIL'}   fps: {snap['fps']}   "
            f"속도: {self.state['speed']}%")
        self.last_pose = snap["pose"]
        if snap["pose"] is None:
            for v in self.pose_vars:
                v.set("—")
        else:
            x, y, z, r, p, yw = snap["pose"]
            for var, txt in zip(self.pose_vars, (
                    f"{x * 1000:+.1f} mm", f"{y * 1000:+.1f} mm",
                    f"{z * 1000:+.1f} mm",
                    f"{math.degrees(r):+.2f}°", f"{math.degrees(p):+.2f}°",
                    f"{math.degrees(yw):+.2f}°")):
                var.set(txt)
        self.last_joints = snap["joints"]
        if snap["joints"] is None:
            for v in self.joint_vars:
                v.set("—")
        else:
            for var, rad in zip(self.joint_vars, snap["joints"]):
                var.set(f"{math.degrees(rad):+.2f}°")

    def _schedule_poll(self):
        if self.worker.idle():
            self.worker.submit("poll")
        self.root.after(POLL_MS, self._schedule_poll)

    def refresh_lists(self):
        self.wp_list.delete(0, "end")
        for name in wp.list_waypoints():
            try:
                self.wp_list.insert("end",
                                    wp.format_waypoint(wp.load_waypoint(name)))
            except Exception:  # noqa: BLE001 — unreadable file: show name only
                self.wp_list.insert("end", name)
        self.seq_list.delete(0, "end")
        for name in sq.list_sequences():
            try:
                self.seq_list.insert(
                    "end", sq.format_sequence(sq.load_sequence(name)))
            except Exception:  # noqa: BLE001
                self.seq_list.insert("end", name)

    def _on_close(self):
        if self.worker.busy and not messagebox.askyesno(
                "Nero — 종료", "명령이 아직 실행 중입니다. 그래도 종료할까요?\n"
                "(모터는 그대로 남고 팔은 현재 자세를 유지합니다.)",
                parent=self.root):
            return
        if self.state.get("seq_rec") is not None:
            self.log_line(f"[seq] 저장 안 된 녹화 '{self.state['seq_rec']['name']}' "
                          "폐기됨.")
        self.worker.submit("quit")
        try:
            self.worker.robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self.root.destroy()


def _parse_args():
    p = argparse.ArgumentParser(
        description="Tkinter GUI for the Nero arm (same cores as nero_cli).")
    p.add_argument("--can", default="can0", help="CAN channel (real arm: can0)")
    p.add_argument("--fw", default="v112",
                   choices=["auto", "default", "v111", "v112", "v120"],
                   help="firmware/parser of this arm (default: v112)")
    p.add_argument("--speed", type=int, default=20,
                   help="initial move speed percent (default: 20)")
    return p.parse_args()


def main():
    args = _parse_args()
    try:
        robot, _ = cc.connect(args.can, args.fw)
    except Exception as exc:  # noqa: BLE001
        print(f"[nero_gui] cannot connect on '{args.can}': "
              f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(f"[nero_gui] connected on {args.can}.")
    state = {"speed": max(1, min(args.speed, 100)), "timeout": 15.0,
             "effector": None, "auto_yes": False, "seq_rec": None,
             "confirm": None, "interact": None}
    root = tk.Tk()
    gui = NeroGui(root, robot, state, args.can)
    gui.refresh_lists()
    gui.log_line(f"[nero_gui] connected on {args.can}. 버튼 명령은 CLI와 같은 "
                 "코어를 타며, 움직임은 항상 팝업 확인을 거칩니다.")
    try:
        root.mainloop()
    finally:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
