#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The 경로 tab of jog_gui: build, plan, and run multi-step paths.

A path is an ordered list of steps between saved waypoints — move, gripper,
wait, signal — stored by nero_ctrl/paths.py. Planning solves the whole chain
ONCE and stores the joint waypoints; running replays them with no IK at all.

Why the chain is solved in one pass, not per segment: move step k starts from
the joints step k-1 actually ended on. That is not the same as the target
waypoint's own saved joints — the two disagree by more than 1 deg in 90% of
segments (median 27 deg), because one TCP pose is reachable in several arm
shapes. Planning each segment independently and stitching them would therefore
put a joint jump of tens of degrees at every junction.

Running is sequential on jog_gui's worker thread:
  move     retime the stored joints under the live speed caps, ramp in if the
           arm sits slightly off the step's start, then execute
  gripper  open / close through gripper_controller, no dialog (a program that
           asked before every grip would not be a program)
  wait     a plain pause
  signal   block until the operator presses 계속. The schema reserves a ROS
           topic source for when one exists; the editor does not offer it yet.

정지 stops between steps AND cancels a move already running on the controller.
It and 계속 are deliberately NOT registered with jog_gui's button lists: those
are all disabled while the worker is busy, which is exactly when a stop button
has to work.
"""
from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

import rclpy
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory

import jog_path as jp
import move_level as ml
import paths as pp
import waypoints as wp

# Poll granularity while a step is blocking, so 정지 stays responsive.
TICK_S = 0.05
DANGER_BG = "#c62828"
DANGER_ACTIVE = "#8e0000"
GO_BG = "#2e7d32"
GO_ACTIVE = "#1b5e20"
OFF_BG = "#9e9e9e"          # a disabled 정지 that still looks armed is a lie


def lead_in(q_now, q_start, qdot_max, min_T=0.3):
    """Short joint ramp from where the arm is onto a step's stored first point.

    Without it the trajectory's first point is commanded at t=0, so any gap
    between the live posture and the stored start becomes a step the controller
    tries to take in one segment — and the controller has no goal tolerance
    configured, so nothing would catch it. Returns (q, t), ending at q_start.
    """
    q_now = np.asarray(q_now, float)
    q_start = np.asarray(q_start, float)
    dq = float(np.abs(q_start - q_now).max())
    T = max(min_T, 2.0 * dq / qdot_max)
    n = max(2, int(T * 20))
    f = np.linspace(0.0, 1.0, n)[:, None]
    return q_now[None] * (1 - f) + q_start[None] * f, np.linspace(0.0, T, n)


class PathTab:
    """Editor + runner for multi-step paths. Owns no ROS state of its own
    beyond one reusable action client; everything else comes from `gui`."""

    def __init__(self, parent, gui):
        self.gui = gui
        self.steps = []              # the step list currently being edited
        self._cache = {}             # name -> saved record
        self._stop = threading.Event()
        self._go = threading.Event()  # set by 계속 during a signal step
        self._running = False
        self._waiting_signal = False
        self._exec_client = None
        self._build(parent)

    # ---------------------------------------------------------------- build
    def _build(self, parent):
        pad = dict(padx=6, pady=3)
        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", anchor="n")
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, anchor="n")

        # --- saved paths
        lf = ttk.LabelFrame(left, text="저장된 경로 (src/data/paths/*.json)")
        lf.pack(fill="both", **pad)
        row = ttk.Frame(lf)
        row.pack(fill="x", padx=6, pady=(3, 1))
        ttk.Label(row, text="이름").pack(side="left")
        self.name_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.name_var, width=16).pack(
            side="left", padx=(2, 6))
        self.gui._btn(row, "저장", self._do_save).pack(side="left", padx=2)

        lrow = ttk.Frame(lf)
        lrow.pack(fill="both", padx=6, pady=2)
        self.list = tk.Listbox(lrow, height=12, width=44, exportselection=False,
                               font=("TkFixedFont", 9), activestyle="dotbox")
        self.list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lrow, orient="vertical", command=self.list.yview)
        sb.pack(side="left", fill="y")
        self.list.configure(yscrollcommand=sb.set)
        self.list.bind("<Double-Button-1>", lambda _e: self._do_load())

        row = ttk.Frame(lf)
        row.pack(fill="x", padx=6, pady=(0, 4))
        self.gui._btn(row, "불러오기", self._do_load).pack(side="left", padx=2)
        self.gui._btn(row, "새로고침", self.refresh).pack(side="right", padx=2)
        self.gui._btn(row, "삭제", self._do_delete).pack(side="right", padx=2)

        # --- start posture
        sf = ttk.LabelFrame(right, text="편집 중인 경로")
        sf.pack(fill="both", expand=True, **pad)
        row = ttk.Frame(sf)
        row.pack(fill="x", padx=6, pady=(3, 1))
        ttk.Label(row, text="시작 자세").pack(side="left")
        self.start_var = tk.StringVar()
        self.start_box = ttk.Combobox(row, textvariable=self.start_var,
                                      width=12, state="readonly")
        self.start_box.pack(side="left", padx=(2, 10))
        ttk.Label(row, text="경로는 이 자세에서 시작합니다 (관절각으로 고정)",
                  foreground="#555").pack(side="left")

        # --- step list
        lrow = ttk.Frame(sf)
        lrow.pack(fill="both", expand=True, padx=6, pady=2)
        self.step_list = tk.Listbox(lrow, height=9, width=52,
                                    exportselection=False,
                                    font=("TkFixedFont", 9), activestyle="dotbox")
        self.step_list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lrow, orient="vertical", command=self.step_list.yview)
        sb.pack(side="left", fill="y")
        self.step_list.configure(yscrollcommand=sb.set)

        row = ttk.Frame(sf)
        row.pack(fill="x", padx=6, pady=1)
        self.gui._btn(row, "위로", lambda: self._move_step(-1)).pack(side="left",
                                                                   padx=2)
        self.gui._btn(row, "아래로", lambda: self._move_step(+1)).pack(side="left",
                                                                    padx=2)
        self.gui._btn(row, "스텝 삭제", self._del_step).pack(side="left", padx=2)
        self.gui._btn(row, "전체 비우기", self._clear_steps).pack(side="left",
                                                              padx=(10, 2))

        # --- step builders
        af = ttk.LabelFrame(sf, text="스텝 추가")
        af.pack(fill="x", padx=6, pady=(4, 2))
        row = ttk.Frame(af)
        row.pack(fill="x", padx=6, pady=2)
        ttk.Label(row, text="이동 →").pack(side="left")
        self.move_to = tk.StringVar()
        self.move_box = ttk.Combobox(row, textvariable=self.move_to, width=12,
                                     state="readonly")
        self.move_box.pack(side="left", padx=(2, 4))
        self.gui._btn(row, "추가", self._add_move).pack(side="left", padx=2)
        ttk.Label(row, text="   그리퍼").pack(side="left")
        self.grip_act = tk.StringVar(value="닫기")
        ttk.Combobox(row, textvariable=self.grip_act, width=6, state="readonly",
                     values=("열기", "닫기")).pack(side="left", padx=(2, 4))
        self.gui._btn(row, "추가", self._add_gripper).pack(side="left", padx=2)

        row = ttk.Frame(af)
        row.pack(fill="x", padx=6, pady=(0, 3))
        ttk.Label(row, text="대기").pack(side="left")
        self.wait_s = tk.StringVar(value="1.0")
        ttk.Entry(row, textvariable=self.wait_s, width=6).pack(side="left",
                                                               padx=2)
        ttk.Label(row, text="초").pack(side="left")
        self.gui._btn(row, "추가", self._add_wait).pack(side="left", padx=(4, 2))
        ttk.Label(row, text="   신호: 계속 버튼 대기, 제한").pack(side="left")
        self.sig_timeout = tk.StringVar(value="0")
        ttk.Entry(row, textvariable=self.sig_timeout, width=6).pack(side="left",
                                                                    padx=2)
        ttk.Label(row, text="초 (0 = 무제한)").pack(side="left")
        self.gui._btn(row, "추가", self._add_signal).pack(side="left", padx=(4, 2))

        # --- plan + run
        row = ttk.Frame(sf)
        row.pack(fill="x", padx=6, pady=(4, 2))
        self.gui._btn(row, "계획 + 저장", self._do_plan).pack(side="left", padx=2)
        self.gui._btn(row, "시작 자세로 이동", self._do_to_start, danger=True,
                      jog=True).pack(side="left", padx=(10, 2))
        self.gui._btn(row, "실행", self._do_run, danger=True, jog=True
                      ).pack(side="left", padx=2)
        # 정지 / 계속 stay out of the gui's button lists on purpose: those are
        # disabled while the worker is busy, which is when these two matter.
        self.stop_btn = tk.Button(row, text="정지", command=self._do_stop,
                                  bg=OFF_BG, fg="white",
                                  activebackground=DANGER_ACTIVE,
                                  activeforeground="white", state="disabled")
        self.stop_btn.pack(side="left", padx=(10, 2))
        self.go_btn = tk.Button(row, text="계속", command=self._do_go,
                                bg=OFF_BG, fg="white", activebackground=GO_ACTIVE,
                                activeforeground="white", state="disabled")
        self.go_btn.pack(side="left", padx=2)

        self.status = tk.StringVar(value="대기 중")
        ttk.Label(sf, textvariable=self.status, font=("TkFixedFont", 10),
                  foreground="#333").pack(anchor="w", padx=6, pady=(1, 2))
        ttk.Label(sf, text="계획은 IK를 한 번만 풉니다 · 실행은 저장된 관절 궤적을 "
                           "그대로 씁니다 · 정지는 실행 중인 이동도 취소합니다",
                  foreground="#555", wraplength=560, justify="left"
                  ).pack(anchor="w", padx=6, pady=(0, 3))
        self._redraw_steps()

    # -------------------------------------------------------------- helpers
    def log(self, line):
        self.gui.log_line(line)

    def set_status(self, text):
        self.gui.ui_q.put(("pathstatus", text))

    def waypoint_names(self, names):
        """Called by jog_gui whenever the waypoint list changes."""
        for box, var in ((self.start_box, self.start_var),
                         (self.move_box, self.move_to)):
            box.configure(values=names)
            if var.get() not in names:
                var.set(names[0] if names else "")

    def _selected_name(self):
        sel = self.list.curselection()
        if not sel:
            return None
        names = list(self._cache)
        return names[sel[0]] if sel[0] < len(names) else None

    def _float(self, var, label):
        try:
            return float(var.get().strip())
        except ValueError:
            raise ValueError(f"'{label}'이(가) 숫자가 아닙니다: {var.get()}") from None

    # ------------------------------------------------------- the step list
    def _redraw_steps(self):
        keep = self.step_list.curselection()
        self.step_list.delete(0, "end")
        for i, step in enumerate(self.steps):
            self.step_list.insert("end", pp.describe_step(step, i))
        if keep and keep[0] < len(self.steps):
            self.step_list.selection_set(keep[0])
        n_move = sum(1 for s in self.steps if s.get("type") == "move")
        solved = sum(1 for s in self.steps
                     if s.get("type") == "move" and s.get("joints_deg"))
        ready = "실행 가능" if n_move and solved == n_move else "계획 필요"
        self.status.set(f"스텝 {len(self.steps)}개 (이동 {n_move}, "
                        f"계획됨 {solved}) — {ready}")

    def _add(self, step):
        sel = self.step_list.curselection()
        at = sel[0] + 1 if sel else len(self.steps)
        self.steps.insert(at, step)
        self._redraw_steps()
        self.step_list.selection_clear(0, "end")
        self.step_list.selection_set(at)

    def _add_move(self):
        to = self.move_to.get()
        if not to:
            self.log("!! 이동할 자세를 고르세요")
            return
        self._add(pp.move_step(to))

    def _add_gripper(self):
        self._add(pp.gripper_step(
            "open" if self.grip_act.get() == "열기" else "close"))

    def _add_wait(self):
        try:
            self._add(pp.wait_step(self._float(self.wait_s, "대기 시간")))
        except ValueError as exc:
            self.log(f"!! {exc}")

    def _add_signal(self):
        try:
            self._add(pp.signal_step(
                "operator", timeout_s=self._float(self.sig_timeout, "제한 시간")))
        except ValueError as exc:
            self.log(f"!! {exc}")

    def _move_step(self, delta):
        sel = self.step_list.curselection()
        if not sel:
            self.log("!! 옮길 스텝을 고르세요")
            return
        i = sel[0]
        j = i + delta
        if not 0 <= j < len(self.steps):
            return
        self.steps[i], self.steps[j] = self.steps[j], self.steps[i]
        self._invalidate("스텝 순서가 바뀌어")
        self._redraw_steps()
        self.step_list.selection_clear(0, "end")
        self.step_list.selection_set(j)

    def _del_step(self):
        sel = self.step_list.curselection()
        if not sel:
            self.log("!! 삭제할 스텝을 고르세요")
            return
        removed = self.steps.pop(sel[0])
        if removed.get("type") == "move":
            self._invalidate("이동 스텝이 삭제되어")
        self._redraw_steps()

    def _clear_steps(self):
        if self.steps and not messagebox.askyesno(
                "Nero — 스텝 비우기", f"편집 중인 스텝 {len(self.steps)}개를 "
                "모두 지울까요?", parent=self.gui.root):
            return
        self.steps = []
        self._redraw_steps()

    def _invalidate(self, why):
        """Drop solved trajectories whose chain no longer holds."""
        dropped = 0
        for step in self.steps:
            if step.get("type") == "move" and step.pop("joints_deg", None):
                step.pop("arc_len_m", None)
                step.pop("angle_deg", None)
                step.pop("metrics", None)
                dropped += 1
        if dropped:
            self.log(f"{why} 계획이 무효가 되었습니다 — 이동 {dropped}개를 다시 "
                     "계획하세요 (체인은 앞 스텝이 끝난 관절각에서 이어집니다)")

    # ------------------------------------------------------- the file list
    def refresh(self):
        keep = self._selected_name()
        self._cache = {}
        bad = []
        for name in pp.list_paths():
            try:
                self._cache[name] = pp.load_path(name)
            except (OSError, ValueError, KeyError) as exc:
                bad.append(f"{name} ({exc})")
        self.list.delete(0, "end")
        for name, rec in self._cache.items():
            self.list.insert("end", pp.format_path(rec))
        if bad:
            self.log("!! 읽지 못한 경로: " + "; ".join(bad))
        names = list(self._cache)
        if keep in names:
            self.list.selection_set(names.index(keep))

    def _do_load(self):
        name = self._selected_name()
        if name is None:
            self.log("!! 불러올 경로를 고르세요")
            return
        rec = self._cache[name]
        self.name_var.set(name)
        self.start_var.set(rec["start"].get("waypoint", ""))
        self.steps = [dict(s) for s in rec["steps"]]
        self._redraw_steps()
        self.log(f"경로 '{name}' 불러옴: {pp.format_path(rec)}")

    def _do_save(self):
        rec = self._build_record()
        if rec is None:
            return
        try:
            f_path, existed = pp.save_path(rec, overwrite=True)
        except (OSError, ValueError) as exc:
            self.log(f"!! 저장 실패: {exc}")
            return
        self.log(f"경로 '{rec['name']}' {'덮어씀' if existed else '저장'} → "
                 f"{f_path}")
        self.refresh()

    def _build_record(self, *, quiet=False):
        """Validate the editor state and build a record, or None after logging."""
        name = self.name_var.get().strip()
        err = pp.check_name(name)
        if err:
            self.log(f"!! 이름 오류: {err}")
            return None
        err = pp.check_steps(self.steps)
        if err:
            self.log(f"!! {err}")
            return None
        start = self.start_var.get()
        rec_a = self.gui._wp_cache.get(start)
        if rec_a is None or not rec_a.get("joints_deg"):
            self.log(f"!! 시작 자세 '{start}'에 관절각이 없습니다")
            return None
        if not quiet and os.path.exists(pp.path_file(name)) \
                and not messagebox.askyesno(
                    "Nero — 덮어쓸까요?",
                    f"'{name}' 경로가 이미 있습니다.\n\n{pp.path_file(name)}\n\n"
                    "덮어쓸까요?", parent=self.gui.root):
            self.log(f"'{name}' 저장 취소됨")
            return None
        a = self.gui.args
        return pp.make_path(
            name, {"waypoint": start, "joints_deg": list(rec_a["joints_deg"])},
            [dict(s) for s in self.steps],
            planner={"step_m": a.step, "ang_step_deg": a.ang_step,
                     "frame": "base_link", "rpy_convention": "Rz*Ry*Rx",
                     "source": f"jog_gui {'real' if a.real else 'sim'}",
                     "tcp_offset": {"xyz_m": [float(v) for v in a.tcp_xyz],
                                    "rpy_rad": [float(v) for v in a.tcp_rpy]}})

    def _do_delete(self):
        name = self._selected_name()
        if name is None:
            self.log("!! 삭제할 경로를 고르세요")
            return
        if not messagebox.askyesno(
                "Nero — 경로 삭제",
                f"저장된 경로 '{name}'을 삭제할까요?\n\n{pp.path_file(name)}\n\n"
                "되돌릴 수 없습니다.", parent=self.gui.root):
            return
        try:
            pp.delete_path(name)
        except (OSError, ValueError) as exc:
            self.log(f"!! 삭제 실패: {exc}")
            return
        self.log(f"경로 '{name}' 삭제됨")
        self.refresh()

    # ------------------------------------------------------------ planning
    def _do_plan(self):
        """Solve the whole chain once and save it. Never moves the arm."""
        rec = self._build_record()
        if rec is None:
            return
        try:
            self.gui._sync_params()
        except ValueError as exc:
            self.log(f"!! {exc}")
            return
        a = self.gui.args
        wp_cache = dict(self.gui._wp_cache)

        def job():
            q = np.deg2rad(np.asarray(rec["start"]["joints_deg"], float))
            R_t, t_t = self.gui._tcp()
            out = []
            t0 = time.time()
            for i, step in enumerate(rec["steps"]):
                if step.get("type") != "move":
                    out.append(step)
                    continue
                target = wp_cache.get(step["to"])
                if target is None or not target.get("cartesian"):
                    print(f"!! {i + 1}번 스텝: 자세 '{step['to']}'에 직교 좌표가 "
                          "없습니다")
                    return
                pose6 = wp.to_cartesian_target(target)
                R1, p1 = jp.pose_from_rpy(pose6[:3], pose6[3:])
                plan = jp.plan_line(q, R1, p1, step=a.step,
                                    ang_step=np.deg2rad(a.ang_step),
                                    R_t=R_t, t_t=t_t)
                if not plan["ok"]:
                    print(f"!! {i + 1}번 스텝 (→ {step['to']}) 계획 실패: "
                          f"{plan['msg']}")
                    print("   앞 스텝의 끝 자세에서 출발하므로, 중간 경유 자세를 "
                          "하나 넣으면 풀리는 경우가 많습니다")
                    return
                qq = plan["q"]
                out.append(pp.move_step(
                    step["to"], qq, plan["s"], plan["ang"],
                    metrics={"points": len(qq),
                             "dist_mm": round(plan["dist"] * 1000, 3),
                             "angle_deg": round(float(np.rad2deg(plan["angle"])), 3),
                             "max_joint_step_deg": round(float(np.rad2deg(
                                 np.abs(np.diff(qq, axis=0)).max())), 3)}))
                q = qq[-1]
            rec["steps"] = out
            f_path, existed = pp.save_path(rec, overwrite=True)
            pts, dist, ang = pp.totals(rec)
            print(f"경로 '{rec['name']}' {'덮어씀' if existed else '저장'}: "
                  f"스텝 {len(out)}개, 이동 {len(pp.move_steps(rec))}구간, "
                  f"{pts} 경로점, {dist:.1f} mm / {ang:.1f}°, "
                  f"IK {time.time() - t0:.3f}초 → {f_path}")
            print("   실행할 때는 IK를 다시 풀지 않습니다")
            self.gui.ui_q.put(("pathsaved", rec["name"]))
        self.gui.worker.submit(job)

    def adopt(self, name):
        """After a plan job saved `name`, show the solved steps in the editor."""
        self.refresh()
        if name not in self._cache:
            return
        self.steps = [dict(s) for s in self._cache[name]["steps"]]
        self._redraw_steps()
        names = list(self._cache)
        self.list.selection_clear(0, "end")
        self.list.selection_set(names.index(name))

    # ------------------------------------------------------------- running
    def _do_stop(self):
        self._stop.set()
        self._go.set()               # release a signal step that is blocking
        self.log("정지 요청 — 실행 중인 이동을 취소합니다")

    def _do_go(self):
        self._go.set()

    @staticmethod
    def _paint(btn, live, bg):
        """Colour a button only while it actually does something."""
        btn.configure(state="normal" if live else "disabled",
                      bg=bg if live else OFF_BG)

    def _set_running(self, running):
        self._running = running
        self._paint(self.stop_btn, running, DANGER_BG)
        if not running:
            self._waiting_signal = False
            self._paint(self.go_btn, False, GO_BG)

    def _set_signal_wait(self, waiting):
        self._waiting_signal = waiting
        self._paint(self.go_btn, waiting, GO_BG)

    def _do_to_start(self):
        rec = self._current_for_run(need_plan=False)
        if rec is None:
            return
        self.gui._run_joint_move(
            f"경로 '{rec['name']}' 시작 자세로",
            np.deg2rad(np.asarray(rec["start"]["joints_deg"], float)))

    def _current_for_run(self, need_plan=True):
        """The record to run: the editor's steps, validated."""
        rec = self._build_record(quiet=True)
        if rec is None:
            return None
        if need_plan and not pp.planned(rec):
            self.log("!! 아직 계획되지 않은 이동 스텝이 있습니다 — '계획 + 저장'을 "
                     "먼저 누르세요")
            return None
        return rec

    def _do_run(self):
        rec = self._current_for_run()
        if rec is None:
            return
        try:
            self.gui._sync_params()
        except ValueError as exc:
            self.log(f"!! {exc}")
            return
        a = self.gui.args
        preview_only = bool(self.gui.preview_var.get())
        if not preview_only and not self.gui.arm_var.get():
            self.log("!! 조그가 비활성화 상태입니다 — '활성화' 또는 '프리뷰만'")
            return
        if self._running:
            self.log("!! 이미 실행 중입니다")
            return
        if not preview_only and not messagebox.askyesno(
                "Nero — 경로 실행",
                f"{'REAL 로봇' if a.real else '시뮬'}에서 경로 '{rec['name']}'을 "
                f"실행할까요?\n\n스텝 {len(rec['steps'])}개, 이동 "
                f"{len(pp.move_steps(rec))}구간.\n그리퍼 스텝은 확인 창 없이 "
                "동작합니다.\n\n주변을 비우고 비상정지를 준비하세요.",
                parent=self.gui.root):
            self.log("경로 실행 취소됨")
            return

        self._stop.clear()
        self._go.clear()
        self.gui.ui_q.put(("pathrun", True))

        def job():
            try:
                self._run_steps(rec, preview_only)
            finally:
                self.gui.ui_q.put(("pathrun", False))
                self.set_status("대기 중")
        self.gui.worker.submit(job)

    def _run_steps(self, rec, preview_only):
        a = self.gui.args
        name = rec["name"]
        steps = rec["steps"]
        mode = "프리뷰" if preview_only else ("REAL" if a.real else "시뮬")
        print(f"=== 경로 '{name}' 실행 시작 ({mode}, 스텝 {len(steps)}개) ===")
        if preview_only:
            print("    프리뷰: 저장된 궤적만 RViz로 보냅니다 (팔 위치 검사 없음, "
                  "그리퍼 동작 없음)")
        for i, step in enumerate(steps):
            if self._stop.is_set():
                print(f"=== 경로 '{name}' 정지됨 ({i + 1}번 스텝 전) ===")
                return
            label = pp.describe_step(step, i).strip()
            self.set_status(f"{i + 1}/{len(steps)}  {label}")
            print(f"[{i + 1}/{len(steps)}] {label}")
            kind = step.get("type")
            if kind == "move":
                if not self._do_move_step(step, preview_only):
                    print(f"=== 경로 '{name}' 중단 ({i + 1}번 스텝) ===")
                    return
            elif kind == "gripper":
                if not self._do_gripper_step(step, preview_only):
                    print(f"=== 경로 '{name}' 중단 ({i + 1}번 스텝) ===")
                    return
            elif kind == "wait":
                if not self._sleep(float(step.get("seconds", 0))):
                    print(f"=== 경로 '{name}' 정지됨 ({i + 1}번 스텝) ===")
                    return
            elif kind == "signal":
                if not self._do_signal_step(step, i, len(steps)):
                    print(f"=== 경로 '{name}' 중단 ({i + 1}번 스텝) ===")
                    return
            else:
                print(f"!! 알 수 없는 스텝 종류 '{kind}' — 중단")
                return
        print(f"=== 경로 '{name}' 완료 ===")
        self.gui._job_poll()

    def _sleep(self, seconds):
        """Pause, checking 정지 every tick. False when stopped."""
        end = time.time() + seconds
        while time.time() < end:
            if self._stop.is_set():
                return False
            time.sleep(min(TICK_S, max(0.0, end - time.time())))
        return True

    def _do_signal_step(self, step, i, n):
        """Today: wait for 계속. timeout_s = 0 waits indefinitely."""
        src = step.get("source")
        if src != "operator":
            print(f"!! 신호 소스 '{src}'는 아직 구현되지 않았습니다 "
                  "(형식에는 자리가 있습니다) — 중단")
            return False
        timeout = float(step.get("timeout_s") or 0.0)
        msg = step.get("message") or "계속하려면 '계속'을 누르세요"
        self._go.clear()
        self.gui.ui_q.put(("pathsignal", True))
        self.set_status(f"{i + 1}/{n}  신호 대기 — '계속'을 누르세요"
                        + (f" (제한 {timeout:g}초)" if timeout else ""))
        print(f"   {msg}")
        start = time.time()
        try:
            while not self._go.is_set():
                if self._stop.is_set():
                    return False
                if timeout and time.time() - start > timeout:
                    print(f"!! 신호 대기 {timeout:g}초 초과 — 중단")
                    return False
                time.sleep(TICK_S)
        finally:
            self.gui.ui_q.put(("pathsignal", False))
        if self._stop.is_set():
            return False
        print(f"   계속 ({time.time() - start:.1f}초 대기)")
        return True

    def _do_gripper_step(self, step, preview_only):
        action = step.get("action")
        what = "열기" if action == "open" else "닫기"
        if preview_only:
            print(f"   (프리뷰: 그리퍼 {what} 건너뜀)")
            return True
        target = self.gui._grip_target(action)
        outcome, st = self.gui._move_grip(target)
        self.gui.ui_q.put(("grip", st))
        w = "?" if st is None else f"{st['width'] * 1000:.1f} mm"
        if outcome == "reached":
            print(f"   그리퍼 {what} 완료: {w}")
            return True
        if outcome == "holding":
            # closing onto a part is the normal, successful outcome
            print(f"   그리퍼 {what}: {w}에서 정지 "
                  f"({'파지' if action == 'close' else '걸림'}, 힘 유지)")
            return action == "close"
        print(f"!! 그리퍼 {what} 실패: {outcome}")
        return False

    def _do_move_step(self, step, preview_only):
        a = self.gui.args
        q = np.asarray(pp.step_joints_rad(step), float)
        t, q, qd, qdd = jp.retime(
            q, np.asarray(pp.step_arc_len(step), float),
            np.asarray(pp.step_angle_rad(step), float),
            v=a.v, w=a.w, a=a.a, qdot=a.qdot, qddot=a.qddot)

        if preview_only:
            # A dry run walks the stored chain, so it must NOT compare against
            # the live arm: nothing is moving, so the arm sits at step 1's
            # start for the whole program and every later step would look
            # hopelessly far off.
            print(f"   {len(q)} 경로점, {t[-1]:.2f}초, IK 없음")
            ml.preview(self.gui.node, q, t, qd, qdd)
            return self._sleep(min(float(t[-1]), 2.0))

        qj = self.gui._read_joints()
        if qj is None:
            return False
        qj = np.asarray(qj, float)
        try:
            gap = pp.gap_deg(qj, step["joints_deg"][0])
        except ValueError as exc:
            print(f"!! {exc}")
            return False
        if gap > pp.START_TOL_DEG:
            print(f"!! 팔이 이 스텝의 시작 자세에서 최대 {gap:.1f}° 벗어나 "
                  f"있습니다 (허용 {pp.START_TOL_DEG:g}°) — 중단")
            print("   '시작 자세로 이동'으로 경로 처음으로 보낸 뒤 다시 "
                  "실행하세요")
            return False
        if gap > 0.1:
            q_l, t_l = lead_in(qj, q[0], a.qdot)
            q = np.vstack([q_l[:-1], q])
            t = np.concatenate([t_l[:-1], t + t_l[-1]])
            qd = qdd = None
            print(f"   진입 {t_l[-1]:.2f}초 (시작 오차 {gap:.2f}°)")
        print(f"   {len(q)} 경로점, {t[-1]:.2f}초, IK 없음")
        if not ml.exec_allowed(self.gui.node, a):
            return False
        ok, why = self._execute(ml.to_joint_trajectory(q, t, qd, qdd))
        if not ok:
            print(f"!! 이동 실패: {why}")
        return ok

    def _execute(self, jt):
        """Send a trajectory and wait, honouring 정지 by cancelling the goal."""
        node = self.gui.node
        if self._exec_client is None:
            self._exec_client = ActionClient(node, ExecuteTrajectory,
                                             "/execute_trajectory")
        if not self._exec_client.wait_for_server(timeout_sec=10.0):
            return False, "MoveIt /execute_trajectory 없음"
        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory = jt
        fut = self._exec_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
        handle = fut.result()
        if handle is None or not handle.accepted:
            return False, "실행 거부됨"
        res = handle.get_result_async()
        cancelled = False
        while not res.done():
            rclpy.spin_once(node, timeout_sec=TICK_S)
            if self._stop.is_set() and not cancelled:
                handle.cancel_goal_async()
                cancelled = True
                print("   이동 취소 요청 보냄")
        if cancelled:
            return False, "정지됨"
        code = res.result().result.error_code.val
        return code == 1, f"error_code={code}"
