"""Interactive operator interface for a single, continuous VLA task."""

from __future__ import annotations

import copy
import logging
import queue
import threading

from .config import AppConfig
from .runtime import VLARuntime

LOG = logging.getLogger(__name__)


class VLARuntimeGUI:
    """Keep tkinter work on the UI thread and robot work on background workers."""

    def __init__(self, cfg: AppConfig, task: str, execute: bool):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.cfg = cfg
        self.execute = execute
        self.root = tk.Tk()
        self.root.title("UR7e VLA Inference")
        self.root.geometry("1080x310")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        # Hardware belongs to the window, rather than an individual VLA session.
        self.runtime = VLARuntime(copy.deepcopy(cfg), execute=execute, keep_hardware_connected=True)
        self.running = False
        self.starting = False
        self.recording = False
        self.restart_requested = False
        self.restart_restore_initial_state = False
        self.pending_task: str | None = None

        tk.Label(self.root, text="Task instruction").pack(anchor="w", padx=16, pady=(14, 2))
        self.task_var = tk.StringVar(value=task or cfg.runtime.task)
        task_row = tk.Frame(self.root)
        task_row.pack(fill="x", padx=16)
        self.task_entry = tk.Entry(task_row, textvariable=self.task_var, font=("Segoe UI", 12))
        self.task_entry.pack(side="left", fill="x", expand=True)
        self.apply_task_button = tk.Button(task_row, text="Apply Task", command=self.apply_task)
        self.apply_task_button.pack(side="left", padx=(8, 0))
        self.restore_var = tk.BooleanVar(value=False)
        tk.Checkbutton(task_row, text="Restore initial state", variable=self.restore_var).pack(side="left", padx=(12, 0))

        mode_row = tk.Frame(self.root)
        mode_row.pack(fill="x", padx=16, pady=(12, 2))
        tk.Label(mode_row, text="Inference mode:").pack(side="left")
        self.mode_var = tk.StringVar(value=cfg.policy.inference_mode)
        self.mode_box = ttk.Combobox(
            mode_row, textvariable=self.mode_var, values=("synchronous", "asynchronous"), state="readonly", width=18
        )
        self.mode_box.pack(side="left", padx=8)
        self.apply_mode_button = tk.Button(mode_row, text="Apply Mode", command=self.apply_mode)
        self.apply_mode_button.pack(side="left")

        self.status_var = tk.StringVar(value="Enter an instruction and click Apply Task to start VLA inference.")
        tk.Label(self.root, textvariable=self.status_var, justify="left", wraplength=1040).pack(
            fill="x", padx=16, pady=14
        )
        buttons = tk.Frame(self.root)
        buttons.pack(fill="x", padx=12)
        self.stop_button = tk.Button(buttons, text="Stop VLA", command=self.stop, state="disabled")
        self.video_button = tk.Button(buttons, text="Start Video", command=self.toggle_video)
        for button in (self.stop_button, self.video_button):
            button.pack(side="left", padx=3, pady=5)

        safety = "Physical execution enabled." if execute else "Dry run: restart with --execute to send robot/gripper commands."
        tk.Label(self.root, text=safety, fg="#276749" if execute else "#9b2c2c").pack(anchor="w", padx=16, pady=8)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._poll)

    def _task(self) -> str:
        task = self.task_var.get().strip()
        if not task:
            raise ValueError("Task instruction cannot be empty")
        return task

    def _worker(self, name: str, fn) -> None:
        def work() -> None:
            try:
                self.events.put((name, fn()))
            except BaseException as exc:
                LOG.exception("VLA GUI operation failed")
                self.events.put(("error", exc))

        threading.Thread(target=work, name=f"vla-gui-{name}", daemon=True).start()

    def _start_vla(self, restore_initial_state: bool = False, task_override: str | None = None) -> None:
        from tkinter import messagebox

        if self.running or self.starting:
            return
        try:
            task = (task_override if task_override is not None else self._task()).strip()
            if not task:
                raise ValueError("Task instruction cannot be empty")
        except ValueError as exc:
            messagebox.showerror("Invalid task", str(exc))
            return
        if self.execute and not messagebox.askyesno(
            "Start VLA execution", "The policy can move the UR7e and Pika gripper. Confirm the cell is clear and E-stop is reachable."
        ):
            return
        run_cfg = copy.deepcopy(self.cfg)
        run_cfg.policy.inference_mode = self.mode_var.get()
        run_cfg.runtime.record_video = False
        run_cfg.initial_state.enabled = restore_initial_state
        run_cfg.validate()
        self.runtime.configure_gui_session(run_cfg)
        self.starting = True
        self.status_var.set(f"Connecting hardware and policy for {run_cfg.policy.inference_mode} inference...")
        self._worker("finished", lambda: self.runtime.run(task, on_ready=lambda: self.events.put(("ready", None)), dynamic_task=True))

    def apply_task(self) -> None:
        try:
            task = self._task()
        except ValueError as exc:
            from tkinter import messagebox
            messagebox.showerror("Invalid task", str(exc))
            return
        if self.restart_requested or self.starting:
            self.pending_task = task
            self.status_var.set("Task queued; it will be used when the current transition completes.")
        elif not self.running:
            self._start_vla(restore_initial_state=self.restore_var.get(), task_override=task)
        elif self.restore_var.get():
            self._restart(True, "Stopping VLA, restoring the initial state, then applying the task.")
        else:
            self.runtime.set_task(task)
            self.status_var.set("Requesting the new task now; stale queued actions will be discarded safely.")

    def apply_mode(self) -> None:
        if not self.running and not self.starting:
            self._start_vla(restore_initial_state=self.restore_var.get())
            return
        try:
            self._task()
        except ValueError as exc:
            from tkinter import messagebox
            messagebox.showerror("Invalid task", str(exc))
            return
        self._restart(False, "Stopping current action queue before applying the selected inference mode.")

    def _restart(self, restore_initial_state: bool, status: str) -> None:
        self.restart_requested = True
        self.restart_restore_initial_state = restore_initial_state
        self.pending_task = self._task()
        self.stop_button.configure(state="disabled")
        self.apply_mode_button.configure(state="disabled")
        # Video controls intentionally remain usable during every VLA transition.
        self.status_var.set(status)
        self.runtime.request_stop()

    def stop(self) -> None:
        self.restart_requested = False
        self.pending_task = None
        self.stop_button.configure(state="disabled")
        self.apply_mode_button.configure(state="disabled")
        self.status_var.set("Stopping VLA; hardware and video recording remain connected.")
        self.runtime.request_stop()

    def toggle_video(self) -> None:
        self.video_button.configure(state="disabled")
        if self.recording:
            self.status_var.set("Stopping video recording; VLA state is unchanged.")
            self._worker("video_stopped", self.runtime.stop_video_recording)
        else:
            self.status_var.set("Starting video recording; VLA state is unchanged.")
            self._worker("video_started", self.runtime.start_video_recording)

    def _poll(self) -> None:
        from tkinter import messagebox

        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "ready":
                    self.starting = False
                    self.running = True
                    self.stop_button.configure(state="normal")
                    self.apply_mode_button.configure(state="normal")
                    self.status_var.set(f"VLA active in {self.mode_var.get()} mode.")
                elif kind == "video_started":
                    self.recording = True
                    self.video_button.configure(text="Stop Video", state="normal")
                    self.status_var.set(f"Video recording: {payload}.")
                elif kind == "video_stopped":
                    self.recording = False
                    self.video_button.configure(text="Start Video", state="normal")
                    self.status_var.set(f"Video saved: {payload}.")
                elif kind == "finished":
                    self.running = self.starting = False
                    if self.restart_requested:
                        restore = self.restart_restore_initial_state
                        task = self.pending_task or self._task()
                        self.restart_requested = False
                        self.pending_task = None
                        self.status_var.set("Applying updated task/mode...")
                        self._start_vla(restore_initial_state=restore, task_override=task)
                    else:
                        self.apply_mode_button.configure(state="normal")
                        self.stop_button.configure(state="disabled")
                        self.status_var.set("VLA stopped. Hardware remains connected; video controls remain available.")
                elif kind == "error":
                    self.running = self.starting = self.recording = False
                    self.restart_requested = False
                    self.pending_task = None
                    self.runtime.request_stop()
                    self.apply_mode_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.video_button.configure(text="Start Video", state="normal")
                    self.status_var.set(f"Operation failed: {payload}")
                    messagebox.showerror("VLA inference failed", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def close(self) -> None:
        from tkinter import messagebox

        if (self.running or self.starting) and not messagebox.askyesno("Exit", "Stop VLA and exit?"):
            return
        self.restart_requested = False
        self.pending_task = None
        self.runtime.request_stop()
        self.runtime.stop_video_recording()
        self.runtime.disconnect_hardware()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch_vla_gui(cfg: AppConfig, task: str, execute: bool) -> None:
    VLARuntimeGUI(cfg, task, execute).run()
