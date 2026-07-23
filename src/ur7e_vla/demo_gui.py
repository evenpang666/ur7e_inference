from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from .config import AppConfig
from .demo_collection import LiveDemoCollector, PendingEpisode, save_pending_episode

LOG = logging.getLogger(__name__)


class DemoCollectorGUI:
    """GUI with independent teleoperation and episode-recording lifecycles."""

    def __init__(self, cfg: AppConfig, task: str, execute: bool):
        import tkinter as tk

        self.tk = tk
        self.cfg = cfg
        self.execute = execute
        self.root = tk.Tk()
        self.root.title("UR7e / Pika Demo Collection")
        self.root.geometry("760x300")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.teleop_stop = threading.Event()
        self.record_event = threading.Event()
        self.pending: Optional[PendingEpisode] = None
        self.record_task: Optional[str] = None
        self.teleop_running = False
        self.teleop_starting = False
        self.recording = False
        self.saving = False

        tk.Label(self.root, text="Task description").pack(anchor="w", padx=16, pady=(14, 2))
        self.task_var = tk.StringVar(value=task)
        self.task_entry = tk.Entry(self.root, textvariable=self.task_var, font=("Segoe UI", 12))
        self.task_entry.pack(fill="x", padx=16)
        self.status_var = tk.StringVar(value="Enter a task, then click Start Teleoperation.")
        tk.Label(self.root, textvariable=self.status_var, justify="left", wraplength=720).pack(
            fill="x", padx=16, pady=16
        )

        row = tk.Frame(self.root)
        row.pack(fill="x", padx=12)
        self.start_teleop_button = tk.Button(row, text="Start Teleoperation", command=self.start_teleoperation)
        self.pause_teleop_button = tk.Button(
            row, text="Pause Teleoperation", command=self.pause_teleoperation, state="disabled"
        )
        self.record_button = tk.Button(row, text="Start Recording", command=self.start_recording, state="disabled")
        self.stop_button = tk.Button(row, text="Stop Recording", command=self.stop_recording, state="disabled")
        self.save_button = tk.Button(row, text="Save Episode", command=self.save, state="disabled")
        self.discard_button = tk.Button(row, text="Discard Episode", command=self.discard, state="disabled")
        for button in (
            self.start_teleop_button,
            self.pause_teleop_button,
            self.record_button,
            self.stop_button,
            self.save_button,
            self.discard_button,
        ):
            button.pack(side="left", padx=3, pady=5)

        mode = "Physical teleoperation enabled." if execute else "Dry run: restart with --execute to enable teleoperation."
        tk.Label(self.root, text=mode, fg="#276749" if execute else "#9b2c2c").pack(anchor="w", padx=16, pady=10)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._poll)

    def _task(self) -> str:
        task = self.task_var.get().strip()
        if not task:
            raise ValueError("Task description cannot be empty")
        return task

    def _worker(self, name: str, fn) -> None:
        def work():
            try:
                self.events.put((name, fn()))
            except BaseException as exc:
                LOG.exception("Demo collection operation failed")
                self.events.put(("error", exc))

        threading.Thread(target=work, name=f"demo-{name}", daemon=True).start()

    def start_teleoperation(self) -> None:
        from tkinter import messagebox

        if not self.execute:
            messagebox.showwarning("Teleoperation disabled", "Restart with --execute to move the physical robot.")
            return
        if self.teleop_running or self.teleop_starting:
            return
        if not messagebox.askyesno(
            "Start teleoperation",
            "The Pika Sensor will control the UR7e and gripper. Confirm the cell is clear and E-stop is reachable.",
        ):
            return
        self.teleop_starting = True
        self.teleop_stop.clear()
        self.record_event.clear()
        self.start_teleop_button.configure(state="disabled")
        self.status_var.set("Connecting Pika Sense, UR7e, gripper, and cameras…")

        def run_teleop() -> None:
            LiveDemoCollector(self.cfg).run(
                self.teleop_stop,
                self.record_event,
                lambda pending: self.events.put(("record_done", pending)),
                lambda: self.events.put(("teleop_ready", None)),
                lambda count: self.events.put(("record_progress", count))
                if count % self.cfg.demo.fps == 0
                else None,
            )

        self._worker("teleop_done", run_teleop)

    def pause_teleoperation(self) -> None:
        if not self.teleop_running and not self.teleop_starting:
            return
        self.record_event.clear()
        self.recording = False
        self.teleop_stop.set()
        self.pause_teleop_button.configure(state="disabled")
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status_var.set("Pausing teleoperation and releasing hardware…")

    def start_recording(self) -> None:
        from tkinter import messagebox

        try:
            self.record_task = self._task()
        except ValueError as exc:
            messagebox.showerror("Invalid task", str(exc))
            return
        if not self.teleop_running:
            messagebox.showwarning("Teleoperation required", "Click Start Teleoperation before recording.")
            return
        if self.pending is not None or self.recording or self.saving:
            return
        self.recording = True
        self.record_event.set()
        self.task_entry.configure(state="disabled")
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.pause_teleop_button.configure(state="normal")
        self.status_var.set("Recording episode while teleoperation continues…")

    def stop_recording(self) -> None:
        if not self.recording:
            return
        self.recording = False
        self.record_event.clear()
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping recording; teleoperation remains active…")

    def save(self) -> None:
        if self.pending is None or self.record_task is None or self.saving:
            return
        pending, task = self.pending, self.record_task
        self.saving = True
        self.save_button.configure(state="disabled")
        self.discard_button.configure(state="disabled")
        self.status_var.set("Encoding and saving the episode… teleoperation remains active.")
        self._worker("save_done", lambda: save_pending_episode(pending, task, self.cfg.demo))

    def discard(self) -> None:
        if self.pending is None or self.saving:
            return
        self.pending.discard()
        self.pending = None
        self.record_task = None
        self.task_entry.configure(state="normal")
        self.save_button.configure(state="disabled")
        self.discard_button.configure(state="disabled")
        self.record_button.configure(state="normal" if self.teleop_running else "disabled")
        self.status_var.set("Episode discarded. Teleoperation remains active.")

    def _poll(self) -> None:
        from tkinter import messagebox

        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "teleop_ready":
                    self.teleop_starting = False
                    self.teleop_running = True
                    self.pause_teleop_button.configure(state="normal")
                    self.record_button.configure(state="normal" if self.pending is None else "disabled")
                    self.status_var.set("Teleoperation active. Enter or edit the task description, then click Start Recording.")
                elif kind == "record_progress":
                    self.status_var.set(f"Recording episode: {int(payload)} frames. Teleoperation is active.")
                elif kind == "record_done":
                    self.pending = payload  # type: ignore[assignment]
                    self.task_entry.configure(state="normal")
                    self.save_button.configure(state="normal")
                    self.discard_button.configure(state="normal")
                    self.record_button.configure(state="disabled")
                    self.status_var.set(
                        f"Recording stopped: {len(self.pending.frames)} frames staged. "
                        "Teleoperation remains active; save or discard this episode."
                    )
                elif kind == "save_done":
                    dataset_root, episode_index = payload  # type: ignore[misc]
                    if self.pending is not None:
                        self.pending.discard()
                    self.pending = None
                    self.record_task = None
                    self.saving = False
                    self.task_entry.configure(state="normal")
                    self.discard_button.configure(state="disabled")
                    self.record_button.configure(state="normal" if self.teleop_running else "disabled")
                    self.status_var.set(f"Saved episode_{episode_index:06d} to {dataset_root}. Teleoperation remains active.")
                    messagebox.showinfo("Episode saved", self.status_var.get())
                elif kind == "teleop_done":
                    self.teleop_starting = False
                    self.teleop_running = False
                    self.recording = False
                    self.record_event.clear()
                    self.start_teleop_button.configure(state="normal")
                    self.pause_teleop_button.configure(state="disabled")
                    self.record_button.configure(state="disabled")
                    self.stop_button.configure(state="disabled")
                    if self.pending is None:
                        self.status_var.set("Teleoperation paused.")
                elif kind == "error":
                    self.teleop_starting = False
                    self.teleop_running = False
                    self.recording = False
                    self.saving = False
                    self.record_event.clear()
                    self.teleop_stop.set()
                    self.task_entry.configure(state="normal")
                    self.start_teleop_button.configure(state="normal")
                    self.pause_teleop_button.configure(state="disabled")
                    self.record_button.configure(state="disabled")
                    self.stop_button.configure(state="disabled")
                    self.status_var.set(f"Operation failed: {payload}")
                    messagebox.showerror("Demo collection failed", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def close(self) -> None:
        from tkinter import messagebox

        if (self.teleop_running or self.teleop_starting or self.saving) and not messagebox.askyesno(
            "Exit", "Teleoperation or saving is active. Stop and exit?"
        ):
            return
        self.record_event.clear()
        self.teleop_stop.set()
        if self.pending is not None:
            self.pending.discard()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch_demo_gui(cfg: AppConfig, task: str, execute: bool) -> None:
    DemoCollectorGUI(cfg, task, execute).run()
