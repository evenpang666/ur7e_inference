from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from .config import AppConfig
from .demo_collection import (
    DemoReplay,
    DemonstrationTrajectory,
    PendingEpisode,
    PikaSenseSource,
    record_sensor_trajectory,
    save_pending_episode,
)

LOG = logging.getLogger(__name__)


class DemoCollectorGUI:
    def __init__(self, cfg: AppConfig, task: str, execute: bool):
        import tkinter as tk

        self.tk = tk
        self.cfg = cfg
        self.execute = execute
        self.root = tk.Tk()
        self.root.title("UR7e / Pika LeRobot 演示采集")
        self.root.geometry("650x330")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.sensor_stop = threading.Event()
        self.replay_stop = threading.Event()
        self.source: Optional[PikaSenseSource] = None
        self.trajectory: Optional[DemonstrationTrajectory] = None
        self.pending: Optional[PendingEpisode] = None
        self.busy = False
        self.discard_requested = False

        tk.Label(self.root, text="任务描述").pack(anchor="w", padx=16, pady=(14, 2))
        self.task_var = tk.StringVar(value=task)
        self.task_entry = tk.Entry(self.root, textvariable=self.task_var, font=("Segoe UI", 12))
        self.task_entry.pack(fill="x", padx=16)
        self.status_var = tk.StringVar(value="就绪：先录制 Pika Sensor 完整路径")
        tk.Label(self.root, textvariable=self.status_var, justify="left", wraplength=610).pack(
            fill="x", padx=16, pady=16
        )
        button_row = tk.Frame(self.root)
        button_row.pack(fill="x", padx=12)
        self.record_button = tk.Button(button_row, text="开始录制 Sensor", command=self.start_recording)
        self.stop_button = tk.Button(button_row, text="停止录制", command=self.stop_recording, state="disabled")
        self.replay_button = tk.Button(button_row, text="开始机械臂回放", command=self.start_replay, state="disabled")
        self.save_button = tk.Button(button_row, text="保存 Episode", command=self.save, state="disabled")
        self.discard_button = tk.Button(button_row, text="舍弃并重录", command=self.discard, state="disabled")
        for button in (self.record_button, self.stop_button, self.replay_button, self.save_button, self.discard_button):
            button.pack(side="left", padx=4, pady=5)
        self.mode_var = tk.StringVar(
            value="真实回放已启用" if execute else "预览模式：未传 --execute，机械臂回放按钮不会启用"
        )
        tk.Label(self.root, textvariable=self.mode_var, fg="#9b2c2c" if not execute else "#276749").pack(
            anchor="w", padx=16, pady=12
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._poll)

    def _task(self) -> str:
        task = self.task_var.get().strip()
        if not task:
            raise ValueError("任务描述不能为空")
        return task

    def _background(self, name: str, fn) -> None:
        self.busy = True

        def worker():
            try:
                self.events.put((name, fn()))
            except BaseException as exc:
                LOG.exception("Demo collection operation failed")
                self.events.put(("error", exc))

        threading.Thread(target=worker, name=f"demo-{name}", daemon=True).start()

    def start_recording(self) -> None:
        from tkinter import messagebox

        try:
            self._task()
        except ValueError as exc:
            messagebox.showerror("任务无效", str(exc))
            return
        if self.pending is not None:
            self.pending.discard()
            self.pending = None
        self.trajectory = None
        self.sensor_stop.clear()
        self.discard_requested = False
        self.task_entry.configure(state="disabled")
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.replay_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.discard_button.configure(state="disabled")
        self.status_var.set("正在连接 Pika Sense 和 Lighthouse 基站……")

        def work():
            if self.source is None or not self.source.is_ready:
                self.source = PikaSenseSource(self.cfg.demo)
                try:
                    self.source.connect()
                except BaseException:
                    self.source.close()
                    self.source = None
                    raise
            self.events.put(("record_progress", 0))
            return record_sensor_trajectory(
                self.source,
                self.cfg.demo.fps,
                self.sensor_stop,
                lambda count: self.events.put(("record_progress", count)) if count % self.cfg.demo.fps == 0 else None,
            )

        self._background("record_done", work)

    def stop_recording(self) -> None:
        self.sensor_stop.set()
        self.stop_button.configure(state="disabled")
        self.status_var.set("正在结束 Sensor 轨迹录制……")

    def start_replay(self) -> None:
        from tkinter import messagebox

        if not self.execute:
            messagebox.showwarning("未启用真实回放", "请关闭窗口并用 --execute 重新启动。")
            return
        if self.trajectory is None:
            return
        if not messagebox.askyesno(
            "确认机械臂回放",
            "即将连接 UR7e、夹爪和两路相机并执行示教轨迹。\n请清空工作区、释放急停并确认示教器处于 Remote Control。",
        ):
            return
        self.replay_stop.clear()
        self.discard_requested = False
        self.replay_button.configure(state="disabled")
        self.record_button.configure(state="disabled")
        self.discard_button.configure(state="normal")
        self.status_var.set("正在做逆解并回放；图像、实际关节角和 action 正在暂存……")

        def work():
            replay = DemoReplay(self.cfg)
            return replay.run(
                self.trajectory,
                self.replay_stop,
                lambda count: self.events.put(("replay_progress", count)) if count % self.cfg.demo.fps == 0 else None,
            )

        self._background("replay_done", work)

    def save(self) -> None:
        if self.pending is None:
            return
        pending = self.pending
        self.save_button.configure(state="disabled")
        self.discard_button.configure(state="disabled")
        self.status_var.set("正在编码视频并提交 LeRobot episode……")
        self._background("save_done", lambda: save_pending_episode(pending, self._task(), self.cfg.demo))

    def discard(self) -> None:
        self.sensor_stop.set()
        self.replay_stop.set()
        if self.busy:
            self.discard_requested = True
            self.record_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.replay_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
            self.discard_button.configure(state="disabled")
            self.status_var.set("正在停止当前操作并舍弃暂存……")
            return
        self._finish_discard()

    def _finish_discard(self) -> None:
        if self.pending is not None:
            self.pending.discard()
            self.pending = None
        self.trajectory = None
        self.busy = False
        self.task_entry.configure(state="normal")
        self.record_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.replay_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.discard_button.configure(state="disabled")
        self.status_var.set("已舍弃，未修改 demo 数据集；可以重新录制 Sensor 路径")

    def _poll(self) -> None:
        from tkinter import messagebox

        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "record_progress":
                    count = int(payload)
                    self.status_var.set(
                        "Sensor 已连接，正在录制……" if count == 0 else f"正在录制 Sensor 路径：{count} 帧"
                    )
                elif kind == "record_done":
                    self.busy = False
                    if self.discard_requested:
                        self.discard_requested = False
                        self._finish_discard()
                        continue
                    self.trajectory = payload  # type: ignore[assignment]
                    self.stop_button.configure(state="disabled")
                    self.replay_button.configure(state="normal" if self.execute else "disabled")
                    self.discard_button.configure(state="normal")
                    self.status_var.set(f"Sensor 路径录制完成：{len(self.trajectory.samples)} 帧。请点击开始机械臂回放。")
                elif kind == "replay_progress":
                    self.status_var.set(f"正在回放并暂存 episode：{int(payload)} 帧")
                elif kind == "replay_done":
                    self.busy = False
                    if self.discard_requested:
                        self.discard_requested = False
                        payload.discard()  # type: ignore[union-attr]
                        self._finish_discard()
                        continue
                    self.pending = payload  # type: ignore[assignment]
                    self.save_button.configure(state="normal")
                    self.discard_button.configure(state="normal")
                    self.status_var.set(
                        f"回放完成：暂存 {len(self.pending.frames)} 帧。点击保存才会写入数据集，或舍弃并重录。"
                    )
                elif kind == "save_done":
                    self.busy = False
                    dataset_root, episode_index = payload  # type: ignore[misc]
                    if self.pending is not None:
                        self.pending.discard()
                    self.pending = None
                    self.trajectory = None
                    self.task_entry.configure(state="normal")
                    self.record_button.configure(state="normal")
                    self.discard_button.configure(state="disabled")
                    self.status_var.set(f"已保存 episode_{episode_index:06d} 到 {dataset_root}")
                    messagebox.showinfo("保存成功", self.status_var.get())
                elif kind == "error":
                    self.busy = False
                    if self.discard_requested:
                        self.discard_requested = False
                        self._finish_discard()
                        continue
                    self.sensor_stop.set()
                    self.replay_stop.set()
                    if self.source is not None:
                        self.source.close()
                        self.source = None
                    self.task_entry.configure(state="normal")
                    self.record_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.replay_button.configure(state="normal" if self.trajectory and self.execute else "disabled")
                    self.discard_button.configure(state="normal" if self.trajectory or self.pending else "disabled")
                    self.status_var.set(f"失败：{payload}")
                    messagebox.showerror("采集失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def close(self) -> None:
        from tkinter import messagebox

        if self.busy and not messagebox.askyesno("退出", "采集或保存仍在进行，确定停止并退出吗？"):
            return
        self.sensor_stop.set()
        self.replay_stop.set()
        if self.pending is not None:
            self.pending.discard()
        if self.source is not None:
            self.source.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch_demo_gui(cfg: AppConfig, task: str, execute: bool) -> None:
    DemoCollectorGUI(cfg, task, execute).run()
