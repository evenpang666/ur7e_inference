# VLA 推理

本项目适配 Sci-VLA/OpenPI 的 `mani_real_pi05`。观测包含 D435i 主视角、Pika 腕部 RGB、UR7e 6 维关节角、1 维夹爪状态和任务文本；服务返回 `[chunk, 7]` 的绝对 action，前 6 维是关节目标，第 7 维是夹爪目标。

夹爪语义为 `0=张开、1=闭合`。使用新采集数据训练的 checkpoint 时保持 `gripper.invert: true`，运行时会自动转换为实体电机角度，无需对 VLA action 再手工反转。旧 checkpoint 若训练数据为相反语义，必须使用匹配的配置，不能与新数据混用。

## 启动与验证

先在推理主机启动服务，具体安装与启动命令见[环境安装](environment_setup.md)。机器人端先验证网络和 action 形状：

```bash
python scripts/probe_policy.py --host 192.168.124.15 --port 8000
```

## 单任务运行

先 dry-run：

```bash
ur7e-vla run --config config.yaml --task "把红色方块放进碗里" --duration 30
```

确认图像、7 维 action、关节方向、夹爪范围和安全限位后，再执行真机：

```bash
ur7e-vla run --config config.yaml --task "把红色方块放进碗里" --execute
```

不传 `--duration` 且 `runtime.duration_s: null` 时会持续运行，使用 `Ctrl+C` 停止。可加 `--record-video` 录制双相机拼接 MP4，使用 `--recording-dir` 覆盖输出目录。

## 多任务序列

```powershell
ur7e-vla run-sequence --config config.yaml `
  --step "open lid" 20 `
  --step "place pcr plate" 30 --execute
```

每个任务使用新的任务文本重新采样，避免前一任务的 action 被用于后一任务。默认从前一任务结束状态继续；加入 `--restore-each-task-initial-state` 可在每个任务前恢复 JSON 初始状态。

## 初始状态恢复

```bash
ur7e-vla run --config config.yaml --task "open lid" \
  --restore-initial-state --execute
```

`initial_robot_states.json` 的任务键会规范化为小写下划线。恢复会插值关节和夹爪，检查到位误差后才开始推理。当前文件标记为 `ur5e`，使用前必须确认关节顺序、零位和工作空间适用于 UR7e。

## 推理模式与安全

`policy.inference_mode` 可设为：

- `synchronous`：执行当前 chunk 的前 `synchronous_execute_steps` 步后请求新 action。
- `asynchronous`：默认模式；低于队列阈值时后台请求新 action，并按 `async_merge_mode` 合并。

可临时覆盖：

```bash
ur7e-vla run --config config.yaml --task "open lid" --inference-mode synchronous --execute
ur7e-vla run --config config.yaml --task "open lid" --inference-mode asynchronous --async-merge-mode weighted_blend --execute
```

上真机前确认关节单位和顺序、夹爪电机范围及反向配置、关节软限位、相机新鲜度和推理延迟。先 dry-run 和低速小范围动作验证，不要绕过 `--execute` 保护。
