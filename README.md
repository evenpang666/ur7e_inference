# UR7e + Pika + D435i 远程 π0.5 推理

机器人侧采集 D435i 主视角、Pika 腕部 RGB、UR7e 关节角、Pika 夹爪状态和任务文本，通过 OpenPI WebSocket 客户端请求 `192.168.124.15:8000`，接收 action chunk 后用 `ur-rtde` 执行。

## mani_real_pi05 数据协议

本工程严格适配以下 Sci-VLA 配置：

- Policy：`third_party/openpi/src/openpi/policies/ur_policy.py`
- Config：`mani_real_pi05`

每次发送的 observation 为：

```python
{
    "observation/image": uint8[224, 224, 3],       # D435i，RGB/HWC
    "observation/wrist_image": uint8[224, 224, 3], # Pika腕部相机，RGB/HWC
    "observation/joints": float32[6],              # UR7e实际关节角，rad
    "observation/gripper": float32[1],             # 训练数据使用的夹爪量纲
    "prompt": str,
}
```

服务端返回 `actions: float[chunk, 7]`。`mani_real_pi05` 在服务端使用 `AbsoluteActions(make_bool_mask(6, -1))`，因此前6维是绝对关节目标，第7维是绝对夹爪目标。本地配置固定为 `joint_position`，夹爪 action 索引为6。

`gripper.policy_min/policy_max` 表示 `mani_real` 训练数据的夹爪范围，已确认为 `[0,1]`；`min_angle_rad/max_angle_rad` 表示 Pika 实际电机角范围。本工程负责在两者之间线性映射。

程序默认 dry-run；只有传入 `--execute` 才会向 UR7e 和夹爪下发命令。每步关节变化、速度、软限位、相机新鲜度和推理延迟均有安全限制。

## 1. 推理主机 192.168.124.15

在你的 Sci-VLA OpenPI 目录启动：

```bash
cd /path/to/Sci-VLA/third_party/openpi
POLICY_DIR=/path/to/mani_real_pi05/checkpoint \
  /path/to/ur7e_inference/scripts/start_pi05_server.sh
```

等价命令：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=mani_real_pi05 \
  --policy.dir=/path/to/mani_real_pi05/checkpoint \
  --port=8000
```

OpenPI 服务监听 `0.0.0.0:8000`。确认防火墙允许机器人侧访问 TCP 8000。

## 2. 机器人侧安装

Pika SDK 官方测试环境为 Ubuntu 20.04/22.04 与 Python 3.9：

```bash
pip install agx-pypika

cd /path/to/Sci-VLA/third_party/openpi
pip install -e packages/openpi-client

cd /path/to/ur7e_inference
pip install -e .
cp config.example.yaml config.yaml
```

查找 Pika UVC 腕部相机编号：

```bash
ur7e-vla list-cameras
```

把编号写入 `cameras.wrist_device`。有多个 RealSense 时填写 `realsense_serial`；同时确认 Pika 串口、USB权限、工具TCP和负载。

机器人侧主机必须同时能访问 UR7e 的 `169.254.175.10` 和推理主机的 `192.168.124.15`。两个地址处于不同网段，通常需要两张网卡或正确的静态路由。

## 3. 运行

先用合成观测完成服务握手、预热并验证返回为 `[chunk, 7]`；该命令不会连接机械臂：

```bash
python scripts/probe_policy.py --host 192.168.124.15 --port 8000
```

先做30秒 dry-run：

```bash
ur7e-vla run --config config.yaml --task "把红色方块放进碗里" --duration 30
```

核对图像、7维 action、关节方向、夹爪范围和安全限位后，降低示教器速度并保持急停可达，再启用真实动作：

```bash
ur7e-vla run --config config.yaml --task "把红色方块放进碗里" --execute
```

### 长序列任务模式

`run-sequence` 会在一次硬件、相机和策略连接中，顺序执行多个原子任务。每个任务都必须提供正的运行时长；前一个任务结束后，程序会丢弃其未执行 action，使用新任务文本重新采样并预取 action，再开始下一个任务。这样 `open lid` 的 action 不会被用于 `place pcr plate`。

```powershell
ur7e-vla run-sequence --config config.yaml `
  --step "open lid" 20 `
  --step "place pcr plate" 30
```

确认 dry-run 后加上 `--execute`。也可把序列写入 `config.yaml` 的 `runtime.sequence`，每项为 `task` 和 `duration_s`，再直接运行 `ur7e-vla run-sequence --config config.yaml --execute`。`--record-video` 会将整个序列录制为一个视频。

默认情况下，后续任务从前一个任务的实际结束状态继续执行。若要在**每个**任务前从 `initial_robot_states.json` 检索并恢复该任务初始态，加入 `--restore-each-task-initial-state`：

```powershell
ur7e-vla run-sequence --config config.yaml `
  --step "open lid" 20 `
  --step "place pcr plate" 30 `
  --restore-each-task-initial-state --execute
```

也可设置 `runtime.restore_before_each_task: true`；其默认值是 `false`。原有的 `--restore-initial-state` 仍只在首任务前恢复。

### 自动录像

通过命令行开启录像：

```bash
ur7e-vla run --config config.yaml --task "open lid" --record-video
```

也可以在 `config.yaml` 中设置 `runtime.record_video: true`。录像默认以30 FPS采样，将 D435i 主视角和 Pika 腕部视角横向拼接到同一个 MP4，保存到 `recordings/run_时间戳.mp4`。使用 `--recording-dir D:\videos` 可以覆盖输出目录。

保持 `runtime.duration_s: null` 且不传 `--duration`，程序会持续运行和录像，按 `Ctrl+C` 后停止并自动完成视频封装。录像可以和 dry-run 或 `--execute` 同时使用。

### 任务初始状态恢复

运行时可在首次 VLA 推理前，从 `initial_robot_states.json` 检索任务对应的初始状态并平滑恢复：

```bash
ur7e-vla run --config config.yaml --task "open lid" --restore-initial-state --execute
```

任务名会规范化为小写下划线形式，例如 `open lid` 对应 JSON 中的 `open_lid`。一个任务包含多个 episode 时，默认选择与当前6轴关节状态最接近、最大单轴移动量最小的 episode。也可以固定选择：

```bash
ur7e-vla run --config config.yaml --task "open lid" \
  --restore-initial-state --initial-state-episode episode_000003 --execute
```

对应配置位于 `initial_state`：`enabled` 控制开关，`path` 指定 JSON，`interpolation_hz` 默认50 Hz，`max_joint_speed_rad_s` 默认0.15，`max_gripper_speed_per_s` 默认0.25。程序会同时插值6轴关节和夹爪，恢复结束后检查关节误差，合格后才连接策略并预取首个 action chunk。dry-run 只检索、校验和显示目标，不发送恢复动作。

当前 `initial_robot_states.json` 的 `robot_type` 标为 `ur5e`，程序会给出警告后按6关节顺序应用。启用前必须确认其关节顺序、零位定义和工作空间路径适用于当前 UR7e。关节空间插值不提供碰撞规划，仍需收紧软限位、清空路径并保持急停可达。

不传 `--duration` 且配置为 `duration_s: null` 时无限运行；`--duration 120` 表示动作循环运行120秒。`Ctrl+C`、SIGTERM、通信异常、超过推理延迟、相机画面陈旧、非法 action 或关节越界都会停止伺服。

`mani_real` 数据集频率已确认为15 FPS。首个 action chunk 会在运动前预取。运行中以15 Hz执行每个chunk的10步，并在第5步后异步采样、推理下一chunk；推理与后5步动作并行。切换chunk时会根据采样后已经过去的控制周期丢弃新chunk开头的过期动作，例如第5步采样、第10步切换时从新chunk的第6个动作开始。如果推理偶尔未在10步结束前返回，控制线程会持续发送最后目标保持UR伺服，并把这些保持周期计入动作时间对齐；整个预测块都已过期时程序会安全停止。

## 上真机前必须确认

- `mani_real` 的关节单位确实为弧度，顺序与 UR RTDE 的6轴顺序一致。
- `mani_real` 夹爪原始范围已确认为 `[0,1]`；若重新制作数据集或更换checkpoint，需要重新核对。
- 标定 Pika 的 `min_angle_rad/max_angle_rad` 和 `invert`。
- 将 `joint_min_rad/joint_max_rad` 收紧为当前工作站的安全软限位。
- 先使用 dry-run 和低速、小范围动作验证，不要绕过 `--execute` 保护。

## LeRobot 示教数据采集

采集器针对 `mani_real_pi05` 生成 LeRobot v2.1 数据：两路 RGB 图像、6 维实际关节角、1 维实际夹爪状态，以及 7 维绝对 action（6 轴目标关节角 + 夹爪目标）。任务文本写入 LeRobot task 元数据，训练时由 `prompt_from_task` 转成 `prompt`。

安装与 OpenPI 配置一致的 LeRobot 版本：

```powershell
pip install -e ".[demo]"
```

Pika 的 Lighthouse 后端还需要 `libsurvive/pysurvive`。Pika SDK 通过 `pysurvive` 读取两个基站和 Tracker；缺少后端时，点击开始录制会显示明确错误，不会连接或移动机械臂。

先编辑 `config.yaml` 的 `demo`：

- `root` 是当前 demo 数据集或数据集集合路径；默认是 `~/.cache/huggingface/lerobot/mani_real`。
- 如果 `root` 本身是 LeRobot 数据集，新任务和新 episode 都追加到该数据集。
- 如果 `root` 是集合目录，采集器会递归查找同任务的 `meta/tasks.jsonl`；找到后追加 episode，找不到则创建 `<root>/<task_name>`。
- `pika_sense_port` 是 Pika Sense 串口，`tracker_device` 可固定 Tracker 名称。
- `gripper_distance_min_mm/max_mm`、`translation_scale` 和 `rotation_scale` 必须按当前 Pika Sense 标定。
- `calibration_path` 是 Sensor 到 UR TCP 的刚体外参文件；它与基站位置、Sensor 安装姿态和 TCP 定义绑定，不应提交到版本库。

首次真实回放前必须做一次多位姿手眼标定。默认采集 5 组姿态（最少 3 组）：先按顺序完整采集全部 UR TCP 参考位姿；随后再按**相同顺序**将 Sensor 放到对应的每个参考位置和朝向并采集。每组姿态都应有明显不同的平移和转动，不能只沿单轴平移。两阶段之间基站、机器人基座和参考物都不能移动，且必须保留参考位姿的编号或标记。执行：

```powershell
ur7e-vla calibrate-demo --config config.yaml
```

该命令会在终端逐步提示确认，只读取 Sensor 位姿和 UR TCP 位姿，求解 Lighthouse→UR 基座与 Sensor→TCP 两个外参，写入 `pika_sensor_to_ur_tcp.json`，不会下发机器人运动。残差超过配置阈值时不会保存。移动任一基站、改变 Sensor 安装方式或修改 UR TCP 后，必须重新标定。旧的单点标定文件不能用于真实回放。

不允许机械臂运动的界面预览：

```powershell
ur7e-vla collect-demo --config config.yaml --task "pick cube"
```

连接 UR7e、夹爪、D435i 和腕部相机后，显式启用真实回放：

```powershell
ur7e-vla collect-demo --config config.yaml --task "pick cube" --execute
```

界面流程如下：

1. 点击“开始录制 Sensor”，完成整条手持示教路径后点击“停止录制”。
2. 清空机械臂工作区，确认示教器处于 Remote Control，再点击“开始机械臂回放”。程序以标定外参将录制的首帧变为绝对 TCP 起点，先低速移动到该起点；到位后才逐帧做逆运动学并回放。
3. 回放期间，两路图像、实际关节/夹爪状态和实际下发 action 只写入 `demo_staging` 暂存区。
4. 回放完成后点击“保存 Episode”才编码并追加 LeRobot 数据；点击“舍弃并重录”不会修改 demo 数据集。

回放前会检查标定文件、Tracker 数据陈旧、平移/旋转工作区、逆解失败和相邻关节目标跳变。这里仍不包含碰撞规划；首次使用必须降低示教器速度滑块、缩小轨迹并保持急停可达。
