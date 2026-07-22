# Pika Sensor 数据采集

采集器生成 LeRobot v2.1 episode：两路 RGB 图像、6 维实际关节角、实际夹爪状态和 7 维绝对 action（6 轴目标关节角与夹爪目标）。任务文本写入 LeRobot task 元数据。

## 数据与夹爪约定

夹爪数据统一为 `0=张开、1=闭合`。实时采集同时启用 `demo.gripper_invert` 和 `gripper.invert`：手柄到实体夹爪的动作方向保持与旧采集流程一致，仅写入数据集的数值改为此约定。

```yaml
gripper:
  policy_min: 0.0
  policy_max: 1.0
  invert: true

demo:
  gripper_invert: true
```

不要将旧的 `0=闭合、1=张开` episode 与新 episode 混合训练。

## 配置

编辑 `config.yaml` 的 `demo`：

- `root`：现有 LeRobot 数据集或数据集集合目录。若根目录不是数据集，采集器会查找同任务的 `meta/tasks.jsonl` 并追加；找不到时创建 `<root>/<task_name>`。
- `pika_sense_port`：Pika Sense 串口；`tracker_device` 可指定 Tracker 名称。
- `gripper_distance_min_mm/max_mm`、`translation_scale`、`rotation_scale`：按当前设备标定。
- `sensor_to_tool_rpy`：Sensor 到工具坐标的固定旋转，默认值与 `RobotControl` 一致。
- `max_translation_m`、`max_rotation_rad`、`max_ik_joint_step_rad`、`max_gripper_step`：真机安全边界。

实时采集以开始时的 Sensor 位姿和 UR TCP 位姿建立相对锚点，不需要绝对 Lighthouse 到 UR 的手眼标定。`calibrate-demo` 仅保留给旧的离线轨迹回放工具。

## 运行

```bash
ur7e-vla collect-demo --config config.yaml --task "pick cube" --execute
```

GUI 操作：

1. 点击“开始遥控采集”，确认工作区已清空、急停可达且示教器为 Remote Control。
2. 移动 Pika Sensor 遥控 UR7e；开合 Sensor 同步控制实体夹爪，并按 `demo.fps` 暂存采集帧。
3. 点击“停止采集”停止伺服并生成暂存 episode。
4. 点击“保存 Episode”才写入数据集；“舍弃并重录”不会修改数据集。

采集会在 Tracker 数据陈旧、工作区越界、逆解失败、关节越限、目标跳变或相机异常时停止。它不包含碰撞规划；首次使用应缩小动作范围并保持急停可达。
