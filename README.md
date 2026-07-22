# UR7e + Pika + D435i 远程 pi0.5 推理

文档：

- [环境安装](docs/environment_setup.md)
- [Pika Sensor 数据采集](docs/demo_collection.md)
- [VLA 推理](docs/vla_inference.md)

## 快捷自查

```bash
# 检查命令行入口和本地相机编号
ur7e-vla --help
ur7e-vla list-cameras

# 检查策略服务连通性和 action 形状；不会连接机械臂
python scripts/probe_policy.py --host 192.168.124.15 --port 8000
```

```bash
# 推理 dry-run；不会向 UR7e 或夹爪下发命令
ur7e-vla run --config config.yaml --task "pick cube" --duration 30

# 实时 Sensor 遥控采集；会驱动真机
ur7e-vla collect-demo --config config.yaml --task "pick cube" --execute
```

只有传入 `--execute` 才会向 UR7e 和夹爪下发命令。真机前确认急停可达、工作区无障碍物，并先完成 dry-run。
