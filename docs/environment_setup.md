# 环境安装

服务端与机器人端使用独立 Python 环境：服务端运行 Sci-VLA/OpenPI 和 GPU 推理；机器人端只运行本项目、相机、Pika 与 `openpi-client`。

## pi0.5 服务端

服务端需要 Linux、NVIDIA GPU、Python 3.11 和 `uv`。在 Sci-VLA 的 OpenPI 目录执行：

```bash
cd /path/to/Sci-VLA/third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

若仓库不是递归克隆，先执行 `git submodule update --init --recursive`。确认 checkpoint 路径可读，并用 `nvidia-smi` 检查 GPU、显存和占用进程。

启动 `mani_real_pi05`：

```bash
cd /path/to/Sci-VLA/third_party/openpi
POLICY_DIR=/path/to/mani_real_pi05/checkpoint \
  /path/to/ur7e_inference/scripts/start_pi05_server.sh
```

服务默认监听 `0.0.0.0:8000`。确认防火墙允许机器人侧访问该端口。

## 机器人端

项目要求 Python 3.9 或更高。包含推理、录像和示教采集的安装：

```powershell
conda create -n ur7e_collector python=3.10 -y
conda activate ur7e_collector
python -m pip install --upgrade pip

cd C:\Users\15261\Documents\projects\Sci-VLA\third_party\openpi
python -m pip install -e packages\openpi-client

cd C:\Users\15261\Documents\projects\ur7e_inference
python -m pip install -e ".[demo]"
Copy-Item config.example.yaml config.yaml
```

只运行实时推理时可使用较小安装：

```bash
pip install -e .
pip install agx-pypika
cp config.example.yaml config.yaml
```

`.[demo]` 会安装 LeRobot 与 Pika SDK。Sensor Lighthouse 追踪还需要 SDK 的 `libsurvive/pysurvive` 后端。RealSense 驱动、D435i/D405 固件及 USB 权限需按设备厂商要求配置。

## 硬件与网络检查

```bash
ur7e-vla --help
ur7e-vla list-cameras
python scripts/probe_policy.py --host 192.168.124.15 --port 8000
```

## 启动交互式 VLA 推理

策略服务通过探测后，可在机器人端启动图形界面：

```powershell
ur7e-vla vla-gui --config config.yaml
```

此命令是 dry-run。只有在确认急停可达、工作空间清空后，才添加 `--execute`：

```powershell
ur7e-vla vla-gui --config config.yaml --execute
```

界面依赖 Python 自带的 Tk 图形组件；若启动时报 `No module named tkinter`，请为当前
Python/conda 环境安装或启用 Tk。MP4 录制使用项目的 OpenCV 依赖，并写入
`runtime.recording_dir`（默认 `recordings`）。

在 `config.yaml` 配置 Pika 串口、两路相机和 UR 地址。机器人主机必须能同时访问 UR7e 的 `169.254.175.10` 与推理主机的 `192.168.124.15`；通常需要两张网卡或正确的静态路由。
