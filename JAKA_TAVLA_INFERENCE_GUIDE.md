# JAKA机器人 + TA-VLA推理指南

本指南介绍如何使用训练好的TA-VLA模型控制JAKA机器人。

## 📋 目录

- [系统架构](#系统架构)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [详细说明](#详细说明)
- [故障排除](#故障排除)

---

## 系统架构

TA-VLA使用**客户端-服务器架构**，与FADP不同：

```
┌─────────────────┐          WebSocket          ┌──────────────────┐
│   GPU服务器     │ ◄─────────────────────────► │  机器人控制机    │
│                 │                              │                  │
│  serve_policy   │                              │ eval_jaka_tavla  │
│  (加载模型)     │                              │ (控制机器人)     │
└─────────────────┘                              └──────────────────┘
```

### 两个独立进程

1. **服务器端** (`serve_policy.py`)
   - 加载训练好的TA-VLA模型
   - 监听WebSocket连接
   - 接收observation，返回action
   - 可以运行在GPU服务器上

2. **客户端** (`eval_jaka_tavla.py`)
   - 连接到TA-VLA服务器
   - 控制JAKA机器人
   - 采集observation发送给服务器
   - 接收action并执行

---

## 环境准备

### 1. TA-VLA环境（服务器端）

```bash
# 进入TA-VLA目录
cd /home/hyx/xiazhi/TA-VLA

# 确保已安装依赖
uv sync

# 验证openpi-client可用
uv run python -c "from openpi_client import websocket_client_policy; print('OK')"
```

### 2. LeRobot环境（客户端）

```bash
# 进入lerobot目录
cd /home/hyx/xiazhi/jaka-Lerobot/lerobot_lab

# 确保已有openpi-client
pip install openpi-client  # 如果未安装

# 或使用TA-VLA的包
export PYTHONPATH=/home/hyx/xiazhi/TA-VLA:$PYTHONPATH
```

### 3. 硬件准备

- JAKA机器人（IP: 192.168.2.64）
- 宇立力传感器（IP: 192.168.0.108）
- USB相机（索引: 0）
- 网络连接（服务器和客户端需要能互相访问）

---

## 快速开始

### 步骤1：启动TA-VLA服务器

在GPU服务器或本地机器上：

```bash
cd /home/hyx/xiazhi/TA-VLA

# 启动服务器
uv run scripts/serve_policy.py \
    --checkpoint=/path/to/your/model.ckpt \
    --host=0.0.0.0 \
    --port=8000
```

**参数说明：**
- `--checkpoint`: 模型checkpoint路径
- `--host`: 监听地址（0.0.0.0表示监听所有网络接口）
- `--port`: 端口号（默认8000）

### 步骤2：运行推理脚本

在机器人控制机上：

```bash
cd /home/hyx/xiazhi/jaka-Lerobot/lerobot_lab

# 运行推理
python eval_jaka_tavla.py \
    --server_host localhost \
    --server_port 8000 \
    --robot_ip 192.168.2.64 \
    --force_ip 192.168.0.108 \
    --camera_index 0 \
    --task_prompt "clean the toilet" \
    --max_steps 500 \
    --frequency 10
```

**参数说明：**
- `--server_host`: TA-VLA服务器地址
  - 本地: `localhost` 或 `127.0.0.1`
  - 远程: GPU服务器的IP地址
- `--server_port`: 服务器端口（与serve_policy一致）
- `--robot_ip`: JAKA机器人IP
- `--force_ip`: 力传感器IP
- `--camera_index`: 相机索引
- `--task_prompt`: 任务描述（会影响模型行为）
- `--max_steps`: 最大步数
- `--frequency`: 控制频率（Hz）

---

## 详细说明

### 数据流程

```
1. 机器人采集observation
   └─ images: (640, 480, 3)
   └─ state: (7,) [x,y,z,rx,ry,rz,gripper] mm,rad
   └─ force: (8, 6) - 8帧历史

2. 转换为TA-VLA格式
   └─ images: {"images": (224, 224, 3) uint8}
   └─ state: (7,) float32 → 自动填充到32维
   └─ effort: (20, 6) float32 - 20帧历史
   └─ prompt: "task description"

3. 通过WebSocket发送到服务器
   └─ 服务器运行推理
   └─ 返回action: (1, 32)

4. 提取前7维action
   └─ [x,y,z,rx,ry,rz,gripper] m,rad

5. 发送到机器人执行
```

### 坐标系说明

- **输入state**: mm单位（JAKA原始单位）
  - 位置: [x_mm, y_mm, z_mm]
  - 姿态: [rx_rad, ry_rad, rz_rad]

- **输出action**: m单位（模型训练时的单位）
  - 位置: [x_m, y_m, z_m]
  - 姿态: [rx_rad, ry_rad, rz_rad]

- **自动转换**: 脚本会在`send_action`中自动将m转为mm

### 力传感器历史

TA-VLA使用20帧力传感器历史：

```python
# 配置（在模型训练时确定）
effort_history_length = 20
effort_dim = 6  # [fx, fy, fz, mx, my, mz]

# 实际采样（示例）
# 从-60帧到0帧，间隔3帧采样20个点
indices = [0, -3, -6, -9, ..., -57]
```

---

## 故障排除

### 1. 服务器连接失败

**错误**: `❌ 服务器连接失败: Connection refused`

**解决**:
```bash
# 检查服务器是否运行
ps aux | grep serve_policy

# 检查端口是否监听
netstat -tlnp | grep 8000

# 检查防火墙
sudo ufw allow 8000

# 检查网络连接
ping <server_host>
telnet <server_host> 8000
```

### 2. 图像格式错误

**错误**: `Image dtype/shape mismatch`

**原因**: TA-VLA要求(224, 224, 3) uint8格式

**解决**: 脚本已自动处理，检查相机是否正常工作

### 3. Action维度错误

**错误**: `Expected 7 dims, got 32`

**原因**: TA-VLA输出32维，只用前7维

**解决**: 脚本已自动提取前7维

### 4. 力传感器数据缺失

**错误**: `Missing force data`

**原因**: 力传感器未连接或未启用

**解决**:
```python
# 如果没有力传感器，脚本会自动用零填充
# 在jaka_config中设置：
force_sensor={"enabled": False}
```

### 5. 推理速度慢

**现象**: 频率低于目标10Hz

**可能原因**:
- 网络延迟（客户端-服务器通信）
- GPU推理慢
- 图像传输慢

**优化**:
```bash
# 1. 使用更快的GPU
# 2. 减小图像大小（但影响效果）
# 3. 使用本地推理（不用WebSocket）
# 4. 降低目标频率
python eval_jaka_tavla.py --frequency 5
```

---

## 高级配置

### 使用不同任务提示

```bash
python eval_jaka_tavla.py \
    --task_prompt "pick up the red cup"
```

### 远程GPU服务器

```bash
# 服务器端（GPU机器）
uv run scripts/serve_policy.py \
    --checkpoint=model.ckpt \
    --host=0.0.0.0 \
    --port=8000

# 客户端（机器人控制机）
python eval_jaka_tavla.py \
    --server_host 192.168.1.100 \
    --server_port 8000
```

### 调试模式

在脚本中添加详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 与FADP的对比

| 特性 | FADP | TA-VLA |
|------|------|--------|
| 框架 | PyTorch | JAX/Flax |
| 架构 | 单进程 | 客户端-服务器 |
| 推理方式 | 直接加载模型 | WebSocket通信 |
| 输入图像 | 可变 | 224x224固定 |
| 状态维度 | 7维 | 自动填充到32维 |
| 输出维度 | 7维 | 32维（用前7维） |
| 力传感器历史 | 8帧 | 20帧 |
| GPU要求 | CUDA | JAX (CPU/GPU) |

---

## 参考资料

- TA-VLA仓库: `/home/hyx/xiazhi/TA-VLA`
- 训练脚本: `TA-VLA/scripts/train.py`
- 测试脚本: `TA-VLA/scripts/test_favla_policy.py`
- LeRobot适配: `jaka-Lerobot/lerobot_lab/lerobot/common/robot_devices/robots/jaka.py`

---

## 许可证

本项目遵循相关开源协议。
