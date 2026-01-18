# JAKA机器人LeRobot集成测试指南

## 📋 测试脚本功能

`test_jaka_integration.py` 是一个完整的集成测试脚本，用于验证：

1. ✅ JAKA机器人连接和控制
2. ✅ 力传感器（宇立）数据读取
3. ✅ 相机图像采集
4. ✅ Observation完整采集（7维state + 6维force + images）
5. ✅ 简单的机械臂控制测试
6. ✅ 实时数据可视化

## 🚀 快速开始

### 1. 准备工作

确保已安装必要的依赖：
```bash
# 安装PyForce（力传感器）
pip install git+https://github.com/Elycyx/PyForce.git

# 安装OpenCV（图像显示）
pip install opencv-python

# 安装PyTorch（如果还没安装）
pip install torch
```

### 2. 配置检查

在运行测试前，检查配置文件：

**编辑 `lerobot/common/robot_devices/robots/configs.py`**

找到 `JakaRobotConfig` 并根据实际情况修改：

```python
@RobotConfig.register_subclass("jaka")
@dataclass
class JakaRobotConfig(RobotConfig):
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": JakaMotorsBusConfig(
                robot_ip="192.168.2.64",  # ⚠️ 修改为实际JAKA机器人IP
                # ...
            ),
        }
    )
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                camera_index=0,  # ⚠️ 修改为实际相机编号
                # ...
            ),
        }
    )
    
    force_sensor: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "ip_addr": "192.168.0.108",  # ⚠️ 修改为实际力传感器IP
            "port": 4008
        }
    )
```

### 3. 运行测试

```bash
cd /home/hyx/xiazhi/jaka-Lerobot/lerobot_lab
python test_jaka_integration.py
```

## 📊 测试流程

### 阶段1: 设备连接测试
- 连接JAKA机器人
- 连接力传感器并校零
- 连接相机
- 移动到初始位置

### 阶段2: 静态采集测试
- 采集10帧observation
- 显示第1帧和第10帧的详细信息
- 验证数据格式和维度

### 阶段3: 简单控制测试
- X方向移动 +5mm
- Y方向移动 +5mm
- Z方向移动 +3mm
- 验证机械臂响应

### 阶段4: 实时循环测试
- 持续采集observation
- 实时显示相机图像
- 在图像上叠加传感器数据
- 支持键盘交互控制

## ⌨️ 键盘控制

运行中的键盘控制：

| 按键 | 功能 |
|------|------|
| `空格键` | 执行X方向+5mm测试移动 |
| `r` | 重置reference_state（episode开始参考位姿） |
| `q` | 退出测试程序 |

## 📈 数据显示说明

### 终端输出

```
============================================================
Frame #1
============================================================

📍 State (7D - 末端位姿):
   Shape: torch.Size([7])
   x      =   400.00 mm
   y      =     0.00 mm
   z      =   300.00 mm
   yaw    =   0.0000 rad (  0.00°)
   pitch  =   0.0000 rad (  0.00°)
   roll   =   3.1400 rad (180.00°)
   gripper=   0.0000

💪 Force (6D - 力/力矩):
   Shape: torch.Size([6])
   fx =    0.123 N
   fy =   -0.056 N
   fz =    2.345 N
   mx =    0.012 Nm
   my =   -0.008 Nm
   mz =    0.003 Nm
   总力 =    2.348 N

📷 Images (1个相机):
      primary: (480, 640, 3) (dtype: uint8)
```

### OpenCV窗口显示

- **图像窗口**：实时显示相机画面
- **叠加信息**：
  - 帧号
  - 末端位置 (x, y, z)
  - 总力大小

## ⚠️ 安全注意事项

1. **初始位置**：确保机器人初始位置安全，不会碰撞
2. **移动范围**：测试移动幅度很小（5mm），但仍需确保周围无障碍物
3. **急停准备**：随时准备按下机器人急停按钮
4. **监控状态**：注意观察终端输出的力传感器数据，异常大的力表示可能碰撞

## 🔧 故障排查

### 问题1: 无法连接JAKA机器人
```
ConnectionError: JAKA机器人连接失败
```
**解决方案**：
- 检查机器人IP地址配置
- 确保机器人已上电并进入就绪状态
- 检查网络连接

### 问题2: 力传感器连接失败
```
警告: 力传感器连接失败
```
**解决方案**：
- 检查力传感器IP地址和端口
- 确保PyForce已正确安装
- 检查传感器电源和网络

### 问题3: 相机无法打开
```
警告: 相机 primary 连接失败
```
**解决方案**：
- 检查camera_index配置
- 使用 `v4l2-ctl --list-devices` 查看可用相机
- 尝试不同的camera_index值

### 问题4: 导入错误
```
ImportError: No module named 'lerobot'
```
**解决方案**：
- 检查脚本中的路径设置
- 确保在正确的Python环境中运行

## 📝 预期输出示例

成功运行时的输出：

```
================================================================================
  JAKA机器人LeRobot集成测试
================================================================================
测试内容：
  1. 设备连接测试
  2. Observation采集测试
  3. 简单控制测试
  4. 实时数据显示

控制说明：
  空格键 - 执行测试移动
  'r'键  - 重置reference_state
  'q'键  - 退出测试
================================================================================

按Enter键开始测试...

================================================================================
  1. 初始化JAKA机器人
================================================================================
创建robot实例...
✓ Robot实例创建成功

机器人类型: jaka
推理模式: True
配置的相机数量: 1
是否启用力传感器: True

正在连接设备...
机器人登录成功
机器人上电成功
机器人使能成功
机器人已进入servo模式
JAKA机器人连接成功
力传感器已连接: 192.168.0.108:4008
力传感器已校零（使用100个样本）
力传感器连接成功并已校零
相机 primary 连接成功
所有设备连接完成
移动到初始末端位姿...
已到达初始位姿
机器人已移动到初始位置
✓ 所有设备连接成功
```

## 🎯 下一步

测试成功后，您可以：

1. **运行Policy推理**：使用训练好的模型控制机器人
2. **数据采集**：如果需要更多训练数据
3. **参数调优**：根据实际情况调整控制参数

## 📚 相关文档

- [JAKA_README.md](lerobot/common/robot_devices/robots/JAKA_README.md) - JAKA机器人详细说明
- [LeRobot数据采集.docx](../../LeRobot数据采集.docx) - 数据采集指南
- [PyForce文档](https://github.com/Elycyx/PyForce) - 力传感器使用说明

## 💡 提示

- 第一次运行建议在安全的开阔区域
- 建议先运行静态测试，确认数据采集正常
- 控制测试前确保机器人周围无障碍物
- 保持对机器人的视线观察
