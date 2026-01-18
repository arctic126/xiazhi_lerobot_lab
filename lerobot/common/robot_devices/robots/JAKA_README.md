# JAKA机器人LeRobot适配说明

本文档说明如何使用JAKA S5机器人与LeRobot进行推理部署。

## 概述

JAKA适配专为**推理模式**设计，允许训练好的LeRobot策略模型控制JAKA S5机器人执行任务。

### 特性

- ✅ 支持6轴JAKA S5机器人
- ✅ 推理模式（运行训练好的策略）
- ✅ 相机集成
- ❌ 不支持遥操作（数据采集需使用其他方式）
- ❌ 无夹爪支持

## 前置要求

### 硬件
1. JAKA S5机器人
2. USB相机（用于观测）
3. 网络连接到机器人控制器

### 软件
1. JAKA Python SDK (jkrc)
   ```bash
   # 将jkrc.so和libjakaAPI.so放在同一目录
   # Linux: 添加到LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=/path/to/jaka/sdk:$LD_LIBRARY_PATH
   ```

2. LeRobot环境
   ```bash
   cd lerobot
   pip install -e .
   ```

## 配置

### 1. 修改配置文件

编辑 `lerobot/common/robot_devices/robots/configs.py` 中的 `JakaRobotConfig`：

```python
@RobotConfig.register_subclass("jaka")
@dataclass
class JakaRobotConfig(RobotConfig):
    inference_time: bool = True
    
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": JakaMotorsBusConfig(
                robot_ip="192.168.2.64",  # 修改为你的机器人IP
                motors={
                    "joint_1": [1, "jaka_s5"],
                    "joint_2": [2, "jaka_s5"],
                    "joint_3": [3, "jaka_s5"],
                    "joint_4": [4, "jaka_s5"],
                    "joint_5": [5, "jaka_s5"],
                    "joint_6": [6, "jaka_s5"],
                },
            ),
        }
    )
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                camera_index=0,  # 修改为你的相机编号
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
```

### 2. 调整初始位置（可选）

编辑 `lerobot/common/robot_devices/motors/jaka.py`：

```python
class JakaMotorsBus:
    def __init__(self, config: JakaMotorsBusConfig):
        # ...
        # 修改初始位置（单位：弧度）
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

## 使用方法

### 基础测试

创建测试脚本 `test_jaka.py`：

```python
import jkrc
from lerobot.common.robot_devices.robots.utils import make_robot

# 方法1: 使用默认配置
robot = make_robot("jaka", inference_time=True)

# 方法2: 自定义配置
from lerobot.common.robot_devices.robots.configs import JakaRobotConfig
from lerobot.common.robot_devices.motors.configs import JakaMotorsBusConfig
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

config = JakaRobotConfig(
    inference_time=True,
    follower_arm={
        "main": JakaMotorsBusConfig(
            robot_ip="192.168.2.64",
            motors={
                "joint_1": [1, "jaka_s5"],
                "joint_2": [2, "jaka_s5"],
                "joint_3": [3, "jaka_s5"],
                "joint_4": [4, "jaka_s5"],
                "joint_5": [5, "jaka_s5"],
                "joint_6": [6, "jaka_s5"],
            },
        ),
    },
    cameras={
        "primary": OpenCVCameraConfig(
            camera_index=0,
            fps=30,
            width=640,
            height=480,
        ),
    },
)

robot = make_robot_from_config(config)

# 连接
robot.connect()

# 捕获观测
obs = robot.capture_observation()
print("关节状态:", obs["observation.state"])
print("图像形状:", obs["observation.images.primary"].shape)

# 断开
robot.disconnect()
```

### 运行策略推理

```python
import torch
from lerobot.common.robot_devices.robots.utils import make_robot

# 加载训练好的策略
policy = torch.load("path/to/your/policy.pth")
policy.eval()

# 创建机器人
robot = make_robot("jaka", inference_time=True)
robot.connect()

try:
    for step in range(100):
        # 捕获当前观测
        obs = robot.capture_observation()
        
        # 模型预测动作
        with torch.no_grad():
            action = policy(obs)
        
        # 发送动作到机器人
        robot.send_action(action)
        
        print(f"步骤 {step} 完成")
        
finally:
    robot.disconnect()
```

## 故障排查

### 连接问题

1. **无法连接机器人**
   - 检查机器人IP地址是否正确
   - 确认网络连接正常
   - 验证jkrc SDK是否正确安装

2. **SDK导入错误**
   ```bash
   # Linux
   export LD_LIBRARY_PATH=/path/to/jaka/sdk:$LD_LIBRARY_PATH
   
   # 验证
   python -c "import jkrc; print('SDK导入成功')"
   ```

### 运动问题

1. **机器人不移动**
   - 确认机器人已上电上使能
   - 检查错误码输出
   - 验证关节角度在有效范围内

2. **运动速度过快/过慢**
   - 调整 `jaka.py` 中的 `speed` 参数
   ```python
   ret = self.robot.joint_move(
       joint_pos=target_joint,
       move_mode=0,
       is_block=False,
       speed=1.0  # 调整此值
   )
   ```

### 相机问题

1. **相机无法连接**
   - 验证相机编号
   ```bash
   # 列出可用相机
   ls /dev/video*
   ```
   - 测试相机
   ```python
   import cv2
   cap = cv2.VideoCapture(0)  # 替换为你的相机编号
   ret, frame = cap.read()
   print("相机工作" if ret else "相机失败")
   ```

## 限制

1. **仅推理模式**: 此适配不支持遥操作数据采集
2. **无夹爪**: 当前实现不包括夹爪控制
3. **6轴机器人**: 仅支持6轴配置

## 扩展

### 添加夹爪支持

如果需要添加夹爪：

1. 修改 `JakaMotorsBusConfig`:
```python
motors={
    "joint_1": [1, "jaka_s5"],
    # ... 其他关节
    "joint_6": [6, "jaka_s5"],
    "gripper": [7, "jaka_gripper"],  # 添加夹爪
}
```

2. 更新 `read()` 和 `write()` 方法处理第7个自由度

### 添加多相机支持

```python
cameras: dict[str, CameraConfig] = field(
    default_factory=lambda: {
        "primary": OpenCVCameraConfig(
            camera_index=0,
            fps=30,
            width=640,
            height=480,
        ),
        "wrist": OpenCVCameraConfig(
            camera_index=2,
            fps=30,
            width=640,
            height=480,
        ),
    }
)
```

## 参考

- [JAKA SDK文档](https://www.jaka.com/docs/guide/SDK/Python.html)
- [LeRobot文档](https://github.com/huggingface/lerobot)
- LeRobot数据采集.docx（项目文档）

## 支持

如有问题，请检查：
1. JAKA SDK是否正确安装
2. 机器人网络连接
3. 相机设备可用性
4. LeRobot环境配置
