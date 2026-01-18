"""
JAKA机器人LeRobot适配 - VLA版本
专为TA-VLA模型设计

两层参考系统：
1. Episode级别参考坐标系：episode开始时建立，保持整个episode不变
2. Chunk级别第一帧位置：每次推理时保存chunk第一帧在参考系下的位置

Action计算：
- action是相对于chunk第一帧的差值
- 目标位置（在参考系下）= chunk第一帧位置（在参考系下）+ action
- 然后变换到世界坐标系

与FADP版本(jaka.py)的关键区别：
1. Action单位：VLA输出是m单位，需要转换为mm
2. 两层参考：Episode参考坐标系 + Chunk第一帧位置
3. Force采样：20帧历史，从前60帧每3帧采样一个
"""
import time
import torch
import numpy as np
from collections import deque
from dataclasses import replace
from scipy.spatial.transform import Rotation as R

from lerobot.common.robot_devices.motors.utils import get_motor_names, make_motors_buses_from_configs
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.robot_devices.robots.configs import JakaRobotVLAConfig


def pose_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    """
    将位姿(x,y,z,rx,ry,rz)转换为4x4齐次变换矩阵
    
    Args:
        pose: (6,) tensor [x, y, z, rx, ry, rz]，欧拉角单位为弧度
    
    Returns:
        T: (4, 4) tensor 齐次变换矩阵
    """
    pose_np = pose.cpu().numpy()
    x, y, z, rx, ry, rz = pose_np
    
    # 从欧拉角创建旋转矩阵 (XYZ约定)
    rot = R.from_euler('xyz', [rx, ry, rz], degrees=False)
    rot_matrix = rot.as_matrix()
    
    # 构建齐次变换矩阵
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot_matrix
    T[:3, 3] = [x, y, z]
    
    return torch.from_numpy(T).to(pose.device)


def matrix_to_pose(T: torch.Tensor) -> torch.Tensor:
    """
    将4x4齐次变换矩阵转换为位姿(x,y,z,rx,ry,rz)
    
    Args:
        T: (4, 4) tensor 齐次变换矩阵
    
    Returns:
        pose: (6,) tensor [x, y, z, rx, ry, rz]，欧拉角单位为弧度
    """
    T_np = T.cpu().numpy()
    
    # 提取平移
    x, y, z = T_np[:3, 3]
    
    # 提取旋转并转换为欧拉角
    rot_matrix = T_np[:3, :3]
    rot = R.from_matrix(rot_matrix)
    rx, ry, rz = rot.as_euler('xyz', degrees=False)
    
    pose = np.array([x, y, z, rx, ry, rz], dtype=np.float32)
    return torch.from_numpy(pose).to(T.device)


class JakaRobotVLA:
    """
    JAKA机器人LeRobot接口 - VLA专用版本
    
    关键特性：
    - Action使用简单加法计算绝对位置（不使用矩阵变换）
    - Force历史：20帧，从前60帧每3帧采样
    - 仅用于推理模式
    """
    def __init__(self, config: JakaRobotVLAConfig | None = None, **kwargs):
        if config is None:
            config = JakaRobotVLAConfig()
        
        # 使用kwargs覆盖config参数
        self.config = replace(config, **kwargs)
        self.robot_type = self.config.type
        self.inference_time = self.config.inference_time
        
        # 构建相机
        self.cameras = make_cameras_from_configs(self.config.cameras)
        
        # 构建电机控制
        self.jaka_motors = make_motors_buses_from_configs(self.config.follower_arm)
        self.arm = self.jaka_motors['main']
        
        # 初始化力传感器
        self.force_sensor = None
        if self.config.force_sensor and self.config.force_sensor.get("enabled", False):
            try:
                from lerobot.common.robot_devices.sensors import ForceSensor
                self.force_sensor = ForceSensor(
                    ip_addr=self.config.force_sensor["ip_addr"],
                    port=self.config.force_sensor["port"]
                )
                print(f"力传感器已初始化: {self.config.force_sensor['ip_addr']}")
            except ImportError as e:
                print(f"警告: 无法导入ForceSensor: {e}")
                print("力传感器将被禁用")
                self.force_sensor = None
        
        # VLA特定：Force历史管理
        # 需要保存原始的60帧数据，然后每3帧采样得到20帧
        self.raw_force_buffer = deque(maxlen=60)  # 保存原始60帧
        self.vla_force_history_length = 20  # VLA需要20帧
        self.vla_force_sample_interval = 3  # 每3帧采样一次
        
        # VLA特定：两层参考系统
        # 1. Episode级别：参考坐标系（episode开始时建立）
        self.is_episode_first = True  # 是否是episode的第一帧
        self.T_world_ref = None   # 参考坐标系 → 世界坐标系的变换矩阵
        self.T_ref_world = None   # 世界坐标系 → 参考坐标系的变换矩阵
        
        # 2. Chunk级别：chunk第一帧在参考系下的位置
        self.chunk_first_in_ref = None  # chunk第一帧在参考坐标系下的位置 (6,)
        
        self.logs = {}
        self.is_connected = False

    @property
    def camera_features(self) -> dict:
        """返回相机特征配置"""
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        """返回电机特征配置（7维：x,y,z,yaw,pitch,roll,gripper）"""
        action_names = get_motor_names(self.jaka_motors)
        state_names = get_motor_names(self.jaka_motors)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }
    
    @property
    def force_features(self) -> dict:
        """返回力传感器特征配置（6维：fx,fy,fz,mx,my,mz）"""
        if self.force_sensor:
            return {
                "observation.force": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["fx", "fy", "fz", "mx", "my", "mz"],
                }
            }
        return {}
    
    @property
    def features(self) -> dict:
        """返回所有特征配置"""
        return {
            **self.camera_features,
            **self.motor_features,
            **self.force_features,
        }

    @property
    def has_camera(self):
        """是否配置了相机"""
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        """相机数量"""
        return len(self.cameras)

    def connect(self) -> None:
        """连接机器人、相机和力传感器"""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "JAKA机器人已经连接。不要重复调用 `robot.connect()`。"
            )
        
        # 连接机器人
        success = self.arm.connect(enable=True)
        if not success:
            raise ConnectionError("JAKA机器人连接失败")
        print("JAKA机器人连接成功")
        
        # 连接力传感器
        if self.force_sensor:
            if self.force_sensor.connect():
                # 力传感器校零
                self.force_sensor.zero(num_samples=100)
                print("力传感器连接成功并已校零")
            else:
                print("警告: 力传感器连接失败")
        
        # 连接相机
        for name in self.cameras:
            self.cameras[name].connect()
            if not self.cameras[name].is_connected:
                print(f"警告: 相机 {name} 连接失败")
            else:
                print(f"相机 {name} 连接成功")
        
        print("所有设备连接完成")
        self.is_connected = True

    def disconnect(self) -> None:
        """断开机器人、相机和力传感器连接"""
        if not self.is_connected:
            return
        
        # 断开机器人（会先移动到安全位置）
        self.arm.safe_disconnect()
        print("JAKA机器人已断开")
        
        # 断开力传感器
        if self.force_sensor:
            self.force_sensor.disconnect()
            print("力传感器已断开")
        
        # 断开相机
        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()
            print("相机已断开")
        
        self.is_connected = False

    def run_calibration(self):
        """移动到初始位置（校准）"""
        if not self.is_connected:
            raise ConnectionError("机器人未连接")
        
        self.arm.apply_calibration()
        print("机器人已移动到初始位置")

    def capture_observation(self) -> dict:
        """
        捕获当前观测数据（末端位姿 + 图像 + 力传感器历史）
        
        VLA特定：Force历史是20帧，从前60帧每3帧采样一个
        
        Returns:
            dict: 包含观测数据的字典 {state(7维), force(20, 6维), images}
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "JAKA机器人未连接。需要先调用 `robot.connect()`。"
            )
        
        # 读取当前末端位姿（7维：x,y,z,yaw,pitch,roll,gripper）
        before_read_t = time.perf_counter()
        state = self.arm.read()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        
        # 转换为tensor（保持mm单位）
        state_tensor = torch.as_tensor(list(state.values()), dtype=torch.float32)
        
        # 读取力传感器数据（6维：fx,fy,fz,mx,my,mz）
        force_data_current = None
        if self.force_sensor:
            before_force_t = time.perf_counter()
            force_np = self.force_sensor.read()
            if force_np is not None:
                force_data_current = torch.from_numpy(force_np)
                # 添加到原始缓冲区（60帧）
                self.raw_force_buffer.append(force_data_current.clone())
            self.logs["read_force_dt_s"] = time.perf_counter() - before_force_t
        
        # 读取相机图像
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t
        
        # 构造观测字典
        obs_dict = {
            "observation.state": state_tensor  # 7维
        }
        
        # VLA特定：添加力传感器历史数据（20帧，从前60帧每3帧采样）
        if force_data_current is not None and len(self.raw_force_buffer) > 0:
            # 从原始缓冲区中采样
            buffer_len = len(self.raw_force_buffer)
            
            if buffer_len >= 60:
                # 有足够的60帧数据：每3帧采样一次，共20帧
                # 采样索引：0, 3, 6, 9, ..., 57
                sampled_indices = list(range(0, 60, self.vla_force_sample_interval))
                force_history = [self.raw_force_buffer[i] for i in sampled_indices[:self.vla_force_history_length]]
            else:
                # 不足60帧：用现有数据填充
                # 先用第一帧重复填充到60帧，再采样
                first_frame = self.raw_force_buffer[0]
                padded_buffer = [first_frame.clone()] * (60 - buffer_len) + list(self.raw_force_buffer)
                sampled_indices = list(range(0, 60, self.vla_force_sample_interval))
                force_history = [padded_buffer[i] for i in sampled_indices[:self.vla_force_history_length]]
            
            # Stack成(20, 6)
            force_tensor = torch.stack(force_history, dim=0)
            obs_dict["observation.force"] = force_tensor  # (20, 6)
        
        # 添加图像
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        
        return obs_dict

    def send_action(self, action: torch.Tensor, reference_state: torch.Tensor = None) -> torch.Tensor:
        """
        发送动作到机器人（VLA版本：两层参考系统）
        
        VLA关键特性：
        - 单位转换：VLA输出是m单位，JAKA需要mm单位
        - 两层参考系统：
          1. Episode参考坐标系：整个episode固定
          2. Chunk第一帧位置：每次推理更新
        - 计算流程：
          1. 目标（在参考系下）= chunk第一帧位置 + action
          2. 变换到世界坐标系
        
        Args:
            action: 模型预测的动作（7维tensor，m单位！）
                    [dx(m), dy(m), dz(m), drx(rad), dry(rad), drz(rad), gripper]
                    注意：action是相对于chunk第一帧的差值，需要转换为mm
            reference_state: 当前位姿（7维tensor，mm单位）
                    [x(mm), y(mm), z(mm), rx(rad), ry(rad), rz(rad), gripper]
                    用于建立参考坐标系（仅episode第一帧时使用）
            
        Returns:
            torch.Tensor: 返回发送的动作
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "JAKA机器人未连接。需要先调用 `robot.connect()`。"
            )
        
        if reference_state is None:
            raise ValueError("reference_state不能为None！")
        
        # Episode第一帧：建立参考坐标系
        if self.is_episode_first:
            self.T_world_ref = pose_to_matrix(reference_state[:6])  # 参考坐标系 → 世界
            self.T_ref_world = torch.inverse(self.T_world_ref)       # 世界 → 参考坐标系
            self.is_episode_first = False
            
            print(f"\n[VLA 建立Episode参考坐标系]")
            print(f"  Episode第一帧位姿 (世界, mm): [{reference_state[0]:.2f}, {reference_state[1]:.2f}, {reference_state[2]:.2f}, "
                  f"{reference_state[3]:.4f}, {reference_state[4]:.4f}, {reference_state[5]:.4f}]")
        
        # 检查是否已设置chunk第一帧位置
        if self.chunk_first_in_ref is None:
            raise ValueError("chunk_first_in_ref未设置！请先调用save_chunk_first_frame()。")
        
        # 单位转换：m → mm（只转位置，姿态保持rad）
        action_mm = action.clone()
        action_mm[0:3] = action_mm[0:3] * 1000.0  # m → mm
        action_mm[0] = -action_mm[0]  # x取反
        action_mm[1] = -action_mm[1]  # y取反

        
        # 计算目标位置（在参考坐标系下）= chunk第一帧位置 + action
        target_in_ref = self.chunk_first_in_ref.clone()
        target_in_ref[:6] = self.chunk_first_in_ref[:6] + action_mm[:6]
        
        # 变换到世界坐标系
        T_ref_target = pose_to_matrix(target_in_ref)
        T_world_target = torch.matmul(self.T_world_ref, T_ref_target)
        target_world = matrix_to_pose(T_world_target)
        
        # 组合完整目标（包含gripper）
        target_absolute = torch.zeros(7, dtype=torch.float32)
        target_absolute[:6] = target_world
        target_absolute[6] = action_mm[6]  # gripper
        
        # 调试输出
        print(f"\n[VLA Action Transform - 两层参考系统]")
        print(f"  Chunk第一帧 (参考系, mm): [{self.chunk_first_in_ref[0]:.2f}, {self.chunk_first_in_ref[1]:.2f}, {self.chunk_first_in_ref[2]:.2f}]")
        print(f"  Action (m): [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
        print(f"  Action (mm): [{action_mm[0]:.2f}, {action_mm[1]:.2f}, {action_mm[2]:.2f}]")
        print(f"  目标位置 (参考系, mm): [{target_in_ref[0]:.2f}, {target_in_ref[1]:.2f}, {target_in_ref[2]:.2f}]")
        print(f"  目标位置 (世界, mm): [{target_absolute[0]:.2f}, {target_absolute[1]:.2f}, {target_absolute[2]:.2f}]")
        
        # 发送mm单位的绝对位置
        before_write_t = time.perf_counter()
        self.arm.write(target_absolute.tolist())
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        
        return action

    def teleop_step(self, record_data=False):
        """
        遥操作步骤（推理模式不需要，保留接口以兼容）
        """
        raise NotImplementedError(
            "JAKA机器人VLA推理模式不支持遥操作。"
            "此配置仅用于运行训练好的策略模型。"
        )

    def teleop_safety_stop(self):
        """
        遥操作安全停止（推理模式不需要，保留接口以兼容）
        """
        if self.is_connected:
            self.run_calibration()

    def reset_reference_frame(self):
        """
        重置Episode参考坐标系
        
        仅在episode开始时调用，建立统一的参考坐标系
        整个episode内保持不变
        """
        self.is_episode_first = True
        self.T_world_ref = None
        self.T_ref_world = None
        self.chunk_first_in_ref = None  # 同时清除chunk第一帧位置
        print("\n[VLA] 已重置Episode参考坐标系")
    
    def save_chunk_first_frame(self, state: torch.Tensor):
        """
        保存Chunk第一帧在参考坐标系下的位置
        
        每次推理（新chunk）开始时调用
        
        Args:
            state: 当前位姿（7维tensor，mm单位）
                   [x(mm), y(mm), z(mm), rx(rad), ry(rad), rz(rad), gripper]
        """
        if self.T_ref_world is None:
            # 如果还没有建立参考坐标系，先建立
            self.T_world_ref = pose_to_matrix(state[:6])
            self.T_ref_world = torch.inverse(self.T_world_ref)
            self.is_episode_first = False
            
            print(f"\n[VLA 建立Episode参考坐标系（首次推理）]")
            print(f"  Episode第一帧位姿 (世界, mm): [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]")
        
        # 将chunk第一帧变换到参考坐标系
        T_world_chunk_first = pose_to_matrix(state[:6])
        T_ref_chunk_first = torch.matmul(self.T_ref_world, T_world_chunk_first)
        self.chunk_first_in_ref = matrix_to_pose(T_ref_chunk_first)
        
        print(f"\n[VLA 保存Chunk第一帧位置]")
        print(f"  Chunk第一帧 (世界, mm): [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]")
        print(f"  Chunk第一帧 (参考系, mm): [{self.chunk_first_in_ref[0]:.2f}, {self.chunk_first_in_ref[1]:.2f}, {self.chunk_first_in_ref[2]:.2f}]")
    
    def __del__(self):
        """析构函数：确保正确断开连接"""
        if self.is_connected:
            self.disconnect()
