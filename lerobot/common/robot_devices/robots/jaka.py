"""
JAKA机器人LeRobot适配
仅支持推理模式（inference mode）
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
from lerobot.common.robot_devices.robots.configs import JakaRobotConfig


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


class JakaRobot:
    """
    JAKA机器人LeRobot接口
    专为推理模式设计（不包含遥操作功能）
    """
    def __init__(self, config: JakaRobotConfig | None = None, **kwargs):
        if config is None:
            config = JakaRobotConfig()
        
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
        
        # 推理模式不需要遥操作
        self.teleop = None
        
        # Force历史缓冲区（用于FADP推理）
        # FADP配置中force_obs_horizon=6（见fadp_force.yaml）
        self.n_obs_steps = 6
        self.force_buffer = deque(maxlen=self.n_obs_steps)
        
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
        
        # 移动到初始位置
        # self.run_calibration()

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
        这是推理模式的核心方法
        
        Returns:
            dict: 包含观测数据的字典 {state(7维), force(n_obs_steps, 6维), images}
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
        
        # 单位：位置(mm)，姿态(rad)，gripper(0)
        # 不做单位转换，保持JAKA原始单位mm
        
        # 读取力传感器数据（6维：fx,fy,fz,mx,my,mz）
        force_data_current = None
        if self.force_sensor:
            before_force_t = time.perf_counter()
            force_np = self.force_sensor.read()
            if force_np is not None:
                force_data_current = torch.from_numpy(force_np)
                # 添加到历史缓冲区
                self.force_buffer.append(force_data_current.clone())
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
        
        # 添加力传感器历史数据
        if force_data_current is not None and len(self.force_buffer) > 0:
            # 将force历史转换为(n_obs_steps, 6)的tensor
            # 如果缓冲区未满，用第一帧填充
            force_history = list(self.force_buffer)
            
            # 如果缓冲区还没有n_obs_steps帧，用第一帧重复填充
            while len(force_history) < self.n_obs_steps:
                force_history.insert(0, force_history[0].clone())
            
            # Stack成(n_obs_steps, 6)
            force_tensor = torch.stack(force_history, dim=0)
            obs_dict["observation.force"] = force_tensor  # (n_obs_steps, 6)
        
        # 添加图像
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        
        return obs_dict

    def send_action(self, action: torch.Tensor, reference_state: torch.Tensor) -> torch.Tensor:
        """
        发送动作到机器人（末端局部坐标系控制，使用linear_move）
        
        重要：
        - action是policy输出（m单位），立即转换为mm
        - reference_state来自capture_observation（mm单位）
        - 所有计算统一使用mm单位
        
        流程：
            1. action从m转为mm
            2. 使用mm单位进行坐标变换
            3. 发送mm单位的绝对位置
        
        Args:
            action: 模型预测的动作（7维tensor）
                    [x(m), y(m), z(m), rx(rad), ry(rad), rz(rad), gripper]
            reference_state: 参考帧位姿（7维tensor，mm单位）
                    [x(mm), y(mm), z(mm), rx(rad), ry(rad), rz(rad), gripper]
            
        Returns:
            torch.Tensor: 返回发送的动作
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "JAKA机器人未连接。需要先调用 `robot.connect()`。"
            )
        
        if reference_state is None:
            raise ValueError("reference_state不能为None！必须提供参考帧位姿。")
        
        # 1. 立即将action从m转为mm（统一单位）
        action_mm = action.clone()
        action_mm[0:3] = action_mm[0:3] * 1000.0  # m → mm
        # action[0]=-action[0]
        # action[1]=-action[1]
        
        # 2. 末端坐标系转换（右乘），所有计算用mm
        # 将参考帧和action转为变换矩阵（只用前6维：位置+姿态）
        T_world_reference = pose_to_matrix(reference_state[:6])  # mm,rad
        T_local_action = pose_to_matrix(action_mm[:6])           # mm,rad
        
        # 右乘：参考帧局部坐标系 -> 世界坐标系
        T_world_target = torch.matmul(T_world_reference, T_local_action)
        
        # 转回位姿表示
        target_pose_6d = matrix_to_pose(T_world_target)  # mm,rad
        
        # 3. 组合完整的7维目标（绝对位置，mm单位）
        target_absolute = torch.zeros(7, dtype=torch.float32)
        target_absolute[:6] = target_pose_6d
        target_absolute[6] = action_mm[6]
        
        # 🔍 调试输出
        # 获取当前force信息（如果有的话）
        force_info = ""
        if self.force_sensor and len(self.force_buffer) > 0:
            force_latest = self.force_buffer[-1].cpu().numpy()  # 最新一帧
            total_force = np.sqrt(force_latest[0]**2 + force_latest[1]**2 + force_latest[2]**2)
            force_info = f"\n  Force: [Fx={force_latest[0]:.2f}, Fy={force_latest[1]:.2f}, Fz={force_latest[2]:.2f}] Total={total_force:.2f}N"
        
        print(f"\n[DEBUG send_action]:")
        print(f"  参考帧 (mm,rad): [{reference_state[0]:.2f}, {reference_state[1]:.2f}, {reference_state[2]:.2f}, {reference_state[3]:.4f}, {reference_state[4]:.4f}, {reference_state[5]:.4f}]")
        print(f"  Action (mm,rad): [{action_mm[0]:.2f}, {action_mm[1]:.2f}, {action_mm[2]:.2f}, {action_mm[3]:.4f}, {action_mm[4]:.4f}, {action_mm[5]:.4f}]")
        print(f"  目标 (mm,rad): [{target_absolute[0]:.2f}, {target_absolute[1]:.2f}, {target_absolute[2]:.2f}, {target_absolute[3]:.4f}, {target_absolute[4]:.4f}, {target_absolute[5]:.4f}]{force_info}")
        
        # 4. 发送mm单位的绝对位置
        before_write_t = time.perf_counter()
        self.arm.write(target_absolute.tolist())
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        
        return action

    def teleop_step(self, record_data=False):
        """
        遥操作步骤（推理模式不需要，保留接口以兼容）
        """
        raise NotImplementedError(
            "JAKA机器人推理模式不支持遥操作。"
            "此配置仅用于运行训练好的策略模型。"
        )

    def teleop_safety_stop(self):
        """
        遥操作安全停止（推理模式不需要，保留接口以兼容）
        """
        if self.is_connected:
            self.run_calibration()

    def __del__(self):
        """析构函数：确保正确断开连接"""
        if self.is_connected:
            self.disconnect()
