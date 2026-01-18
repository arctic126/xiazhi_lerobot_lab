#!/usr/bin/env python3
"""
JAKA机器人 + FADP模型推理脚本（仅相机版本）

使用训练好的Force-Aware Diffusion Policy模型控制JAKA机器人
此版本只使用相机，不使用力传感器，用于验证运动是否正确

使用方法:
python eval_jaka_fadp_camera_only.py \
    --checkpoint /home/sjtu/xiazhi/model/ckpt_1215/dp_noup/19.34.34_train_diffusion_unet_timm_fadp_umi_fadp_umi/checkpoints/latest.ckpt \
    --robot_ip 192.168.31.112 \
    --camera_index 0 \
    --max_steps 500 \
    --frequency 10

控制说明:
    - 按'q'键: 退出程序
    - 按's'键: 停止当前episode
"""

import sys
import time
import argparse
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import dill
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as R

# 添加lerobot路径
sys.path.insert(0, str(Path(__file__).parent))

# 添加FADP路径 - 重要！
fadp_path = Path(__file__).parent.parent / "Force-Aware-Diffusion-Policy-main"
if fadp_path.exists():
    sys.path.insert(0, str(fadp_path))
    print(f"已添加FADP路径: {fadp_path}")
else:
    print(f"警告: FADP路径不存在: {fadp_path}")
    print("请确保Force-Aware-Diffusion-Policy-main目录存在")

from lerobot.common.robot_devices.robots.utils import make_robot_from_config


def pose_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    """
    将位姿(x,y,z,rx,ry,rz)转换为4x4齐次变换矩阵
    
    Args:
        pose: (6,) tensor [x, y, z, rx, ry, rz]，欧拉角单位为弧度，位置单位为mm
    
    Returns:
        T: (4, 4) tensor 齐次变换矩阵
    """
    if isinstance(pose, torch.Tensor):
        pose_np = pose.cpu().numpy()
    else:
        pose_np = np.array(pose)
    x, y, z, rx, ry, rz = pose_np[:6]
    
    # 从欧拉角创建旋转矩阵 (XYZ约定)
    rot = R.from_euler('xyz', [rx, ry, rz], degrees=False)
    rot_matrix = rot.as_matrix()
    
    # 构建齐次变换矩阵
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot_matrix
    T[:3, 3] = [x, y, z]
    
    return torch.from_numpy(T)


def matrix_to_pose(T: torch.Tensor) -> torch.Tensor:
    """
    将4x4齐次变换矩阵转换为位姿(x,y,z,rx,ry,rz)
    
    Args:
        T: (4, 4) tensor 齐次变换矩阵
    
    Returns:
        pose: (6,) tensor [x, y, z, rx, ry, rz]，欧拉角单位为弧度
    """
    if isinstance(T, torch.Tensor):
        T_np = T.cpu().numpy()
    else:
        T_np = np.array(T)
    
    # 提取平移
    x, y, z = T_np[:3, 3]
    
    # 提取旋转并转换为欧拉角
    rot_matrix = T_np[:3, :3]
    rot = R.from_matrix(rot_matrix)
    rx, ry, rz = rot.as_euler('xyz', degrees=False)
    
    pose = np.array([x, y, z, rx, ry, rz], dtype=np.float32)
    return torch.from_numpy(pose)


# 全局变量：用于保存第一帧参考坐标系
_reference_frame = {
    'T_world_ref': None,  # 第一帧坐标系 → 世界坐标系的变换矩阵
    'T_ref_world': None,  # 世界坐标系 → 第一帧坐标系的变换矩阵
    'initialized': False
}


def reset_reference_frame():
    """重置参考坐标系（每个episode开始时调用）"""
    global _reference_frame
    _reference_frame['T_world_ref'] = None
    _reference_frame['T_ref_world'] = None
    _reference_frame['initialized'] = False
    print("[FADP] 已重置参考坐标系")


def euler_to_rotation_6d(rx, ry, rz):
    """
    将欧拉角转换为rotation_6d表示
    
    rotation_6d是旋转矩阵的前两列展平
    
    Args:
        rx, ry, rz: 欧拉角（弧度，XYZ约定）
        
    Returns:
        rotation_6d: (6,) numpy array
    """
    rot = R.from_euler('xyz', [rx, ry, rz], degrees=False)
    rot_matrix = rot.as_matrix()  # 3x3
    # rotation_6d: 取前两列并展平
    rotation_6d = rot_matrix[:, :2].T.flatten()  # 6维
    return rotation_6d.astype(np.float32)


def get_relative_pose_rotation6d(current_state: torch.Tensor) -> torch.Tensor:
    """
    计算当前位姿相对于第一帧的位姿（使用rotation_6d格式）
    
    Args:
        current_state: (7,) tensor [x, y, z, rx, ry, rz, gripper]，mm单位
        
    Returns:
        relative_pose: (9,) tensor [pos(3) + rot_6d(6)]，位置为m单位
    """
    global _reference_frame
    
    # 第一帧：初始化参考坐标系
    if not _reference_frame['initialized']:
        _reference_frame['T_world_ref'] = pose_to_matrix(current_state[:6])  # 第一帧 → 世界
        _reference_frame['T_ref_world'] = torch.inverse(_reference_frame['T_world_ref'])  # 世界 → 第一帧
        _reference_frame['initialized'] = True
        
        print(f"[FADP] 建立参考坐标系")
        print(f"  第一帧位姿 (世界, mm): [{current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f}]")
    
    # 计算相对位姿
    T_world_current = pose_to_matrix(current_state[:6])  # 当前位姿的变换矩阵
    T_ref_current = torch.matmul(_reference_frame['T_ref_world'], T_world_current)  # 变换到第一帧坐标系
    
    # 提取位置和旋转
    T_ref_current_np = T_ref_current.cpu().numpy()
    pos = T_ref_current_np[:3, 3]  # 位置
    rot_matrix = T_ref_current_np[:3, :3]  # 旋转矩阵
    
    # 转换为rotation_6d
    rotation_6d = rot_matrix[:, :2].T.flatten()  # 6维
    
    # 单位转换：mm → m
    pos = pos / 1000.0
    
    # 组合为(9,)格式: [pos(3) + rot_6d(6)]
    relative_pose = np.concatenate([pos, rotation_6d]).astype(np.float32)
    
    return torch.from_numpy(relative_pose)


def get_relative_pose(current_state: torch.Tensor) -> torch.Tensor:
    """
    计算当前位姿相对于第一帧的位姿
    
    Args:
        current_state: (7,) tensor [x, y, z, rx, ry, rz, gripper]，mm单位
        
    Returns:
        relative_pose: (6,) tensor 相对于第一帧的位姿，m单位
    """
    global _reference_frame
    
    # 第一帧：初始化参考坐标系
    if not _reference_frame['initialized']:
        _reference_frame['T_world_ref'] = pose_to_matrix(current_state[:6])  # 第一帧 → 世界
        _reference_frame['T_ref_world'] = torch.inverse(_reference_frame['T_world_ref'])  # 世界 → 第一帧
        _reference_frame['initialized'] = True
        
        print(f"[FADP] 建立参考坐标系")
        print(f"  第一帧位姿 (世界, mm): [{current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f}]")
    
    # 计算相对位姿
    T_world_current = pose_to_matrix(current_state[:6])  # 当前位姿的变换矩阵
    T_ref_current = torch.matmul(_reference_frame['T_ref_world'], T_world_current)  # 变换到第一帧坐标系
    relative_pose = matrix_to_pose(T_ref_current)  # 当前位置在第一帧坐标系中的表示
    
    # 单位转换：mm → m
    relative_pose[:3] = relative_pose[:3] / 1000.0
    
    return relative_pose


def load_fadp_model(checkpoint_path: str, device: str = 'cuda'):
    """
    加载FADP模型checkpoint
    
    Args:
        checkpoint_path: checkpoint文件路径 (.ckpt)
        device: 设备 ('cuda' or 'cpu')
        
    Returns:
        policy: FADP policy模型
        cfg: 模型配置
    """
    print(f"加载FADP模型: {checkpoint_path}")
    
    # 加载checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    print(f"模型类型: {cfg.name}")
    
    # 安全地访问配置字段
    if hasattr(cfg, 'n_obs_steps'):
        print(f"n_obs_steps: {cfg.n_obs_steps}")
    if hasattr(cfg, 'n_action_steps'):
        print(f"n_action_steps: {cfg.n_action_steps}")
    if hasattr(cfg, 'horizon'):
        print(f"Horizon: {cfg.horizon}")
    
    # 创建workspace并加载模型
    import hydra
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 获取policy
    policy = workspace.model
    if hasattr(cfg, 'training') and hasattr(cfg.training, 'use_ema') and cfg.training.use_ema:
        policy = workspace.ema_model
        print("使用EMA模型")
    
    # 设置为评估模式
    policy.eval().to(device)
    
    # 设置推理参数
    if hasattr(policy, 'num_inference_steps'):
        policy.num_inference_steps = 16  # DDIM推理步数
    
    print(f"✓ 模型加载成功")
    
    # 安全地打印shape信息
    if hasattr(cfg, 'task') and hasattr(cfg.task, 'shape_meta'):
        if 'action' in cfg.task.shape_meta:
            print(f"  Action shape: {cfg.task.shape_meta['action']['shape']}")
        if 'obs' in cfg.task.shape_meta:
            if 'camera_0' in cfg.task.shape_meta['obs']:
                print(f"  Image shape: {cfg.task.shape_meta['obs']['camera_0']['shape']}")
    
    return policy, cfg


def prepare_observation_for_policy(obs_dict, obs_history, cfg, device):
    """
    准备observation送入policy（仅相机版本）
    
    FADP需要：
    - camera0_rgb: (B, img_obs_horizon, C, H, W) - 默认2帧
    - robot0_eef_pos: (B, obs_horizon, 6) - 相对于第一帧的末端位姿
    
    注意：此版本不使用力传感器数据
    
    Args:
        obs_dict: 当前observation字典（来自robot.capture_observation()）
        obs_history: observation历史deque
        cfg: 模型配置
        device: torch device
        
    Returns:
        policy_obs: 准备好的observation字典
    """
    # 从shape_meta获取真实的horizon配置
    shape_meta = cfg.task.shape_meta
    
    # 获取各种输入的horizon
    img_obs_horizon = 2  # 默认值
    eef_obs_horizon = 2  # 默认值
    
    if 'obs' in shape_meta:
        # 查找camera key的horizon
        for key in shape_meta['obs']:
            if 'camera' in key.lower() or 'rgb' in key.lower():
                img_obs_horizon = shape_meta['obs'][key].get('horizon', 2)
            if 'eef' in key.lower() or 'pos' in key.lower():
                eef_obs_horizon = shape_meta['obs'][key].get('horizon', 2)
    
    # 添加当前observation到历史
    obs_history.append(obs_dict)
    
    # 构建policy observation
    policy_obs = {}
    
    # 1. 图像历史 (img_obs_horizon, C, H, W)
    image_key = 'observation.images.primary'  # 使用primary相机
    if image_key in obs_dict:
        # 收集历史图像
        image_history = []
        for obs in list(obs_history):
            img = obs[image_key]  # (H, W, C)
            
            # 检查数据类型并转换
            if isinstance(img, torch.Tensor):
                # 如果已经是Tensor，转为numpy
                img = img.cpu().numpy()
            
            # 转换为 (C, H, W) 并归一化
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            # Resize到训练时的尺寸 (3, 224, 224)
            img = TF.resize(img, [224, 224], antialias=True)
            image_history.append(img)
        
        # Stack成(img_obs_horizon, C, H, W)
        if len(image_history) < img_obs_horizon:
            # 如果不足，用第一帧填充
            while len(image_history) < img_obs_horizon:
                image_history.insert(0, image_history[0].clone())
        
        image_tensor = torch.stack(image_history[-img_obs_horizon:], dim=0)
        policy_obs['camera0_rgb'] = image_tensor.unsqueeze(0).to(device)  # (1, img_obs_horizon, C, H, W)
    
    # 2. 机器人末端位姿 - 分别提供位置和旋转
    # 模型期望: robot0_eef_pos (3维) + robot0_eef_rot_axis_angle (3或6维)
    state_key = 'observation.state'
    if state_key in obs_dict:
        # 从shape_meta获取实际需要的格式
        pos_dim = 3  # 默认: 位置3维
        rot_dim = 3  # 默认: 旋转3维 (axis-angle)
        
        if 'obs' in shape_meta:
            for key, attr in shape_meta['obs'].items():
                if 'eef_pos' in key.lower() and 'shape' in attr:
                    pos_dim = attr['shape'][0]
                    print(f"[DEBUG] robot0_eef_pos 维度: {pos_dim}")
                if 'rot' in key.lower() and 'shape' in attr:
                    rot_dim = attr['shape'][0]
                    print(f"[DEBUG] robot0_eef_rot 维度: {rot_dim}")
        
        # 收集历史位姿
        pos_history = []
        rot_history = []
        
        for obs in list(obs_history):
            state = obs[state_key]  # (7,) - x,y,z,rx,ry,rz,gripper (mm单位)
            
            # 获取相对位姿（包含位置和旋转）
            relative_full = get_relative_pose_rotation6d(state)  # (9,) = [pos(3) + rot_6d(6)]
            
            # 位置 (3维, m单位)
            pos = relative_full[:3]
            
            # 旋转
            if rot_dim == 6:
                # rotation_6d格式
                rot = relative_full[3:9]  # (6,)
            elif rot_dim == 3:
                # axis-angle格式 - 从rotation_6d转换
                rot_6d = relative_full[3:9].numpy()
                rot_matrix = np.zeros((3, 3), dtype=np.float32)
                rot_matrix[:, 0] = rot_6d[:3]  # 第一列
                rot_matrix[:, 1] = rot_6d[3:6]  # 第二列
                # 计算第三列 (叉积)
                rot_matrix[:, 2] = np.cross(rot_matrix[:, 0], rot_matrix[:, 1])
                # 转换为axis-angle
                r = R.from_matrix(rot_matrix)
                axis_angle = r.as_rotvec()  # (3,)
                rot = torch.from_numpy(axis_angle.astype(np.float32))
            else:
                # 其他维度，使用rotation_6d
                rot = relative_full[3:9]
            
            pos_history.append(pos)
            rot_history.append(rot)
        
        # 填充不足的历史帧
        while len(pos_history) < eef_obs_horizon:
            pos_history.insert(0, pos_history[0].clone())
            rot_history.insert(0, rot_history[0].clone())
        
        # Stack
        pos_tensor = torch.stack(pos_history[-eef_obs_horizon:], dim=0)
        rot_tensor = torch.stack(rot_history[-eef_obs_horizon:], dim=0)
        
        policy_obs['robot0_eef_pos'] = pos_tensor.unsqueeze(0).to(device)  # (1, T, 3)
        policy_obs['robot0_eef_rot_axis_angle'] = rot_tensor.unsqueeze(0).to(device)  # (1, T, rot_dim)
        
        # 3. 夹爪宽度 robot0_gripper_width (1维)
        gripper_history = []
        for obs in list(obs_history):
            state = obs[state_key]  # (7,) - x,y,z,rx,ry,rz,gripper
            gripper = state[6:7]  # (1,) - 保持维度，取gripper值
            gripper_history.append(gripper)
        
        # 填充不足的历史帧
        while len(gripper_history) < eef_obs_horizon:
            gripper_history.insert(0, gripper_history[0].clone())
        
        gripper_tensor = torch.stack(gripper_history[-eef_obs_horizon:], dim=0)
        policy_obs['robot0_gripper_width'] = gripper_tensor.unsqueeze(0).to(device)  # (1, T, 1)
    
    return policy_obs


def visualize_observation(obs_dict, step_count, action_info=None):
    """
    可视化当前observation
    
    Args:
        obs_dict: observation字典
        step_count: 步数
        action_info: action信息（可选）
    
    Returns:
        按键码
    """
    # 获取图像
    image_key = 'observation.images.primary'
    if image_key not in obs_dict:
        return -1
    
    img = obs_dict[image_key]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    # 转换为BGR用于OpenCV显示
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    
    vis_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 添加信息叠加
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (5, 5), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
    
    # 添加文本
    y_offset = 25
    cv2.putText(vis_img, f"Step: {step_count} (Camera Only)", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 添加state信息
    if 'observation.state' in obs_dict:
        state = obs_dict['observation.state']
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        y_offset += 25
        cv2.putText(vis_img, f"Pos: ({state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f})",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 添加action信息
    if action_info is not None:
        y_offset += 25
        cv2.putText(vis_img, f"Action steps: {action_info['n_steps']}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # 显示
    cv2.imshow('JAKA + FADP Inference (Camera Only)', vis_img)
    key = cv2.waitKey(1) & 0xFF
    
    return key


def main():
    parser = argparse.ArgumentParser(description='JAKA + FADP推理脚本（仅相机版本）')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='FADP模型checkpoint路径 (.ckpt)')
    parser.add_argument('--robot_ip', '-r', type=str, default='192.168.2.64',
                        help='JAKA机器人IP地址')
    parser.add_argument('--camera_index', '-cam', type=int, default=0,
                        help='相机索引')
    parser.add_argument('--max_steps', '-m', type=int, default=500,
                        help='每个episode最大步数')
    parser.add_argument('--frequency', '-hz', type=float, default=10.0,
                        help='控制频率 (Hz)')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='设备 (cuda or cpu)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("JAKA机器人 + FADP模型推理（仅相机版本）")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Robot IP: {args.robot_ip}")
    print(f"Camera: {args.camera_index}")
    print(f"Control frequency: {args.frequency} Hz")
    print("注意: 此版本不使用力传感器，force数据将用零填充")
    print("="*80)
    
    # 1. 检查CUDA
    print("\n检查CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA可用")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        device = torch.device(args.device)
    else:
        print("⚠ CUDA不可用，使用CPU")
        device = torch.device('cpu')
    
    print(f"\n使用设备: {device}")
    
    # 2. 加载FADP模型
    policy, cfg = load_fadp_model(args.checkpoint, device=device)
    
    # 3. 初始化JAKA机器人（不使用力传感器）
    print("\n初始化JAKA机器人（仅相机）...")
    
    # 导入配置类
    from lerobot.common.robot_devices.robots.configs import JakaRobotConfig
    from lerobot.common.robot_devices.motors.configs import JakaMotorsBusConfig
    from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
    
    # 创建JAKA配置（不包含力传感器）
    jaka_config = JakaRobotConfig(
        inference_time=True,
        follower_arm={
            "main": JakaMotorsBusConfig(
                robot_ip=args.robot_ip,
                end_effector_dof={
                    "x": [0, "jaka_s5"],
                    "y": [1, "jaka_s5"],
                    "z": [2, "jaka_s5"],
                    "rx": [3, "jaka_s5"],
                    "ry": [4, "jaka_s5"],
                    "rz": [5, "jaka_s5"],
                    "gripper": [6, "jaka_s5"],
                },
            ),
        },
        cameras={
            "primary": OpenCVCameraConfig(
                camera_index=args.camera_index,
                fps=30,
                width=640,
                height=480,
            ),
        },
        force_sensor={
            "enabled": False,  # 禁用力传感器
        },
        mock=False
    )
    
    # 创建robot实例
    robot = make_robot_from_config(jaka_config)
    
    # 连接机器人
    print("\n连接设备...")
    robot.connect()
    
    # 等待设备稳定
    time.sleep(2)
    
    print("\n✓ 所有设备已连接")
    print("\n控制说明:")
    print("  - 按'q'键: 退出程序")
    print("  - 按's'键: 停止当前episode")
    print("\n准备开始推理...")
    time.sleep(1)
    
    # 4. 推理循环
    try:
        dt = 1.0 / args.frequency
        
        # 预热policy
        print("\n预热policy...")
        
        # 确定obs_history的maxlen（只需要camera的horizon）
        shape_meta = cfg.task.shape_meta
        max_obs_horizon = 2  # 默认值（camera的horizon）
        
        if 'obs' in shape_meta:
            for key in shape_meta['obs']:
                if 'camera' in key.lower() or 'rgb' in key.lower():
                    max_obs_horizon = shape_meta['obs'][key].get('horizon', 2)
                    break
        
        print(f"  Observation history buffer size: {max_obs_horizon}")
        
        dummy_obs = robot.capture_observation()
        obs_history = deque(maxlen=max_obs_horizon)
        policy_obs = prepare_observation_for_policy(dummy_obs, obs_history, cfg, device)
        
        with torch.no_grad():
            policy.reset()
            _ = policy.predict_action(policy_obs)
        
        print("✓ 预热完成\n")
        print("="*80)
        print("开始推理...")
        print("="*80)
        
        # 重置参考坐标系和observation历史
        reset_reference_frame()
        obs_history.clear()
        
        step_count = 0
        t_start = time.time()
        
        # 用于MPC：缓存的action序列
        cached_actions = None
        action_index = 0  # 当前执行到第几步
        n_action_steps = cfg.n_action_steps  # 16
        reference_state = None  # 保存参考帧状态
        
        while step_count < args.max_steps:
            step_start = time.time()
            
            # 判断是否需要重新推理
            # 当没有缓存action，或者已经执行完所有步骤时，重新推理
            need_inference = (cached_actions is None) or (action_index >= len(cached_actions))
            
            if need_inference:
                # 先获取observation用于准备policy输入
                obs = robot.capture_observation()
                
                # 准备给policy的observation（可能比较耗时：图像resize、tensor转换等）
                policy_obs = prepare_observation_for_policy(obs, obs_history, cfg, device)
                
                # 在推理前再次读取最新状态作为参考帧（时间戳更接近推理时刻）
                current_state = robot.arm.read()
                reference_state = torch.as_tensor(list(current_state.values()), dtype=torch.float32)
                
                # 运行推理
                with torch.no_grad():
                    inference_start = time.time()
                    result = policy.predict_action(policy_obs)
                    # FADP输出: (16, 13) = [7维机器人控制 + 6维force预测]
                    cached_actions = result['action'][0].detach().to('cpu')  # (16, 13)
                    # 只取前7维用于机器人控制，后6维是force预测（不用于执行）
                    cached_actions = cached_actions[:, :7]  # (16, 7)
                    inference_time = time.time() - inference_start
                
                action_index = 0  # 重置索引
                
                print(f"\n{'='*60}")
                print(f"Step {step_count} - 重新推理")
                print(f"  推理时间: {inference_time*1000:.1f}ms")
                print(f"  预测了 {len(cached_actions)} 步action")
                print(f"  将执行全部 {len(cached_actions)} 步，然后重新推理")
                print(f"  参考状态: x={reference_state[0]:.3f}m, y={reference_state[1]:.3f}m, z={reference_state[2]:.3f}m")
            
            # 执行当前步的action（都使用同一个reference_state）
            action_to_execute = cached_actions[action_index]
            
            print(f"\n  执行 action[{action_index}]:")
            print(f"    位置: x={action_to_execute[0]:.3f}m ({action_to_execute[0]*1000:.1f}mm), "
                  f"y={action_to_execute[1]:.3f}m ({action_to_execute[1]*1000:.1f}mm), "
                  f"z={action_to_execute[2]:.3f}m ({action_to_execute[2]*1000:.1f}mm)")
            print(f"    姿态: yaw={action_to_execute[3]:.3f}, pitch={action_to_execute[4]:.3f}, roll={action_to_execute[5]:.3f}")
            
            # 发送action（传入reference_state）
            robot.send_action(action_to_execute, reference_state)
            
            # 更新索引
            action_index += 1
            step_count += 1
            
            # 可视化（每4步可视化一次以减少开销）
            if step_count % 4 == 0:
                obs_for_vis = robot.capture_observation()
                action_info = {
                    'n_steps': len(cached_actions),
                    'current_step': action_index
                }
                key = visualize_observation(obs_for_vis, step_count, action_info)
                
                if key == ord('q'):
                    print("\n用户请求退出")
                    break
                elif key == ord('s'):
                    print("\n用户停止episode")
                    break
            
            # 控制频率
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
            actual_dt = time.time() - step_start
            actual_hz = 1.0 / actual_dt if actual_dt > 0 else 0
            print(f"  实际频率: {actual_hz:.1f} Hz")
        
        # 统计
        total_time = time.time() - t_start
        avg_hz = step_count / total_time if total_time > 0 else 0
        
        print("\n" + "="*80)
        print("推理完成")
        print("="*80)
        print(f"总步数: {step_count}")
        print(f"总时间: {total_time:.2f}s")
        print(f"平均频率: {avg_hz:.2f} Hz")
        
    except KeyboardInterrupt:
        print("\n\n收到中断信号...")
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 断开连接
        print("\n断开设备连接...")
        robot.disconnect()
        cv2.destroyAllWindows()
        print("✓ 完成")


if __name__ == '__main__':
    main()
