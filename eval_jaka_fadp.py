#!/usr/bin/env python3
"""
JAKA机器人 + FADP模型推理脚本

使用训练好的Force-Aware Diffusion Policy模型控制JAKA机器人

使用方法:
python eval_jaka_fadp.py \
    --checkpoint /home/sjtu/xiazhi/model/ckpt_1215/fadp_noup/20.04.54_train_diffusion_unet_fadp_force_fadp_force/checkpoints/epoch=0500-val_loss=0.194.ckpt \
    --robot_ip 192.168.31.112 \
    --force_ip 192.168.0.108 \
    --camera_index 0 \
    --max_steps 500 \
    --frequency 10

控制说明:
    - 按'q'键: 退出程序
    - 按's'键: 停止当前episode
    - 按'u'键: 返回初始点
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
            if 'force' in cfg.task.shape_meta['obs']:
                print(f"  Force shape: {cfg.task.shape_meta['obs']['force']['shape']}")
            if 'camera_0' in cfg.task.shape_meta['obs']:
                print(f"  Image shape: {cfg.task.shape_meta['obs']['camera_0']['shape']}")
    
    return policy, cfg


def prepare_observation_for_policy(obs_dict, obs_history, cfg, device, policy=None):
    """
    准备observation送入policy
    
    FADP需要：
    - camera_0: (img_obs_horizon, C, H, W) - 默认2帧
    - force: (force_obs_horizon, 6) - 从模型权重推断
    
    注意：camera和force使用不同的observation horizon！
    
    Args:
        obs_dict: 当前observation字典（来自robot.capture_observation()）
        obs_history: observation历史deque
        cfg: 模型配置
        device: torch device
        policy: FADP policy模型（用于推断force_obs_horizon）
        
    Returns:
        policy_obs: 准备好的observation字典
    """
    # 从shape_meta获取真实的horizon配置
    shape_meta = cfg.task.shape_meta
    
    # 获取camera和force的独立horizon
    img_obs_horizon = 2  # 默认值
    force_obs_horizon = 4  # 默认值
    
    if 'obs' in shape_meta:
        # 查找camera key的horizon
        for key in shape_meta['obs']:
            if 'camera' in key.lower() or 'rgb' in key.lower():
                img_obs_horizon = shape_meta['obs'][key].get('horizon', 2)
                break
        
        # ⭐ 从policy模型权重推断实际的force_obs_horizon
        if policy is not None and hasattr(policy, 'obs_encoder'):
            obs_encoder = policy.obs_encoder
            if hasattr(obs_encoder, 'force_encoder') and obs_encoder.force_encoder is not None:
                # 获取force_encoder第一层的输入维度
                first_layer = obs_encoder.force_encoder.mlp[0]  # Linear layer
                actual_force_input_dim = first_layer.in_features
                force_dim = 6  # [fx, fy, fz, mx, my, mz]
                force_obs_horizon = actual_force_input_dim // force_dim
                print(f"  [INFO] 从模型权重推断的force_obs_horizon: {force_obs_horizon} (input_dim={actual_force_input_dim})")
        else:
            # 从配置读取（fallback）
            if 'force' in shape_meta['obs']:
                force_obs_horizon = shape_meta['obs']['force'].get('horizon', 6)
                print(f"  [INFO] 从配置读取的force_obs_horizon: {force_obs_horizon}")
    
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
    
    # 2. Force历史 (force_obs_horizon, 6) - 只取最新的force_obs_horizon帧
    force_key = 'observation.force'
    if force_key in obs_dict:
        full_force_tensor = obs_dict[force_key]  # robot返回的完整force buffer，可能是(6, 6)
        # ⭐ 只取最新的force_obs_horizon帧（模型需要的数量）
        force_tensor = full_force_tensor[-force_obs_horizon:, :]  # (force_obs_horizon, 6)
        policy_obs['force'] = force_tensor.unsqueeze(0).to(device)  # (1, force_obs_horizon, 6)
    
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
    cv2.rectangle(overlay, (5, 5), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
    
    # 添加文本
    y_offset = 25
    cv2.putText(vis_img, f"Step: {step_count}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 添加state信息
    if 'observation.state' in obs_dict:
        state = obs_dict['observation.state']
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        y_offset += 25
        cv2.putText(vis_img, f"Pos: ({state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f})",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 添加force信息
    if 'observation.force' in obs_dict:
        force = obs_dict['observation.force']
        if isinstance(force, torch.Tensor):
            force = force.cpu().numpy()
        # 只显示最新帧
        if force.ndim == 2:
            force_latest = force[-1]
        else:
            force_latest = force
        total_force = np.sqrt(force_latest[0]**2 + force_latest[1]**2 + force_latest[2]**2)
        y_offset += 25
        cv2.putText(vis_img, f"Force: {total_force:.2f} N",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # 添加action信息
    if action_info is not None:
        y_offset += 25
        cv2.putText(vis_img, f"Action steps: {action_info['n_steps']}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # 显示
    cv2.imshow('JAKA + FADP Inference', vis_img)
    key = cv2.waitKey(1) & 0xFF
    
    return key


def main():
    parser = argparse.ArgumentParser(description='JAKA + FADP推理脚本')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='FADP模型checkpoint路径 (.ckpt)')
    parser.add_argument('--robot_ip', '-r', type=str, default='192.168.2.64',
                        help='JAKA机器人IP地址')
    parser.add_argument('--force_ip', '-f', type=str, default='192.168.0.108',
                        help='宇立力传感器IP地址')
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
    print("JAKA机器人 + FADP模型推理")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Robot IP: {args.robot_ip}")
    print(f"Force IP: {args.force_ip}")
    print(f"Camera: {args.camera_index}")
    print(f"Control frequency: {args.frequency} Hz")
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
        print("  建议检查：")
        print("  1. NVIDIA驱动是否安装")
        print("  2. CUDA toolkit是否安装")
        print("  3. PyTorch是否为CUDA版本")
        print("  4. CUDA_VISIBLE_DEVICES环境变量")
        device = torch.device('cpu')
    
    print(f"\n使用设备: {device}")
    
    # 2. 加载FADP模型
    policy, cfg = load_fadp_model(args.checkpoint, device=device)
    
    # 3. 初始化JAKA机器人
    print("\n初始化JAKA机器人...")
    
    # 导入配置类
    from lerobot.common.robot_devices.robots.configs import JakaRobotConfig
    from lerobot.common.robot_devices.motors.configs import JakaMotorsBusConfig
    from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
    
    # 创建JAKA配置
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
            "enabled": True,
            "ip_addr": args.force_ip,
            "port": 4008
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
        
        # 确定obs_history的maxlen（需要容纳camera和force的最大horizon）
        # 训练时：img_obs_horizon=2, force_obs_horizon=6
        # obs_history需要至少容纳max(2, 6) = 6帧
        shape_meta = cfg.task.shape_meta
        max_obs_horizon = 6  # 默认值（force的horizon）
        
        if 'obs' in shape_meta:
            obs_horizons = []
            for key, attr in shape_meta['obs'].items():
                if 'horizon' in attr:
                    obs_horizons.append(attr['horizon'])
            if obs_horizons:
                max_obs_horizon = max(obs_horizons)
        
        print(f"  Observation history buffer size: {max_obs_horizon}")
        
        dummy_obs = robot.capture_observation()
        obs_history = deque(maxlen=max_obs_horizon)
        policy_obs = prepare_observation_for_policy(dummy_obs, obs_history, cfg, device, policy)
        
        with torch.no_grad():
            policy.reset()
            _ = policy.predict_action(policy_obs)
        
        print("✓ 预热完成\n")
        print("="*80)
        print("开始推理...")
        print("="*80)
        
        # 重置机器人状态
        robot.force_buffer.clear()
        obs_history.clear()
        
        # 保存初始位置（用于'u'键返回）
        initial_state = robot.arm.read()
        initial_reference_state = torch.as_tensor(list(initial_state.values()), dtype=torch.float32)
        print(f"\n初始位置已保存:")
        print(f"  x={initial_reference_state[0]:.3f}m, y={initial_reference_state[1]:.3f}m, z={initial_reference_state[2]:.3f}m")
        print(f"  按'u'键可随时返回此位置\n")
        
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
                policy_obs = prepare_observation_for_policy(obs, obs_history, cfg, device, policy)
                
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
                elif key == ord('u'):
                    print("\n" + "="*60)
                    print("[返回初始点并暂停]")
                    print("="*60)
                    print(f"  从当前位置: x={reference_state[0]:.3f}m, y={reference_state[1]:.3f}m, z={reference_state[2]:.3f}m")
                    print(f"  返回初始位置: x={initial_reference_state[0]:.3f}m, y={initial_reference_state[1]:.3f}m, z={initial_reference_state[2]:.3f}m")
                    
                    # 发送零action，使用初始位置作为参考
                    zero_action = torch.zeros(7, dtype=torch.float32)
                    robot.send_action(zero_action, initial_reference_state)
                    print("  ✓ 已返回初始点")
                    time.sleep(0.5)
                    
                    # 进入暂停状态，持续保持在初始点
                    print("\n  机器人已停在初始点")
                    print("  控制选项:")
                    print("    - 按's'键: 继续推理")
                    print("    - 按'q'键: 退出程序")
                    print("  等待指令...\n")
                    
                    # 暂停循环：持续保持位置直到收到指令
                    while True:
                        # 持续发送零action保持在初始点
                        robot.send_action(zero_action, initial_reference_state)
                        
                        # 检查键盘输入
                        pause_key = cv2.waitKey(100) & 0xFF
                        
                        if pause_key == ord('s'):
                            print("  继续推理...")
                            # 清空缓存，强制重新推理
                            cached_actions = None
                            action_index = 0
                            # 清空观察历史，从当前状态重新开始
                            obs_history.clear()
                            robot.force_buffer.clear()
                            break  # 退出暂停循环
                        elif pause_key == ord('q'):
                            print("\n  用户请求退出")
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
