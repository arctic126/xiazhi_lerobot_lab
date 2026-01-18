#!/usr/bin/env python3
"""
JAKA机器人 + TA-VLA模型推理脚本

使用训练好的TA-VLA模型通过WebSocket连接控制JAKA机器人

TA-VLA使用客户端-服务器架构：
1. 先启动服务器：u
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi0_lora_favla \
    --policy.dir /home/sjtu/xiazhi/model/ckpt_1215/forceumi_noup/46000
2. 再运行此脚本连接服务器并控制机器人

使用方法:
python eval_jaka_tavla.py \
    --server_host localhost \
    --server_port 8000 \
    --robot_ip 192.168.31.112 \
    --force_ip 192.168.0.108 \
    --camera_index 0 \
    --task_prompt "clean the basin" \
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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 添加lerobot路径
sys.path.insert(0, str(Path(__file__).parent))

# 添加TA-VLA路径
tavla_path = Path(__file__).parent.parent / "TA-VLA"
if tavla_path.exists():
    sys.path.insert(0, str(tavla_path))
    print(f"已添加TA-VLA路径: {tavla_path}")
else:
    print(f"警告: TA-VLA路径不存在: {tavla_path}")
    print("请确保TA-VLA目录存在于正确位置")

from openpi_client import websocket_client_policy as _websocket_client_policy
from lerobot.common.robot_devices.robots.utils import make_robot_from_config


class ForcePlotter:
    """
    实时三轴力数据可视化
    
    只显示力 (fx, fy, fz)
    """
    def __init__(self, history_length=500):
        """
        初始化力曲线绘制器
        
        Args:
            history_length: 保持的历史数据点数量（默认500）
        """
        self.history_length = history_length
        
        # 配置字体，避免方框
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 数据缓冲区 - 只保留3个力分量
        self.force_histories = {
            'fx': deque(maxlen=history_length),
            'fy': deque(maxlen=history_length),
            'fz': deque(maxlen=history_length),
        }
        
        # 使用非阻塞模式
        plt.ion()
        
        # 创建单个图表
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('Force Real-time Monitor')
        
        # 初始化曲线
        self.lines = {}
        
        # 力的颜色
        force_colors = {
            'fx': '#1f77b4',  # 蓝色
            'fy': '#ff7f0e',  # 橙色
            'fz': '#2ca02c',  # 绿色
        }
        
        # 绘制三条力曲线
        for key in ['fx', 'fy', 'fz']:
            self.lines[key], = self.ax.plot([], [], label=key, 
                                            color=force_colors[key], linewidth=2)
        
        # 设置图表属性
        self.ax.set_xlim(0, history_length)
        self.ax.set_ylim(-50, 50)  # 初始范围，会自动调整
        self.ax.set_xlabel('Time Step')
        self.ax.set_ylabel('Force (N)')
        self.ax.set_title('Force (fx, fy, fz)')
        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.ax.grid(True, alpha=0.3)
        
        # 调整布局，确保图例不被裁剪
        plt.subplots_adjust(right=0.88)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        
        plt.show(block=False)
        plt.pause(0.001)
        
        print("Force plotter initialized")
    
    def update(self, force_data):
        """
        更新力数据并刷新图表
        
        Args:
            force_data: numpy array (6,) 或 torch.Tensor (6,)
                       [fx, fy, fz, mx, my, mz]
                       只使用前3个力分量
        """
        # 转换为numpy
        if isinstance(force_data, torch.Tensor):
            force_data = force_data.cpu().numpy()
        
        # 添加新数据点（只取前3个：fx, fy, fz）
        force_keys = ['fx', 'fy', 'fz']
        for i, key in enumerate(force_keys):
            self.force_histories[key].append(float(force_data[i]))
        
        # 更新曲线数据
        for key in force_keys:
            data = list(self.force_histories[key])
            x = list(range(len(data)))
            self.lines[key].set_data(x, data)
        
        # 动态调整Y轴范围
        force_values = []
        for key in force_keys:
            if len(self.force_histories[key]) > 0:
                force_values.extend(self.force_histories[key])
        
        if len(force_values) > 0:
            y_min = min(force_values)
            y_max = max(force_values)
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            self.ax.set_ylim(y_min - margin, y_max + margin)
        
        # 更新X轴范围
        current_length = len(list(self.force_histories['fx']))
        if current_length < self.history_length:
            self.ax.set_xlim(0, current_length)
        else:
            self.ax.set_xlim(0, self.history_length)
        
        # 刷新图表（非阻塞）
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def close(self):
        """关闭图表窗口"""
        plt.close(self.fig)
        print("✓ 力曲线绘制器已关闭")


def prepare_observation_for_tavla(obs_dict, task_prompt):
    """
    准备observation送入TA-VLA服务器
    
    TA-VLA需要：
    - images: {"images": (224, 224, 3) uint8}
    - state: (7,) float32 - 会自动填充到32维
    - effort: (20, 6) float32 - robot已经从60帧每3帧采样好了20帧
    - prompt: 任务描述字符串
    
    Args:
        obs_dict: 机器人observation字典
        task_prompt: 任务描述
        
    Returns:
        tavla_obs: 准备好的observation字典
    """
    # 1. 图像 - 调整大小到224x224
    image_key = 'observation.images.primary'
    if image_key in obs_dict:
        img = obs_dict[image_key]  # (H, W, C)
        
        # 转换为numpy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # 确保是uint8
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # 调整大小到224x224
        img_resized = cv2.resize(img, (224, 224))
        
        # TA-VLA需要嵌套的字典
        images_dict = {"images": img_resized}
    else:
        raise ValueError("缺少primary相机图像")
    
    # 2. State (7,) - mm单位，不需要转换
    state_key = 'observation.state'
    if state_key in obs_dict:
        state = obs_dict[state_key]
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        state = state[:7].astype(np.float32)  # 只用前7维
    else:
        raise ValueError("缺少state")
    
    # 3. Effort历史 (20, 6) - 直接使用robot返回的数据
    # robot.capture_observation()已经处理好了采样：
    #   - 保存60帧原始force数据
    #   - 每3帧采样1个，返回20帧
    # 这里直接使用即可，无需再处理
    force_key = 'observation.force'
    if force_key in obs_dict:
        force_tensor = obs_dict[force_key]  # 已经是(20, 6)的采样数据
        if isinstance(force_tensor, torch.Tensor):
            effort_array = force_tensor.cpu().numpy().astype(np.float32)
        else:
            effort_array = force_tensor.astype(np.float32)
        
        # 验证shape
        assert effort_array.shape == (20, 6), \
            f"Expected force shape (20, 6), got {effort_array.shape}"
    else:
        # 如果没有力传感器，用零填充
        print("警告: 没有force数据，使用零填充")
        effort_array = np.zeros((20, 6), dtype=np.float32)
    
    # 4. 组合
    tavla_obs = {
        "images": images_dict,
        "state": state,
        "effort": effort_array,
        "prompt": task_prompt,
    }
    
    return tavla_obs


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
        # 显示最新帧（第20帧）
        force_latest = force[-1]  # (6,)
        total_force = np.sqrt(force_latest[0]**2 + force_latest[1]**2 + force_latest[2]**2)
        y_offset += 25
        cv2.putText(vis_img, f"Force: {total_force:.2f} N",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # 添加action信息
    if action_info is not None:
        y_offset += 25
        cv2.putText(vis_img, f"Action: {action_info['action_summary']}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # 显示
    cv2.imshow('JAKA + TA-VLA Inference', vis_img)
    key = cv2.waitKey(1) & 0xFF
    
    return key


def main():
    parser = argparse.ArgumentParser(description='JAKA + TA-VLA推理脚本')
    parser.add_argument('--server_host', '-sh', type=str, default='localhost',
                        help='TA-VLA服务器地址')
    parser.add_argument('--server_port', '-sp', type=int, default=8000,
                        help='TA-VLA服务器端口')
    parser.add_argument('--robot_ip', '-r', type=str, default='192.168.2.64',
                        help='JAKA机器人IP地址')
    parser.add_argument('--force_ip', '-f', type=str, default='192.168.0.108',
                        help='宇立力传感器IP地址')
    parser.add_argument('--camera_index', '-cam', type=int, default=0,
                        help='相机索引')
    parser.add_argument('--task_prompt', '-p', type=str, default='clean the toilet',
                        help='任务描述')
    parser.add_argument('--max_steps', '-m', type=int, default=500,
                        help='每个episode最大步数')
    parser.add_argument('--frequency', '-hz', type=float, default=10.0,
                        help='控制频率 (Hz)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("JAKA机器人 + TA-VLA模型推理")
    print("="*80)
    print(f"服务器: {args.server_host}:{args.server_port}")
    print(f"Robot IP: {args.robot_ip}")
    print(f"Force IP: {args.force_ip}")
    print(f"Camera: {args.camera_index}")
    print(f"Task: {args.task_prompt}")
    print(f"Control frequency: {args.frequency} Hz")
    print("="*80)
    
    # 1. 连接TA-VLA服务器
    print("\n连接TA-VLA服务器...")
    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.server_host,
            port=args.server_port,
        )
        server_metadata = policy.get_server_metadata()
        print(f"✓ 服务器连接成功")
        print(f"  服务器元数据: {server_metadata}")
    except Exception as e:
        print(f"❌ 服务器连接失败: {e}")
        print("\n请确保TA-VLA服务器正在运行：")
        print("  uv run scripts/serve_policy.py --checkpoint=path/to/model.ckpt")
        return
    
    # 2. 初始化JAKA机器人（VLA版本）
    print("\n初始化JAKA机器人（VLA版本）...")
    
    from lerobot.common.robot_devices.robots.configs import JakaRobotVLAConfig
    from lerobot.common.robot_devices.motors.configs import JakaMotorsBusConfig
    from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
    
    jaka_config = JakaRobotVLAConfig(
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
    
    robot = make_robot_from_config(jaka_config)
    
    # 连接机器人
    print("\n连接设备...")
    robot.connect()
    
    # 等待设备稳定
    time.sleep(2)
    
    print("\n✓ 所有设备已连接")
    
    # 3. 初始化力曲线绘制器
    print("\n初始化力曲线绘制器...")
    force_plotter = ForcePlotter(history_length=500)
    
    print("\n控制说明:")
    print("  - 按'q'键: 退出程序")
    print("  - 按's'键: 停止当前episode")
    print("\n准备开始推理...")
    time.sleep(1)
    
    # 4. 推理循环
    try:
        dt = 1.0 / args.frequency
        
        # 预热：发送第一个observation
        print("\n预热policy...")
        dummy_obs = robot.capture_observation()
        tavla_obs = prepare_observation_for_tavla(dummy_obs, args.task_prompt)
        _ = policy.infer(tavla_obs)
        print("✓ 预热完成\n")
        
        print("="*80)
        print("开始推理...")
        print("="*80)
        
        # 重置VLA参考坐标系（每个episode开始时必须调用）
        robot.reset_reference_frame()
        
        # 重置force buffer（清空历史数据）
        robot.raw_force_buffer.clear()
        
        # 保存初始位置（用于'u'键返回）
        initial_obs = robot.capture_observation()
        initial_reference_state = initial_obs["observation.state"]
        print(f"\n初始位置已保存:")
        print(f"  x={initial_reference_state[0]:.3f}m, y={initial_reference_state[1]:.3f}m, z={initial_reference_state[2]:.3f}m")
        print(f"  按'u'键可随时返回此位置\n")
        
        step_count = 0
        t_start = time.time()
        
        # MPC: 缓存的action序列
        cached_actions = None
        action_index = 0
        action_chunk_size = 50  # TA-VLA预测50步
        reference_state = None
        
        while step_count < args.max_steps:
            step_start = time.time()
            
            # 判断是否需要重新推理
            # 当没有缓存action，或者已经执行了一半时，重新推理
            need_inference = (cached_actions is None) or (action_index >= action_chunk_size // 2)
            
            if need_inference:
                # 获取observation
                obs = robot.capture_observation()
                
                # 保存chunk第一帧位置（在参考坐标系下）
                # 注意：参考坐标系在episode开始时已建立，这里只保存chunk第一帧位置
                chunk_first_state = obs["observation.state"]
                robot.save_chunk_first_frame(chunk_first_state)
                
                # 保存当前位置（用于send_action建立参考坐标系，仅首次有效）
                reference_state = chunk_first_state
                
                # 准备给policy的observation
                # robot已经返回采样好的(20, 6) force数据
                tavla_obs = prepare_observation_for_tavla(obs, args.task_prompt)
                
                # 运行推理
                inference_start = time.time()
                result = policy.infer(tavla_obs)
                inference_time = time.time() - inference_start
                
                # 提取action chunk
                action_array = result['actions']
                
                # 处理不同的action shape
                if action_array.ndim == 3:
                    # Shape: (batch, action_horizon, action_dim) - 例如 (1, 50, 32)
                    # 提取: (50, 7)
                    # 注意：必须使用.copy()避免只读numpy数组导致tensor写入失败
                    cached_actions = torch.from_numpy(action_array[0, :, :7].copy()).float()
                    action_chunk_size = cached_actions.shape[0]
                elif action_array.ndim == 2:
                    # Shape可能是：
                    # 1. (action_horizon, action_dim) - 例如 (50, 7) - 已经去掉batch维度
                    # 2. (batch, action_dim) - 例如 (1, 32) - 单步action
                    
                    # 判断方法：如果第二维是7，说明已经提取好了；如果>7，需要切片
                    if action_array.shape[1] == 7:
                        # 已经是(action_horizon, 7)格式，直接使用
                        # 注意：必须使用.copy()避免只读numpy数组导致tensor写入失败
                        cached_actions = torch.from_numpy(action_array.copy()).float()
                        action_chunk_size = cached_actions.shape[0]
                    else:
                        # (batch, action_dim)格式，取前7维
                        # 注意：必须使用.copy()避免只读numpy数组导致tensor写入失败
                        cached_actions = torch.from_numpy(action_array[:, :7].copy()).float()
                        action_chunk_size = cached_actions.shape[0]
                else:
                    raise ValueError(f"Unexpected action shape: {action_array.shape}")
                
                print(f"\n[DEBUG] Action提取结果:")
                print(f"  原始shape: {action_array.shape}")
                print(f"  提取后shape: {cached_actions.shape}")
                print(f"  action_chunk_size: {action_chunk_size}")
                
                action_index = 0  # 重置索引
                
                print(f"\n{'='*60}")
                print(f"Step {step_count} - 重新推理")
                print(f"  推理时间: {inference_time*1000:.1f}ms")
                print(f"  预测了 {action_chunk_size} 步action chunk")
                if action_chunk_size > 1:
                    print(f"  将执行前 {action_chunk_size//2} 步，然后重新推理")
                print(f"  当前状态 (reference_state): x={reference_state[0]:.2f}mm, y={reference_state[1]:.2f}mm, z={reference_state[2]:.2f}mm")
            
            # 执行当前步的action
            action_to_execute = cached_actions[action_index]
            
            if step_count % 10 == 0 or action_index == 0:
                print(f"\n  执行 action[{action_index}]:")
                print(f"    位置delta: x={action_to_execute[0]:.4f}m, y={action_to_execute[1]:.4f}m, z={action_to_execute[2]:.4f}m")
                print(f"    姿态delta: rx={action_to_execute[3]:.3f}rad, ry={action_to_execute[4]:.3f}rad, rz={action_to_execute[5]:.3f}rad")
            
            # 发送action（reference_state只在episode首帧用于建立参考坐标系）
            # 两层参考系统：
            # 1. Episode参考坐标系：整个episode固定
            # 2. Chunk第一帧位置：每次推理时更新（已通过save_chunk_first_frame保存）
            robot.send_action(action_to_execute, reference_state)
            
            # 更新索引
            action_index += 1
            step_count += 1
            
            # 可视化（每4步可视化一次）
            if step_count % 4 == 0:
                obs_for_vis = robot.capture_observation()
                
                # 更新力曲线（取最新的力数据）
                if 'observation.force' in obs_for_vis and obs_for_vis['observation.force'] is not None:
                    force_latest = obs_for_vis['observation.force'][-1]  # 取最新帧 (6,)
                    force_plotter.update(force_latest)
                
                action_info = {
                    'action_summary': f"chunk[{action_index-1}/{action_chunk_size}]: [{action_to_execute[0]:.3f}, {action_to_execute[1]:.3f}, {action_to_execute[2]:.3f}]"
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
                    
                    # 获取当前位置
                    current_obs = robot.capture_observation()
                    current_state = current_obs["observation.state"]
                    
                    print(f"  从当前位置: x={current_state[0]:.3f}mm, y={current_state[1]:.3f}mm, z={current_state[2]:.3f}mm")
                    print(f"  返回初始位置: x={initial_reference_state[0]:.3f}mm, y={initial_reference_state[1]:.3f}mm, z={initial_reference_state[2]:.3f}mm")
                    
                    # VLA关键修复：重置chunk第一帧到初始位置
                    # 这样 zero_action 的目标位置就是 初始位置 + 0 = 初始位置
                    robot.save_chunk_first_frame(initial_reference_state)
                    print("  ✓ 已重置chunk第一帧到初始位置")
                    
                    zero_action = torch.zeros(7, dtype=torch.float32)
                    robot.send_action(zero_action, initial_reference_state)
                    print("  ✓ 正在返回初始点...")
                    time.sleep(0.5)
                    
                    # 进入暂停状态，持续保持在初始点
                    print("\n  机器人正在返回初始点并保持")
                    print("  控制选项:")
                    print("    - 按's'键: 继续推理")
                    print("    - 按'q'键: 退出程序")
                    print("  等待指令...\n")
                    
                    # 暂停循环：持续发送零action保持在初始点
                    while True:
                        # 持续发送zero_action，目标位置 = 初始位置 + 0 = 初始位置
                        robot.send_action(zero_action, initial_reference_state)
                        
                        # 检查键盘输入
                        pause_key = cv2.waitKey(100) & 0xFF
                        
                        if pause_key == ord('s'):
                            print("  继续推理...")
                            # 清空缓存，强制重新推理
                            cached_actions = None
                            action_index = 0
                            # 清空force buffer，从当前状态重新开始
                            robot.raw_force_buffer.clear()
                            break  # 退出暂停循环
                        elif pause_key == ord('q'):
                            print("\n  用户请求退出")
                            # 直接退出整个程序
                            robot.disconnect()
                            cv2.destroyAllWindows()
                            return
                    
                    print("="*60)
                    # 退出暂停后，跳过本次循环，重新开始
                    continue
            
            # 控制频率
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
            actual_dt = time.time() - step_start
            if step_count % 10 == 0:
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
        
        # 关闭力曲线绘制器
        if 'force_plotter' in locals():
            force_plotter.close()
        
        print("✓ 完成")


if __name__ == '__main__':
    main()
