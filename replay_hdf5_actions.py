#!/usr/bin/env python3
"""
从HDF5文件回放action，测试运动控制

这个脚本用于调试：
- 从HDF5文件读取action数据
- 使用固定的参考坐标系（第1帧）
- 依次发送所有action到机械臂
- 检查运动控制是否正确

用法：
python replay_hdf5_actions.py --hdf5 /home/sjtu/xiazhi/ForceUMI/examples/data/session_20251221_164436/episode0.hdf5 --robot_ip 192.168.31.112 --num_frames 500
"""

import sys
import time
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

# 添加lerobot路径
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.robots.configs import JakaRobotConfig
from lerobot.common.robot_devices.motors.configs import JakaMotorsBusConfig


def load_actions_from_hdf5(hdf5_path: str, num_frames: int = None):
    """
    从HDF5文件加载action数据
    
    Args:
        hdf5_path: HDF5文件路径
        num_frames: 要加载的帧数（None或0表示加载全部）
        
    Returns:
        actions: numpy array (num_frames, 7)
    """
    print(f"\n加载HDF5文件: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 打印数据集信息
        print(f"\nHDF5数据集包含:")
        for key in f.keys():
            print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")
        
        # 读取action
        if 'action' not in f:
            raise ValueError("HDF5文件中没有'action'数据集")
        
        all_actions = f['action'][:]
        total_frames = len(all_actions)
        
        print(f"\n总帧数: {total_frames}")
        
        # 如果num_frames为None或0，加载全部
        if num_frames is None or num_frames == 0:
            actions = all_actions
            print(f"将加载全部 {total_frames} 帧")
        else:
            actions = all_actions[:num_frames]
            print(f"将加载前 {min(num_frames, total_frames)} 帧")
        
        # ❌ 注释掉：数据收集时已经做了坐标变换，不需要在replay时重复修正
        # print(f"\n修正rz角度：加 π/2 ({np.pi/2:.4f} rad)")
        # actions[:, 5] += np.pi / 2
        
        print(f"\nAction shape: {actions.shape}")
        print(f"Action范围（修正后）:")
        print(f"  位置 (m): x=[{actions[:, 0].min():.4f}, {actions[:, 0].max():.4f}], "
              f"y=[{actions[:, 1].min():.4f}, {actions[:, 1].max():.4f}], "
              f"z=[{actions[:, 2].min():.4f}, {actions[:, 2].max():.4f}]")
        print(f"  姿态 (rad): rx=[{actions[:, 3].min():.4f}, {actions[:, 3].max():.4f}], "
              f"ry=[{actions[:, 4].min():.4f}, {actions[:, 4].max():.4f}], "
              f"rz=[{actions[:, 5].min():.4f}, {actions[:, 5].max():.4f}]")
        
        return actions


def main():
    parser = argparse.ArgumentParser(description='HDF5 Action回放测试')
    parser.add_argument('--hdf5', type=str, required=True,
                        help='HDF5文件路径')
    parser.add_argument('--robot_ip', type=str, default='192.168.2.64',
                        help='JAKA机器人IP地址')
    parser.add_argument('--num_frames', type=int, default=100,
                        help='要回放的帧数')
    parser.add_argument('--frequency', type=float, default=10.0,
                        help='回放频率 (Hz)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HDF5 Action回放测试")
    print("="*80)
    print(f"HDF5文件: {args.hdf5}")
    print(f"机器人IP: {args.robot_ip}")
    print(f"回放帧数: {args.num_frames}")
    print(f"回放频率: {args.frequency} Hz")
    print("="*80)
    
    # 1. 加载HDF5数据
    try:
        actions = load_actions_from_hdf5(args.hdf5, args.num_frames)
    except Exception as e:
        print(f"\n❌ 加载HDF5失败: {e}")
        return
    
    # 2. 初始化机器人
    print("\n初始化JAKA机器人...")
    
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
        cameras={},  # 不需要相机
        force_sensor={"enabled": False},  # 不需要力传感器
        mock=False
    )
    
    robot = make_robot_from_config(jaka_config)
    
    # 连接机器人
    print("\n连接机器人...")
    robot.connect()
    
    print("\n✓ 机器人已连接")
    
    # 3. 获取当前位置作为参考坐标系
    print("\n获取参考坐标系...")
    obs = robot.capture_observation()
    reference_state = obs["observation.state"]  # (7,) tensor, mm单位
    
    print(f"参考坐标系 (mm, rad):")
    print(f"  位置: x={reference_state[0]:.2f}, y={reference_state[1]:.2f}, z={reference_state[2]:.2f}")
    print(f"  姿态: rx={reference_state[3]:.4f}, ry={reference_state[4]:.4f}, rz={reference_state[5]:.4f}")
    
    # 4. 回放action
    print("\n" + "="*80)
    print("开始回放...")
    print("="*80)
    
    dt = 1.0 / args.frequency
    
    try:
        for i, action_np in enumerate(actions):
            step_start = time.time()
            
            # 转为tensor
            action = torch.from_numpy(action_np).float()
            # ❌ 注释掉：数据收集时rotate_frame_z_90_ccw已经处理了坐标变换
            # action[0]=-action[0]
            # action[1]=-action[1]
            action[5]+=3.1415926/2
            
            # 打印信息
            if i % 10 == 0:  # 每10帧打印一次
                print(f"\n帧 {i}/{len(actions)}:")
                print(f"  Action (m, rad): [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}, "
                      f"{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}]")
            
            # 发送action（使用固定的参考坐标系）
            robot.send_action(action, reference_state)
            
            # 控制频率
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
            actual_dt = time.time() - step_start
            if i % 10 == 0:
                print(f"  实际频率: {1.0/actual_dt:.1f} Hz")
        
        print("\n" + "="*80)
        print(f"✓ 回放完成！共{len(actions)}帧")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n用户中断...")
    
    except Exception as e:
        print(f"\n❌ 回放出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 断开连接
        print("\n断开机器人...")
        robot.disconnect()
        print("✓ 完成")


if __name__ == '__main__':
    main()
