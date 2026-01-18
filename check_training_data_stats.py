#!/usr/bin/env python3
"""
检查ForceUMI训练数据的统计信息
查看action的实际幅度，确定模型输出小是否正常
"""

import h5py
import numpy as np
from pathlib import Path

def check_episode_stats(episode_path):
    """检查单个episode的统计信息"""
    with h5py.File(episode_path, 'r') as f:
        # 读取数据
        actions = f['action'][:]
        
        print(f"\n文件: {episode_path.name}")
        print(f"{'='*60}")
        
        # Action shape
        print(f"Action shape: {actions.shape}")
        print(f"  - 总帧数: {actions.shape[0]}")
        print(f"  - 维度: {actions.shape[1]} (x,y,z,yaw,pitch,roll,gripper)")
        
        # 位置维度统计 (x, y, z)
        print(f"\n位置维度 (x, y, z) 统计 [单位: m]:")
        labels = ['x', 'y', 'z']
        for i, label in enumerate(labels):
            data = actions[:, i]
            print(f"  {label}:")
            print(f"    范围: [{data.min():.4f}, {data.max():.4f}]")
            print(f"    均值: {data.mean():.4f}")
            print(f"    标准差: {data.std():.4f}")
            print(f"    平均绝对值: {np.abs(data).mean():.4f}")
            print(f"    最大绝对值: {np.abs(data).max():.4f}")
        
        # 检查相邻帧的差值（这才是真正的增量）
        print(f"\n相邻帧差值 (真实移动量) [单位: m]:")
        action_diff = np.diff(actions[:, :3], axis=0)
        for i, label in enumerate(labels):
            diff = action_diff[:, i]
            print(f"  Δ{label}:")
            print(f"    范围: [{diff.min():.4f}, {diff.max():.4f}]")
            print(f"    均值: {diff.mean():.4f}")
            print(f"    标准差: {diff.std():.4f}")
            print(f"    平均绝对值: {np.abs(diff).mean():.4f}")
            print(f"    最大绝对值: {np.abs(diff).max():.4f}")
        
        # 姿态维度统计 (yaw, pitch, roll)
        print(f"\n姿态维度 (yaw, pitch, roll) 统计 [单位: rad]:")
        labels = ['yaw', 'pitch', 'roll']
        for i, label in enumerate(labels, start=3):
            data = actions[:, i]
            print(f"  {label}:")
            print(f"    范围: [{data.min():.4f}, {data.max():.4f}]")
            print(f"    均值: {data.mean():.4f}")
            print(f"    最大绝对值: {np.abs(data).max():.4f}")


def main():
    """主函数"""
    print("="*80)
    print("ForceUMI训练数据统计分析")
    print("="*80)
    
    # 数据目录（使用绝对路径）
    data_dir = Path("/home/hyx/xiazhi/ForceUMI/data/session_20251025_142256")
    
    if not data_dir.exists():
        print(f"\n错误: 数据目录不存在: {data_dir}")
        print("请确认路径是否正确")
        return
    
    # 获取所有episode文件
    episode_files = sorted(data_dir.glob("episode*.hdf5"))
    
    if not episode_files:
        print(f"\n错误: 未找到episode文件在: {data_dir}")
        return
    
    print(f"\n找到 {len(episode_files)} 个episode文件")
    
    # 分析前3个episode
    print(f"\n分析前3个episode:")
    for episode_file in episode_files[:3]:
        check_episode_stats(episode_file)
    
    # 汇总统计
    print("\n" + "="*80)
    print("汇总统计 (所有episodes)")
    print("="*80)
    
    all_actions = []
    all_diffs = []
    
    for episode_file in episode_files:
        with h5py.File(episode_file, 'r') as f:
            actions = f['action'][:]
            all_actions.append(actions)
            # 计算增量
            diff = np.diff(actions[:, :3], axis=0)
            all_diffs.append(diff)
    
    all_actions = np.concatenate(all_actions, axis=0)
    all_diffs = np.concatenate(all_diffs, axis=0)
    
    print(f"\n总帧数: {all_actions.shape[0]}")
    
    print(f"\n位置绝对值汇总 [单位: m]:")
    labels = ['x', 'y', 'z']
    for i, label in enumerate(labels):
        data = all_actions[:, i]
        print(f"  {label}: 范围=[{data.min():.4f}, {data.max():.4f}], "
              f"均值={data.mean():.4f}, 标准差={data.std():.4f}")
    
    print(f"\n相邻帧增量汇总 (这是模型学习的目标) [单位: m]:")
    for i, label in enumerate(labels):
        diff = all_diffs[:, i]
        print(f"  Δ{label}: 最大绝对值={np.abs(diff).max():.4f}, "
              f"平均绝对值={np.abs(diff).mean():.4f}, "
              f"标准差={diff.std():.4f}")
    
    # 关键判断
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    
    max_diff = np.abs(all_diffs).max(axis=0)
    avg_diff = np.abs(all_diffs).mean(axis=0)
    
    print(f"\n每个维度的最大单步移动量:")
    for i, label in enumerate(labels):
        print(f"  {label}: {max_diff[i]*1000:.2f} mm")
    
    print(f"\n每个维度的平均单步移动量:")
    for i, label in enumerate(labels):
        print(f"  {label}: {avg_diff[i]*1000:.2f} mm")
    
    print(f"\n推理中看到的action值 (x=0.009, y=0.012, z=0.006):")
    inference_vals = [0.009, 0.012, 0.006]
    for i, (label, val) in enumerate(zip(labels, inference_vals)):
        print(f"  {label}={val:.3f}m ({val*1000:.1f}mm) - ", end="")
        if val <= max_diff[i]:
            print(f"✓ 在训练数据范围内 (max={max_diff[i]:.4f}m)")
        else:
            print(f"⚠ 超出训练数据范围 (max={max_diff[i]:.4f}m)")
    
    print("\n如果推理值在训练范围内，说明：")
    print("  1. 模型行为正常，学习到了数据中的运动模式")
    print("  2. 如果需要更大幅度运动，需要重新采集数据")
    print("  3. 或者这就是任务本身的特点（精细操作）")


if __name__ == "__main__":
    main()
