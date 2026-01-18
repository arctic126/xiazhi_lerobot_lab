#!/usr/bin/env python3
"""
JAKA机器人LeRobot集成测试脚本

测试功能：
1. JAKA机器人连接和控制
2. 力传感器数据读取
3. 相机图像采集
4. Observation采集（state + force + images）
5. 简单的机械臂控制测试

使用方法：
    python test_jaka_integration.py
    
控制说明：
    - 空格键：执行小幅度测试移动
    - 'r': 重置reference_state
    - 'q': 退出测试
"""

import sys
import time
import numpy as np
import cv2
import torch

# 添加lerobot路径
#sys.path.insert(0, '/home/hyx/xiazhi/jaka-Lerobot/lerobot_lab')

from lerobot.common.robot_devices.robots.utils import make_robot


def print_separator(char="=", length=80):
    """打印分隔线"""
    print(char * length)


def print_section(title):
    """打印章节标题"""
    print_separator()
    print(f"  {title}")
    print_separator()


def visualize_observation(obs_dict, frame_count, show_images=True):
    """
    可视化观测数据
    
    Args:
        obs_dict: observation字典
        frame_count: 帧计数
        show_images: 是否显示图像窗口
    """
    print(f"\n{'='*60}")
    print(f"Frame #{frame_count}")
    print(f"{'='*60}")
    
    # 1. 打印State（7维：x, y, z, yaw, pitch, roll, gripper）
    if "observation.state" in obs_dict:
        state = obs_dict["observation.state"]
        print(f"\n📍 State (7D - 末端位姿):")
        print(f"   Shape: {state.shape}")
        state_np = state.cpu().numpy() if torch.is_tensor(state) else state
        print(f"   x      = {state_np[0]:>8.2f} mm")
        print(f"   y      = {state_np[1]:>8.2f} mm")
        print(f"   z      = {state_np[2]:>8.2f} mm")
        print(f"   yaw    = {state_np[3]:>8.4f} rad ({np.degrees(state_np[3]):>6.2f}°)")
        print(f"   pitch  = {state_np[4]:>8.4f} rad ({np.degrees(state_np[4]):>6.2f}°)")
        print(f"   roll   = {state_np[5]:>8.4f} rad ({np.degrees(state_np[5]):>6.2f}°)")
        print(f"   gripper= {state_np[6]:>8.4f}")
    
    # 2. 打印Force（历史数据：(n_obs_steps, 6)，6维：fx, fy, fz, mx, my, mz）
    if "observation.force" in obs_dict:
        force = obs_dict["observation.force"]
        print(f"\n💪 Force (历史力/力矩数据):")
        print(f"   Shape: {force.shape}")
        force_np = force.cpu().numpy() if torch.is_tensor(force) else force
        
        # 获取最新帧的force数据
        if force_np.ndim == 2:
            force_latest = force_np[-1]  # 最新帧
            print(f"   (显示最新帧 #{force_np.shape[0]}/{ force_np.shape[0]})")
        else:
            force_latest = force_np
        
        print(f"   fx = {force_latest[0]:>8.3f} N")
        print(f"   fy = {force_latest[1]:>8.3f} N")
        print(f"   fz = {force_latest[2]:>8.3f} N")
        print(f"   mx = {force_latest[3]:>8.3f} Nm")
        print(f"   my = {force_latest[4]:>8.3f} Nm")
        print(f"   mz = {force_latest[5]:>8.3f} Nm")
        
        # 计算合力
        total_force = np.sqrt(force_latest[0]**2 + force_latest[1]**2 + force_latest[2]**2)
        print(f"   总力 = {total_force:>8.3f} N")
    else:
        print(f"\n⚠️  Force数据不可用")
    
    # 3. 打印Images信息
    image_keys = [k for k in obs_dict.keys() if k.startswith("observation.images")]
    if image_keys:
        print(f"\n📷 Images ({len(image_keys)}个相机):")
        for key in image_keys:
            img = obs_dict[key]
            img_np = img.cpu().numpy() if torch.is_tensor(img) else img
            cam_name = key.replace("observation.images.", "")
            print(f"   {cam_name:>12}: {img_np.shape} (dtype: {img_np.dtype})")
    
    # 4. 显示图像窗口（如果启用）
    if show_images and image_keys:
        for key in image_keys:
            img = obs_dict[key]
            img_np = img.cpu().numpy() if torch.is_tensor(img) else img
            
            # 确保图像格式正确（OpenCV使用BGR）
            if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                img_np = (img_np * 255).astype(np.uint8)
            
            # 如果是RGB，转为BGR
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_display = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_display = img_np.copy()
            
            # 在图像上叠加信息
            cam_name = key.replace("observation.images.", "")
            overlay = img_display.copy()
            
            # 添加半透明背景
            cv2.rectangle(overlay, (5, 5), (400, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0, img_display)
            
            # 添加文本
            y_offset = 25
            cv2.putText(img_display, f"Frame: {frame_count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if "observation.state" in obs_dict:
                state_np = obs_dict["observation.state"].cpu().numpy() if torch.is_tensor(obs_dict["observation.state"]) else obs_dict["observation.state"]
                y_offset += 20
                cv2.putText(img_display, f"Pos: ({state_np[0]:.1f}, {state_np[1]:.1f}, {state_np[2]:.1f})", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            if "observation.force" in obs_dict:
                force_np = obs_dict["observation.force"].cpu().numpy() if torch.is_tensor(obs_dict["observation.force"]) else obs_dict["observation.force"]
                # 获取最新帧的force
                force_latest = force_np[-1] if force_np.ndim == 2 else force_np
                total_force = np.sqrt(force_latest[0]**2 + force_latest[1]**2 + force_latest[2]**2)
                y_offset += 20
                cv2.putText(img_display, f"Force: {total_force:.2f} N", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 显示图像
            cv2.imshow(f"JAKA Camera - {cam_name}", img_display)


def test_small_movement(robot, direction="x", delta=5.0):
    """
    测试小幅度移动
    
    Args:
        robot: JakaRobot实例
        direction: 移动方向 ('x', 'y', 'z', 'yaw', 'pitch', 'roll')
        delta: 移动量（位置单位mm，姿态单位rad）
    """
    print(f"\n🔧 测试小幅度移动: {direction} += {delta}")
    obs = robot.capture_observation()
    reference_state = obs["observation.state"] 
    # 创建action（相对于reference_state的增量）
    action = torch.zeros(7, dtype=torch.float32)
    
    direction_map = {
        'x': 0, 'y': 1, 'z': 2,
        'yaw': 3, 'pitch': 4, 'roll': 5,
        'gripper': 6
    }
    
    if direction in direction_map:
        idx = direction_map[direction]
        action[idx] = delta
        
        print(f"   Action: {action.numpy()}")
        
        # 发送action
        robot.send_action(action,reference_state)
        print(f"   ✓ Action已发送")
        
        # 等待运动完成
        time.sleep(0.5)
        
        # 读取新状态
        obs = robot.capture_observation()
        new_state = obs["observation.state"].cpu().numpy()
        print(f"   新位置: x={new_state[0]:.2f}, y={new_state[1]:.2f}, z={new_state[2]:.2f}")
    else:
        print(f"   ⚠️ 未知方向: {direction}")


def main():
    """主测试函数"""
    
    print_section("JAKA机器人LeRobot集成测试")
    print("测试内容：")
    print("  1. 设备连接测试")
    print("  2. Observation采集测试")
    print("  3. 简单控制测试")
    print("  4. 实时数据显示")
    print("\n控制说明：")
    print("  空格键 - 执行测试移动")
    print("  'r'键  - 重置reference_state")
    print("  'q'键  - 退出测试")
    print_separator()
    
    input("\n按Enter键开始测试...")
    
    robot = None
    
    try:
        # ==================== 1. 初始化和连接 ====================
        print_section("1. 初始化JAKA机器人")
        
        print("创建robot实例...")
        robot = make_robot("jaka", inference_time=True)
        print("✓ Robot实例创建成功")
        
        print(f"\n机器人类型: {robot.robot_type}")
        print(f"推理模式: {robot.inference_time}")
        print(f"配置的相机数量: {robot.num_cameras}")
        print(f"是否启用力传感器: {robot.force_sensor is not None}")
        
        print("\n正在连接设备...")
        robot.connect()
        print("✓ 所有设备连接成功")
        
        # 等待初始化稳定
        time.sleep(2)
        
        # ==================== 2. 静态测试 ====================
        print_section("2. 静态Observation采集测试（10帧）")
        
        for i in range(10):
            print(f"\n采集第 {i+1}/10 帧...")
            obs = robot.capture_observation()
            
            # 只打印第一帧和最后一帧的详细信息
            if i == 0 or i == 9:
                visualize_observation(obs, i+1, show_images=(i==0))
            else:
                print(f"  ✓ 帧 {i+1} 采集成功")
            
            if i == 0:
                print("\n按任意键继续...")
                cv2.waitKey(0)
            
            time.sleep(0.1)
        
        print("\n✓ 静态采集测试完成")
        
        # ==================== 3. 简单控制测试 ====================
        print_section("3. 简单控制测试")
        
        print("\n准备执行小幅度移动测试...")
        input("按Enter键继续...")
        
        # 测试X方向移动
        test_small_movement(robot, direction='x', delta=5.0)
        time.sleep(1)
        
        # 测试Y方向移动
        test_small_movement(robot, direction='y', delta=5.0)
        time.sleep(1)
        
        # 测试Z方向移动
        test_small_movement(robot, direction='z', delta=3.0)
        time.sleep(1)
        
        print("\n✓ 简单控制测试完成")
        
        # ==================== 4. 实时循环测试 ====================
        print_section("4. 实时数据采集（按'q'退出）")
        
        frame_count = 0
        start_time = time.time()
        
        print("\n开始实时采集...")
        print("控制说明：")
        print("  空格键 - X方向+5mm")
        print("  'r'键  - 重置reference")
        print("  'q'键  - 退出")
        
        while True:
            frame_count += 1
            
            # 采集observation
            obs = robot.capture_observation()
            
            # 显示（每5帧打印一次详细信息）
            show_detail = (frame_count % 5 == 0)
            if show_detail:
                visualize_observation(obs, frame_count, show_images=True)
            else:
                # 只更新图像窗口
                image_keys = [k for k in obs.keys() if k.startswith("observation.images")]
                for key in image_keys:
                    img = obs[key]
                    img_np = img.cpu().numpy() if torch.is_tensor(img) else img
                    if img_np.dtype == np.float32:
                        img_np = (img_np * 255).astype(np.uint8)
                    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                        img_display = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    else:
                        img_display = img_np
                    
                    cam_name = key.replace("observation.images.", "")
                    cv2.imshow(f"JAKA Camera - {cam_name}", img_display)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n用户请求退出")
                break
            elif key == ord(' '):
                print("\n执行测试移动...")
                test_small_movement(robot, direction='x', delta=5.0)
            elif key == ord('r'):
                print("\n重置reference_state...")
                robot.reference_state = None
                print("✓ Reference已重置，下次capture_observation将记录新参考")
            
            # 控制帧率（约30Hz）
            time.sleep(0.03)
        
        # 统计信息
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n统计信息：")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {elapsed:.2f}s")
        print(f"  平均FPS: {avg_fps:.2f}")
        
    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在安全退出...")
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # ==================== 5. 断开连接 ====================
        print_section("5. 断开连接")
        
        if robot and robot.is_connected:
            print("正在断开设备...")
            robot.disconnect()
            print("✓ 所有设备已断开")
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        print_separator()
        print("测试完成！")


if __name__ == "__main__":
    main()
