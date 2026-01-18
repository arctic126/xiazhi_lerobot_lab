#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
使用teleop对piper机械臂进行遥操作，用endpose控制
参考teleop_ik.py和piper sdk
"""

import sys
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from piper_sdk import C_PiperInterface_V2
from teleop import Teleop


class PiperRobotArm:
    def __init__(self):
        # 初始化机械臂接口
        self.piper = C_PiperInterface_V2("can0")
        self.piper.ConnectPort()
        
        # 初始化阈值参数
        self.pose_factor = 1000  # 单位 0.001mm或0.001度
        
        # 初始化相对位置模式标志
        self.relative_mode = False
        
        # 启用机械臂
        self.enable_arm()
        
    def enable_arm(self):
        """启用机械臂并检查状态"""
        enable_flag = False
        loop_flag = False
        timeout = 5
        start_time = time.time()
        
        while not loop_flag:
            print("--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            
            enable_flag = all(enable_list)
            self.piper.EnableArm(7)  # 启用所有关节(1+2+4=7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)  # 初始化夹爪
            
            print(f"使能状态: {enable_flag}")
            print("--------------------")
            
            if enable_flag:
                loop_flag = True
                break
            
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print("启用超时...")
                break
                
            time.sleep(0.5)
        
        return enable_flag

    def disable_arm(self):
        """禁用机械臂"""
        self.piper.DisableArm(7)  # 禁用所有关节
        self.piper.GripperCtrl(0, 1000, 0x02, 0)  # 禁用夹爪
        
    def get_current_pose(self):
        """获取当前末端位姿"""
        return self.piper.GetFK('feedback')[-1]
        
    def move_to_position(self, target_pose, gripper_state=None):
        """移动到目标位姿
        Args:
            target_pose: 目标位姿 [x, y, z, rx, ry, rz] 单位为mm和度
            gripper_state: 夹爪状态，1为打开，0为关闭，None为不改变
        """
        if self.relative_mode:
            # 获取当前位姿
            current_pose = self.piper.GetFK('feedback')[-1]
            
            # 计算目标位姿（绝对位姿）
            eef_x = round((target_pose[0] + current_pose[0]) * self.pose_factor)
            eef_y = round((target_pose[1] + current_pose[1]) * self.pose_factor)
            eef_z = round((target_pose[2] + current_pose[2]) * self.pose_factor)
            eef_rx = round((target_pose[3] + current_pose[3]) * self.pose_factor)
            eef_ry = round((target_pose[4] + current_pose[4]) * self.pose_factor)
            eef_rz = round((target_pose[5] + current_pose[5]) * self.pose_factor)
        else:
            # 直接使用绝对位姿
            eef_x = round(target_pose[0] * self.pose_factor)
            eef_y = round(target_pose[1] * self.pose_factor)
            eef_z = round(target_pose[2] * self.pose_factor)
            eef_rx = round(target_pose[3] * self.pose_factor)
            eef_ry = round(target_pose[4] * self.pose_factor)
            eef_rz = round(target_pose[5] * self.pose_factor)
        
        # 设置末端位姿控制模式并发送位姿指令
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)  # 末端位姿控制模式
        self.piper.EndPoseCtrl(eef_x, eef_y, eef_z, eef_rx, eef_ry, eef_rz)
        
        # 控制夹爪（如果提供）
        if gripper_state is not None:
            gripper_angle = 85000 if gripper_state == 1.0 else 0  # 夹爪角度（0.001度）
            self.piper.GripperCtrl(gripper_angle, 1000, 0x01, 0)  # 控制夹爪
    
    def set_relative_mode(self, relative=False):
        """设置相对/绝对位姿模式"""
        self.relative_mode = relative

    @staticmethod
    def matrix_to_euler(rotation_matrix, degrees=True):
        """
        将旋转矩阵转换为欧拉角（ZYX顺序，即先绕Z轴，再绕Y轴，最后绕X轴）
        
        Args:
            rotation_matrix: 3x3旋转矩阵
            degrees: 是否返回角度制的欧拉角（默认为True）
            
        Returns:
            欧拉角 [rx, ry, rz]
        """
        r = R.from_matrix(rotation_matrix)
        euler = r.as_euler('ZYX', degrees=degrees)
        # 注意：scipy的ZYX顺序返回的是[rz, ry, rx]，因此需要反转顺序
        return euler[::-1]  # 返回[rx, ry, rz]
    
    @staticmethod
    def euler_to_quaternion(euler_angles, degrees=True):
        """
        将欧拉角转换为四元数
        
        Args:
            euler_angles: 欧拉角 [rx, ry, rz]，ZYX顺序
            degrees: 欧拉角是否为角度制（默认为True）
            
        Returns:
            四元数 [qx, qy, qz, qw]
        """
        # 反转顺序为scipy要求的ZYX顺序
        rx, ry, rz = euler_angles
        r = R.from_euler('ZYX', [rz, ry, rx], degrees=degrees)
        return r.as_quat()  # 返回[qx, qy, qz, qw]
    
    @staticmethod
    def quaternion_to_euler(quaternion, degrees=True):
        """
        将四元数转换为欧拉角
        
        Args:
            quaternion: 四元数 [qx, qy, qz, qw]
            degrees: 是否返回角度制的欧拉角（默认为True）
            
        Returns:
            欧拉角 [rx, ry, rz]
        """
        r = R.from_quat(quaternion)
        euler = r.as_euler('ZYX', degrees=degrees)
        # 注意：scipy的ZYX顺序返回的是[rz, ry, rx]，因此需要反转顺序
        return euler[::-1]  # 返回[rx, ry, rz]
    
    @staticmethod
    def matrix_to_quaternion(rotation_matrix):
        """
        将旋转矩阵转换为四元数
        
        Args:
            rotation_matrix: 3x3旋转矩阵
            
        Returns:
            四元数 [qx, qy, qz, qw]
        """
        r = R.from_matrix(rotation_matrix)
        return r.as_quat()


def main():
    # 初始化目标位姿和夹爪状态
    target_pose = None
    gripper_state = None
    
    # 创建机械臂和遥操作对象
    robot = PiperRobotArm()
    teleop = Teleop()
    
    # 设置遥操作回调函数
    def on_teleop_callback(pose, message):
        nonlocal target_pose, gripper_state
        
        if message["move"]:
            # 从pose矩阵中提取位置和旋转矩阵
            position = pose[:3, 3]  # x, y, z
            rotation_matrix = pose[:3, :3]  # 3x3旋转矩阵
            
            # 使用scipy的Rotation计算欧拉角（以度为单位）
            euler_angles = PiperRobotArm.matrix_to_euler(rotation_matrix, degrees=True)
            
            # 将位置从米转换为毫米
            position = position * 1000
            
            # 设置目标位姿
            target_pose = [position[0], position[1], position[2], euler_angles[0], euler_angles[1], euler_angles[2]]
            
            # 如果需要，可以计算四元数（这里只是演示，实际中如果不需要可以删除）
            # quaternion = PiperRobotArm.matrix_to_quaternion(rotation_matrix)
            # print(f"四元数: {quaternion}")
        
        # 如果有夹爪控制信息
        if "gripper" in message:
            gripper_state = 1.0 if message["gripper"] > 0.5 else 0.0

    # 获取当前位姿并设置到teleop对象中
    current_pose = robot.get_current_pose()
    # 创建一个4x4的变换矩阵
    matrix = np.eye(4)
    
    # 设置初始位置（毫米转米）
    matrix[0, 3] = current_pose[0] / 1000  # x
    matrix[1, 3] = current_pose[1] / 1000  # y
    matrix[2, 3] = current_pose[2] / 1000  # z
    
    # 旋转部分可以从欧拉角转换到旋转矩阵，但这里为了简单，使用单位矩阵
    # 如果需要使用实际的旋转，可以这样处理:
    euler_angles = [current_pose[3] / 1000, current_pose[4] / 1000, current_pose[5] / 1000]
    r = R.from_euler('ZYX', [euler_angles[2], euler_angles[1], euler_angles[0]], degrees=True)
    matrix[:3, :3] = r.as_matrix()
    
    # 设置初始姿态
    teleop.set_pose(matrix)
    
    # 订阅遥操作事件并启动线程
    teleop.subscribe(on_teleop_callback)
    thread = threading.Thread(target=teleop.run)
    thread.start()

    try:
        # 主循环
        while True:
            if target_pose is not None:
                robot.move_to_position(target_pose, gripper_state)
                target_pose = None  # 处理完后重置目标位姿
            time.sleep(0.01)  # 短暂休眠以避免CPU占用过高
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 清理并退出
        teleop.stop()
        thread.join()
        robot.disable_arm()
        print("已禁用机械臂，程序退出")


if __name__ == "__main__":
    main() 