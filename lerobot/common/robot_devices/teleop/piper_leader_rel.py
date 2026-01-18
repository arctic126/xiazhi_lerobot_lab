from typing import Dict
import time
from piper_sdk import C_PiperInterface_V2
from lerobot.common.robot_devices.robots.configs import PiperRobotConfig, Piper2RobotConfig

class PiperLeader:
    def __init__(self, config: PiperRobotConfig | Piper2RobotConfig):
        # 初始化机械臂接口
        self.piper = C_PiperInterface_V2(config.follower_arm['main'].can_name)
        self.piper.ConnectPort()
        
        # 初始化关节和夹爪状态
        self.joints = [0.823, -1.723, 1.726, -1.152, 2.368, 21.049]  # 6个关节
        self.gripper = 0.0  # 夹爪状态
        
        # 设置运行标志
        self.running = True
    
    def update_joints(self):
        """更新关节状态，从主臂读取数据"""
        # 获取主臂关节信息
        master_joint_info = self.piper.GetArmJointCtrl()
        master_gripper_info = self.piper.GetArmGripperCtrl()
        # print('master_joint_info', master_joint_info)
        slave_joint_info = self.piper.GetArmJointMsgs()
        if master_joint_info is not None:
            
            joint_angles = [
                (master_joint_info.joint_ctrl.joint_1) / 1000,  
                (master_joint_info.joint_ctrl.joint_2) / 1000,
                (master_joint_info.joint_ctrl.joint_3) / 1000,
                (master_joint_info.joint_ctrl.joint_4) / 1000,
                (master_joint_info.joint_ctrl.joint_5) / 1000,
                (master_joint_info.joint_ctrl.joint_6) / 1000
            ]
            
            for i in range(6):
                self.joints[i] = joint_angles[i]
        
        if master_gripper_info is not None:
            # 夹爪角度转换为二进制状态：0(关闭)或1(打开)
            gripper_angle = master_gripper_info.gripper_ctrl.grippers_angle
            # 如果夹爪角度大于设定阈值，则认为是打开状态
            self.gripper = 1.0 if gripper_angle > 30000 else 0.0
    
    def get_action(self) -> Dict:
        """返回机械臂的当前状态"""

        self.update_joints()
        return {
            'joint1': self.joints[0],
            'joint2': self.joints[1],
            'joint3': self.joints[2],
            'joint4': self.joints[3],
            'joint5': self.joints[4],
            'joint6': self.joints[5],
            'gripper': self.gripper
        }
    
    def stop(self):
        """停止运行"""
        self.running = False
        print("Piper leader exits")

    def reset(self):
        """重置状态"""
        self.joints = [0.823, 0.0, 0.0, -1.152, 2.368, 13.309]   # 6个关节
        self.gripper = 0.0  # 夹爪状态

# 使用示例
if __name__ == "__main__":
    piper_leader = PiperLeader()
    try:
        while True:
            piper_leader.update_joints()
            print(piper_leader.get_action())
            time.sleep(0.1)
    except KeyboardInterrupt:
        piper_leader.stop()