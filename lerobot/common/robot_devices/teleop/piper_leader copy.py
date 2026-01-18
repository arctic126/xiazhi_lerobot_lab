from typing import Dict
import time
from piper_sdk import C_PiperInterface_V2

class PiperLeader:
    def __init__(self):
        # 初始化机械臂接口
        self.piper = C_PiperInterface_V2("can0")
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
        
        if master_joint_info is not None:
            
            joint_angles = [
                master_joint_info.joint_ctrl.joint_1 / 1000,  
                master_joint_info.joint_ctrl.joint_2 / 1000,
                master_joint_info.joint_ctrl.joint_3 / 1000,
                master_joint_info.joint_ctrl.joint_4 / 1000,
                master_joint_info.joint_ctrl.joint_5 / 1000,
                master_joint_info.joint_ctrl.joint_6 / 1000
            ]
            
            for i in range(6):
                self.joints[i] = joint_angles[i]
        
        if master_gripper_info is not None:
            # 夹爪角度单位为0.001度，需要映射到0~0.08的范围
            gripper_angle = master_gripper_info.gripper_ctrl.grippers_angle
            # 假设夹爪角度范围是0-90000（90度，单位0.001度），线性映射到0-0.08
            self.gripper = (gripper_angle / 1000)
    
    def get_action(self) -> Dict:
        """返回机械臂的当前状态"""

        self.update_joints()
        return {
            'joint0': self.joints[0],
            'joint1': self.joints[1],
            'joint2': self.joints[2],
            'joint3': self.joints[3],
            'joint4': self.joints[4],
            'joint5': self.joints[5],
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