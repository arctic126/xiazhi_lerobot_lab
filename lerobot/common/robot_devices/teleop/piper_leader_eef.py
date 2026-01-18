from typing import Dict
import time
from piper_sdk import C_PiperInterface_V2

class PiperLeader:
    def __init__(self):
        # 初始化机械臂接口
        self.piper = C_PiperInterface_V2("can_right")
        self.piper.ConnectPort()
        
        # 初始化关节和夹爪状态
        self.joints = [42.087, -0.028, 159.877, 144.797, 55.171, 139.970, 0.0]  # 6个关节
        self.gripper = 0  # 夹爪状态，0表示关闭，1表示打开
        
        # 设置运行标志
        self.running = True
    
    def update_joints(self):
        """更新关节状态，从主臂读取数据"""
        # 获取主臂关节信息
        eef_pose_ctrl = self.piper.GetFK('control')[-1]
        master_gripper_info = self.piper.GetArmGripperCtrl()
        # print('master_joint_info', master_joint_info)
        current_eef_pose = self.piper.GetFK('feedback')[-1]
        
        if eef_pose_ctrl is not None:
            
            eef_pose = [
                eef_pose_ctrl[0] - current_eef_pose[0],  
                eef_pose_ctrl[1] - current_eef_pose[1],
                eef_pose_ctrl[2] - current_eef_pose[2],
                eef_pose_ctrl[3] - current_eef_pose[3],
                eef_pose_ctrl[4] - current_eef_pose[4],
                eef_pose_ctrl[5] - current_eef_pose[5] 
            ]
            
            for i in range(6):
                self.joints[i] = eef_pose[i]
        
        if master_gripper_info is not None:
            # 夹爪角度转换为二进制状态：0(关闭)或1(打开)
            gripper_angle = master_gripper_info.gripper_ctrl.grippers_angle
            # 如果夹爪角度大于设定阈值，则认为是打开状态
            self.gripper = 1.0 if gripper_angle > 30000 else 0.0
    
    def get_action(self) -> Dict:
        """返回机械臂的当前状态"""

        self.update_joints()
        return {
            'x': self.joints[0],
            'y': self.joints[1],
            'z': self.joints[2],
            'roll': self.joints[3],
            'yaw': self.joints[4],
            'pitch': self.joints[5],
            'gripper': self.gripper
        }
    
    def stop(self):
        """停止运行"""
        self.running = False
        print("Piper leader exits")

    def reset(self):
        """重置状态"""
        self.joints = [42.087, -0.028, 159.877, 144.797, 55.171, 139.970, 0.0]  # 6个关节
        self.gripper = 0  # 夹爪状态，关闭

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