import time
from typing import Dict
from piper_sdk import *
from lerobot.common.robot_devices.motors.configs import PiperEef2MotorsBusConfig

class PiperEef2MotorsBus:
    """
        对Piper SDK的二次封装
    """
    def __init__(self, 
                 config: PiperEef2MotorsBusConfig):
        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()
        self.motors = config.motors
        self.init_eef_position = [42.087, -0.028, 159.877, 144.797, 55.171, 139.970, 0.0] # [6 eef pose + 1 gripper] (0.001度)
        self.safe_disable_position = [42.087, -0.028, 159.877, 144.797, 55.171, 139.970, 0.0]
        self.pose_factor = 1000 # 单位 0.001mm
        self.joint_factor = 1000 # 1000*180/3.14， rad -> 度（单位0.001度）

        '''
        time stamp:1743841259.096878
        Hz:200.0
        ArmMsgEndPoseFeedBack:
         X_axis : 49424, 49.424
        Y_axis : 1459, 1.459
        Z_axis : 159507, 159.507
        RX_axis : 138068, 138.068
        RY_axis : 64283, 64.283
        RZ_axis : 136684, 136.684


        '''

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]


    def connect(self, enable:bool) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        loop_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print(f"--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if(enable):
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0,1000,0x01, 0)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0,1000,0x02, 0)
            print(f"使能状态: {enable_flag}")
            print(f"--------------------")
            if(enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else: 
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print(f"超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        print(f"Returning response: {resp}")
        return resp
    
    def motor_names(self):
        return

    def set_calibration(self):
        return
    
    def revert_calibration(self):
        return

    def apply_calibration(self):
        """
            移动到初始位置
        """
        self.write(target_eef=self.init_eef_position)

    def write(self, target_eef:list):
        """
            eef control
            - target joint
        """
        # current_eef_pose = self.piper.GetFK('feedback')[-1]

        eef_x = round(target_eef[0]*self.pose_factor)
        eef_y = round(target_eef[1]*self.pose_factor)
        eef_z = round(target_eef[2]*self.pose_factor)
        eef_rx = round(target_eef[3]*self.pose_factor)
        eef_ry = round(target_eef[4]*self.pose_factor)
        eef_rz = round(target_eef[5]*self.pose_factor)
        gripper_state = target_eef[6]  # 0表示关闭，1表示打开
        
        # print('eef_x', eef_x)
        # print('eef_y', eef_y)
        # print('eef_z', eef_z)
        # print('eef_rx', eef_rx)
        # print('eef_ry', eef_ry)
        # print('eef_rz', eef_rz)
        # print('gripper_state', gripper_state)
        # Comment out the following lines when teleop is used
        # -------------------------------------------------------------------------------------------
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00) # eef control
        self.piper.EndPoseCtrl(eef_x, eef_y, eef_z, eef_rx, eef_ry, eef_rz)
        # 如果gripper是1（打开），则设置夹爪角度为90000（90度），否则设置为0（关闭）
        gripper_angle = 85000 if gripper_state >= 0.5 else 0
        self.piper.GripperCtrl(gripper_angle, 1000, 0x01, 0) # 单位 0.001°
        # -------------------------------------------------------------------------------------------
    

    def read(self) -> Dict:
        """
            - 机械臂eef消息,单位0.001度
            - 机械臂夹爪消息
        """
        eef_msg = self.piper.GetFK('feedback')[-1]

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state

        
        return {
            "x": eef_msg[0],
            "y": eef_msg[1],
            "z": eef_msg[2],
            "roll": eef_msg[3],
            "yaw": eef_msg[4],
            "pitch": eef_msg[5],
            "gripper": 1.0 if gripper_state.grippers_angle > 45000 else 0.0  # 夹爪状态：0表示关闭，1表示打开
        }
    
    def safe_disconnect(self):
        """ 
            Move to safe disconnect position
        """
        self.write(target_eef=self.safe_disable_position)