"""
JAKA机器人电机控制封装 - 末端位姿控制版本
基于JAKA Python SDK (jkrc)
使用servo_p进行末端位姿控制（高速响应模式）
"""
import time
from typing import Dict
import jkrc
from lerobot.common.robot_devices.motors.configs import JakaMotorsBusConfig


class JakaMotorsBus:
    """
    对JAKA SDK的二次封装，用于LeRobot集成
    使用servo_p进行末端位姿控制（高速响应模式）
    """
    def __init__(self, config: JakaMotorsBusConfig):
        self.robot_ip = config.robot_ip
        self.end_effector_dof = config.end_effector_dof
        
        # 初始化JAKA机器人对象
        self.robot = jkrc.RC(self.robot_ip)
        
        # 初始末端位姿（单位：mm, rad）
        # 格式：[x, y, z, rx, ry, rz, gripper]
        # 可以根据实际需求调整这些值
        self.init_tcp_position = [400.0, 0.0, 300.0, 3.14, 0.0, 0.0, 0.0]
        self.safe_disable_position = [400.0, 0.0, 300.0, 3.14, 0.0, 0.0, 0.0]
        
        # JAKA没有夹爪，gripper值固定为0
        self.gripper_value = 0.0
        
        self.is_connected = False

    @property
    def motor_names(self) -> list[str]:
        """返回末端自由度名称"""
        return list(self.end_effector_dof.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.end_effector_dof.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.end_effector_dof.values()]

    def connect(self, enable: bool) -> bool:
        """
        连接并使能/去使能机器人
        
        Args:
            enable: True为上电上使能，False为下使能下电
            
        Returns:
            bool: 连接是否成功
        """
        try:
            if enable:
                # 登录
                ret = self.robot.login()
                if ret[0] != 0:
                    print(f"登录失败，错误码: {ret[0]}")
                    return False
                print("机器人登录成功")
                
                # 上电
                ret = self.robot.power_on()
                if ret[0] != 0:
                    print(f"上电失败，错误码: {ret[0]}")
                    return False
                print("机器人上电成功")
                time.sleep(1)  # 等待上电稳定
                
                # 使能
                ret = self.robot.enable_robot()
                if ret[0] != 0:
                    print(f"使能失败，错误码: {ret[0]}")
                    return False
                print("机器人使能成功")
                
                # 🔧 FIX: 强制退出servo模式（清理可能残留的状态）
                # 如果机器人之前在servo模式中异常退出，需要先清理状态
                print("清理servo模式状态...")
                ret = self.robot.servo_move_enable(False)
                # 忽略错误码（可能本来就不在servo模式）
                print("已确保退出servo模式")
                
                # 🔧 FIX: 设置滤波器参数必须在进入servo模式之前！
                # 设置笛卡尔空间非线性滤波器来控制速度
                ret = self.robot.servo_move_use_carte_NLF(
                    max_vp=20,    # 线速度上限 mm/s
                    max_ap=100,   # 加速度上限 mm/s²
                    max_jp=500,   # 加加速度上限 mm/s³
                    max_vr=1.0,   # 角速度上限 rad/s
                    max_ar=5.0,   # 角加速度上限 rad/s²
                    max_jr=25.0   # 角加加速度上限 rad/s³
                )
                if ret[0] != 0:
                    print(f"设置滤波器失败，错误码: {ret[0]}")
                    return False
                print("已设置速度滤波器: 线速度≤20mm/s, 角速度≤1.0rad/s")
                
                # 进入servo模式（必须在滤波器配置之后）
                ret = self.robot.servo_move_enable(True)
                if ret[0] != 0:
                    print(f"进入servo模式失败，错误码: {ret[0]}")
                    return False
                print("已进入servo_p模式")
                
                self.is_connected = True
                return True
            else:
                # 退出servo模式
                ret = self.robot.servo_move_enable(False)
                if ret[0] != 0:
                    print(f"退出servo模式失败，错误码: {ret[0]}")
                print("已退出servo_p模式")
                
                # 下使能
                ret = self.robot.disable_robot()
                if ret[0] != 0:
                    print(f"下使能失败，错误码: {ret[0]}")
                
                # 下电
                ret = self.robot.power_off()
                if ret[0] != 0:
                    print(f"下电失败，错误码: {ret[0]}")
                
                # 登出
                ret = self.robot.logout()
                if ret[0] != 0:
                    print(f"登出失败，错误码: {ret[0]}")
                
                self.is_connected = False
                return True
                
        except Exception as e:
            print(f"连接过程发生异常: {e}")
            return False

    def apply_calibration(self):
        """
        移动到初始末端位姿（校准位置）
        注意：servo_p模式下不适合用于校准，暂时跳过
        """
        if not self.is_connected:
            raise ConnectionError("机器人未连接")
        
        print("servo_p模式下跳过校准移动")
        print("提示：请在使用前手动将机器人移动到合适位置")

    def read(self) -> Dict:
        """
        读取机器人当前末端位姿 + gripper（7维）
        
        Returns:
            dict: 末端位姿字典，键为自由度名称 [x, y, z, yaw, pitch, roll, gripper]
                  值的单位：位置(mm)，姿态(rad)，gripper(固定为0)
        """
        if not self.is_connected:
            raise ConnectionError("机器人未连接")
        
        ret = self.robot.get_tcp_position()
        
        if ret[0] != 0:
            print(f"读取末端位姿失败，错误码: {ret[0]}")
            # 返回零值字典
            return {name: 0.0 for name in self.motor_names}
        
        # ret[1] 是包含6个元素的元组: (x, y, z, rx, ry, rz)
        tcp_pos = ret[1]
        
        # 直接使用JAKA的格式 [x, y, z, rx, ry, rz, gripper]
        # 不做名称转换，保持与模型训练时一致
        result = {
            "x": tcp_pos[0],        # x (mm)
            "y": tcp_pos[1],        # y (mm)
            "z": tcp_pos[2],        # z (mm)
            "rx": tcp_pos[3],       # rx (rad)
            "ry": tcp_pos[4],       # ry (rad)
            "rz": tcp_pos[5],       # rz (rad)
            "gripper": self.gripper_value  # 固定为0（JAKA无夹爪）
        }
        
        return result

    def write(self, target_pose: list):
        """
        发送目标末端位姿到机器人（使用servo_p高速位置控制）
        
        Args:
            target_pose: 目标末端位姿列表 [x, y, z, rx, ry, rz, gripper]
                        单位：位置(mm)，姿态(rad)，gripper被忽略（JAKA无夹爪）
        """
        if not self.is_connected:
            raise ConnectionError("机器人未连接")
        
        if len(target_pose) != 7:
            raise ValueError(f"期望7个参数（含gripper），但收到{len(target_pose)}个")
        
        # 输入已经是mm单位，直接使用前6维
        # gripper值被忽略（JAKA无夹爪）
        jaka_pose = target_pose[:6]
        
        # 🔍 调试输出
        print(f"\n[DEBUG write]:")
        print(f"  发送到JAKA (mm,rad): [{jaka_pose[0]:.2f}, {jaka_pose[1]:.2f}, {jaka_pose[2]:.2f}, {jaka_pose[3]:.4f}, {jaka_pose[4]:.4f}, {jaka_pose[5]:.4f}]")
        
        # 使用servo_p发送目标位姿（绝对位置模式）
        # 参数：end_pos, move_mode(0=绝对运动), step_num(倍分周期)
        # servo_p运动周期为step_num*8ms，step_num=1时周期为8ms（最快）
        ret = self.robot.servo_p(
            end_pos=jaka_pose,  # 修正参数名：cartesian_pose -> end_pos
            move_mode=0,        # 绝对运动
            step_num=1          # 最小周期8ms，最快响应
        )
        
        # 🔍 调试输出返回值
        print(f"  servo_p返回: {ret}")
        if ret[0] != 0:
            print(f"  ❌ 发送失败！错误码: {ret[0]}")
        else:
            print(f"  ✓ 发送成功")

    def safe_disconnect(self):
        """
        安全断开：先移动到安全位置，然后断开连接
        """
        if not self.is_connected:
            return
        
        # 断开连接
        self.connect(enable=False)
