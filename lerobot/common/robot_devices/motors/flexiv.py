import time
from typing import Dict
import flexivrdk
from lerobot.common.robot_devices.motors.configs import FlexivMotorsBusConfig
from lerobot.common.robot_devices.robots.utils import quat2eulerZYX, eulerZYX2quat
from scipy.spatial.transform import Rotation as R

class FlexivMotorsBus:
    """
        For flexiv_rdk v1.7
    """
    def __init__(self, 
                 config: FlexivMotorsBusConfig):
        self.robot = flexivrdk.Robot(config.serial_port)
        self.gripper = flexivrdk.Gripper(self.robot)
        self.tool = flexivrdk.Tool(self.robot)
        self.motors = config.motors
        self.gripper_name = config.gripper_name

        # External TCP force threshold for collision detection, value is only for demo purpose [N]
        EXT_FORCE_THRESHOLD = 10.0

        # External joint torque threshold for collision detection, value is only for demo purpose [Nm]
        EXT_TORQUE_THRESHOLD = 5.0

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
            使能机械臂并检测使能状态。如果使能超时则退出程序。
            
            Args:
                enable (bool): True表示使能机械臂，False表示失能机械臂
                
            Returns:
                bool: 使能/失能是否成功
        '''
        try:
            if enable:
                # Clear fault on the connected robot if any
                if self.robot.fault():
                    print("Fault occurred on the connected robot, trying to clear ...")
                    # Try to clear the fault
                    if not self.robot.ClearFault():
                        print("Fault cannot be cleared, exiting ...")
                        return 1
                    print("Fault on the connected robot is cleared")

                 # Enable the robot, make sure the E-stop is released before enabling
                print("Enabling robot ...")
                self.robot.Enable()

                print(f"Enabling gripper [{self.gripper_name}]")
                self.gripper.Enable(self.gripper_name)

                print(f"Switching robot tool to [{self.gripper_name}]")
                self.tool.Switch(self.gripper_name)

                # Wait for the robot to become operational
                while not self.robot.operational():
                    time.sleep(1)

                print("Robot is now operational")

                # Zero Force-torque Sensor
                # =========================================================================================
                self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
                # IMPORTANT: must zero force/torque sensor offset for accurate force/torque measurement
                self.robot.ExecutePrimitive("ZeroFTSensor", dict())

                # WARNING: during the process, the robot must not contact anything, otherwise the result
                # will be inaccurate and affect following operations
                print(
                    "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
                )

                # Wait for primitive to finish
                while not self.robot.primitive_states()["terminated"]:
                    time.sleep(1)
                print("Sensor zeroing complete")
                return True

            else:
                self.robot.SwitchMode(flexivrdk.Mode.NRT_IDLE)
                print("Robot is now idle")
                self.gripper.Stop()
                return True
                
        except Exception as e:
            print(f"发生错误: {str(e)}")
            return False
    
    def motor_names(self):
        return

    def set_calibration(self):
        return
    
    def revert_calibration(self):
        return

    def apply_calibration(self):
        """
        移动到初始位置 (Home position) 并打开夹爪
        """
        try:
            # 切换到 primitive execution 模式
            self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
            
            print("正在移动到初始位置...")
            # 执行 Home primitive
            self.robot.ExecutePrimitive("Home", dict())
            
            # Wait for reached target
            # Note: primitive_states() returns a dictionary of {pt_state_name, [pt_state_values]}
            while not robot.primitive_states()["reachedTarget"]:
                time.sleep(1)
                
            print("已到达初始位置")
            
            # 打开夹爪到最大位置
            print("正在打开夹爪...")
            self.gripper.Move(0.09, 0.1, 20)
            time.sleep(2)  # 等待夹爪动作完成
            print("夹爪已打开")
            
            return True
            
        except Exception as e:
            print(f"移动到初始位置时发生错误: {str(e)}")
            return False

    def write(self, target_tcp_delta: list):
        """
        通过TCP pose增量控制机器人位置和姿态
        
        Args:
            target_tcp_delta (list): 目标TCP位姿的增量和夹爪状态
                [dx, dy, dz, drx, dry, drz, gripper]
                - dx,dy,dz: TCP位置的增量 (单位: 米，需要除以20)
                - drx,dry,drz: TCP姿态的增量 (单位: 弧度，需要除以20)
                - gripper: 夹爪状态 (0.0: 关闭, 1.0: 打开)
        """
        try:
            # 获取当前机器人状态
            robot_states = self.robot.states()
            current_tcp = robot_states.tcp_pose  # [x,y,z,qw,qx,qy,qz]
            
            current_p = current_tcp[:3]
            current_q = R.from_quat([current_tcp[4], current_tcp[5], current_tcp[6], current_tcp[3]])

            delta_p = target_tcp_delta[:3]
            delta_q = R.from_euler('xyz', target_tcp_delta[3:6])

            # calculate new tcp pose
            new_p = current_p + delta_p
            new_q = delta_q * current_q
            new_quat = new_q.as_quat()
            # convert to [qw,qx,qy,qz]
            new_quat = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

            new_tcp = np.concatenate([new_p, new_quat])
            
            # switch to cartesian motion force mode
            self.robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
            # Set all Cartesian axis(s) to motion control
            self.robot.SetForceControlAxis([False, False, False, False, False, False])
            # send command
            self.robot.SendCartesianMotionForce(new_tcp)
            
            # 控制夹爪（将二值状态转换为实际位置）
            gripper_width = 0.09 if target_tcp_delta[6] > 0.5 else 0.01  # 大于0.5认为是打开状态
            self.gripper.Move(gripper_width, 0.1, 20)
            
            return True
            
        except Exception as e:
            print(f"TCP控制时发生错误: {str(e)}")
            return False

    def read(self) -> Dict:
        """
        读取机器人末端位姿状态
        
        Returns:
            Dict: 包含末端位姿和夹爪状态的字典
                - x,y,z: TCP在基坐标系中的位置 (单位: 米)
                - rx,ry,rz: TCP在基坐标系中的欧拉角(xyz顺序) (单位: 弧度)
                - gripper: 夹爪状态 (0: 关闭, 1: 打开)
        """
        try:
            # 获取机器人状态
            robot_states = self.robot.states()
            # 获取当前TCP位姿 [x,y,z,qw,qx,qy,qz]
            current_tcp = robot_states.tcp_pose
            
            # 提取位置
            p = current_tcp[:3]
            
            # 将四元数转换为欧拉角
            q = quat2eulerZYX(current_tcp[3:7])
            
            # 获取夹爪状态
            gripper_width = self.gripper.states().width
            
            # convert to binary state
            gripper_state = 1.0 if gripper_width > 0.08 else 0.0
            
            return {
                "x": p[0],
                "y": p[1],
                "z": p[2],
                "rx": q[0],
                "ry": q[1],
                "rz": q[2],
                "gripper": gripper_state  # 返回二值状态
            }
            
        except Exception as e:
            print(f"读取机器人状态时发生错误: {str(e)}")
            return {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "rx": 0.0,
                "ry": 0.0,
                "rz": 0.0,
                "gripper": 0.0  # 错误时默认返回关闭状态
            }
    
    def safe_disconnect(self):
        """ 
            Move to safe disconnect position
        """
        self.set_calibration()