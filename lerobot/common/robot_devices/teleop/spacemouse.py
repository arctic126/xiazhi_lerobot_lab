# Copyright (c) 2025 yihuai
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from robotmq import RMQClient
import numpy as np
import numpy.typing as npt
import time
from typing import Dict
import threading


class SpacemouseClient:
    def __init__(
        self,
        rmq_server_address: str = "tcp://localhost:15557",
        connection_timeout_s: float = 1.0,
    ):
        self.rmq_client = RMQClient("spacemouse_client", rmq_server_address)
        connect_start_time = time.time()
        print(f"Connecting to spacemouse server at {rmq_server_address}")
        self.get_latest_state()
        print("Spacemouse server connected")
        # while time.time() - connect_start_time < connection_timeout_s:
        #     raw_data, _ = self.rmq_client.peek_data("spacemouse_state", "latest", 1)
        #     if len(raw_data) > 0:
        #         break
        #     time.sleep(0.01)
        # else:
        #     raise RuntimeError(f"Failed to connect to spacemouse server in {connection_timeout_s} seconds. Please check if the spacemouse server is running.")

    def get_latest_state(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raw_data, timestamp = self.rmq_client.peek_data("spacemouse_state", "latest", 1)
        spacemouse_state = np.frombuffer(raw_data[0], dtype=np.float64)
        return spacemouse_state[:6], spacemouse_state[6:]

    def get_average_state(
        self, n: int, average_buttons: bool = False
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raw_data, timestamps = self.rmq_client.peek_data(
            "spacemouse_state", "latest", n
        )
        spacemouse_states = np.array(
            [np.frombuffer(data, dtype=np.float64) for data in raw_data]
        )
        if average_buttons:
            return np.mean(spacemouse_states[:, :6], axis=0), np.mean(
                spacemouse_states[:, 6:], axis=0
            )
        else:
            return np.mean(spacemouse_states[:, :6], axis=0), spacemouse_states[0, 6:]


class SpaceMouseController:
    def __init__(self):
        self.spacemouse_client = SpacemouseClient()
        
        self.running = True
        self.thread = threading.Thread(target=self.update_joints)
        self.thread.start()
        self.gripper = -1
        self.x = 0
        self.y = 0
        self.z = 0
        self.rx = 0
        self.ry = 0
        self.rz = 0
        # 添加一个变量来跟踪上一次的gripper状态
        self.prev_gripper_button = 0
        
    
    def update_joints(self):
        while self.running:
                
            input = self.spacemouse_client.get_latest_state()
            
            [x, y, z, rx, ry, rz] = input[0]
            gripper_button = input[1][0]
            
            # 直接使用硬编码的阈值
            threshold = 0.1
            # 直接使用硬编码的factor
            factor = 12
            
            # 应用阈值滤波
            x = 0 if abs(x) < threshold else -2*x
            y = 0 if abs(y) < threshold else -2*y
            z = 0 if abs(z) < threshold else 2*z
            rx = 0 if abs(rx) < threshold else 0.5*rx
            ry = 0 if abs(ry) < threshold else 0.5*ry
            rz = 0 if abs(rz) < threshold else 0.5*rz
            
            # 映射输入到速度
            self.x = x*factor
            self.y = y*factor
            self.z = z*factor
            self.rx = rx*factor
            self.ry = ry*factor
            self.rz = rz*factor
            # 只有当gripper按钮从0变为1时才改变gripper的值
            if gripper_button == 1 and self.prev_gripper_button == 0:
                self.gripper = -1 * self.gripper
            
            # 更新上一次的按钮状态
            self.prev_gripper_button = gripper_button
            
            # 控制更新频率
            time.sleep(0.02)
    
    def get_action(self) -> Dict:
        # 返回机械臂的当前状态
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'rx': self.rx,
            'ry': self.ry,
            'rz': self.rz,
            'gripper': self.gripper
        }
    
    def stop(self):
        # 停止更新线程
        self.running = False
        self.thread.join()
        print("SpaceMouse exits")

    def reset(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.rx = 0
        self.ry = 0
        self.rz = 0
        self.gripper = -1

if __name__ == "__main__":
    controller = SpaceMouseController()
    try:
        while True:
            print(controller.get_action())
            time.sleep(0.02)
    except KeyboardInterrupt:
        controller.stop()