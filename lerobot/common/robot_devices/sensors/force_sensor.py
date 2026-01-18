"""
Force Sensor Device Interface

封装PyForce库，提供6轴力/力矩传感器接口
适用于宇立(Sunrise)力传感器
"""

import numpy as np
from typing import Optional
import time

try:
    from pyforce import ForceSensor as PyForceSensor
    PYFORCE_AVAILABLE = True
except ImportError:
    PYFORCE_AVAILABLE = False
    PyForceSensor = None


class ForceSensor:
    """6轴力/力矩传感器封装（宇立/Sunrise）"""
    
    def __init__(self, ip_addr: str = "192.168.0.108", port: int = 4008):
        """
        初始化力传感器
        
        Args:
            ip_addr: 传感器IP地址
            port: 传感器端口号
        """
        if not PYFORCE_AVAILABLE:
            raise ImportError(
                "PyForce未安装。请安装: pip install git+https://github.com/Elycyx/PyForce.git"
            )
        
        self.ip_addr = ip_addr
        self.port = port
        self.sensor = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """
        连接力传感器
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 创建传感器实例
            self.sensor = PyForceSensor(ip_addr=self.ip_addr, port=self.port)
            
            # 连接
            if not self.sensor.connect():
                print(f"连接力传感器失败: {self.ip_addr}:{self.port}")
                return False
            
            # 启动数据流
            if not self.sensor.start_stream():
                print("启动力传感器数据流失败")
                self.sensor.disconnect()
                return False
            
            # 等待数据流稳定
            time.sleep(0.5)
            
            self.is_connected = True
            print(f"力传感器已连接: {self.ip_addr}:{self.port}")
            return True
            
        except Exception as e:
            print(f"连接力传感器时发生异常: {e}")
            if self.sensor:
                self.sensor.disconnect()
            return False
    
    def zero(self, num_samples: int = 100) -> bool:
        """
        力传感器校零（去除初始偏置）
        
        Args:
            num_samples: 用于计算偏置的样本数
            
        Returns:
            bool: 校零是否成功
        """
        if not self.is_connected:
            print("力传感器未连接，无法校零")
            return False
        
        try:
            if self.sensor.zero(num_samples=num_samples):
                print(f"力传感器已校零（使用{num_samples}个样本）")
                return True
            else:
                print("力传感器校零失败")
                return False
        except Exception as e:
            print(f"力传感器校零时发生异常: {e}")
            return False
    
    def read(self) -> Optional[np.ndarray]:
        """
        读取力/力矩数据
        
        Returns:
            np.ndarray: 6维数据 [fx, fy, fz, mx, my, mz]，失败返回None
        """
        if not self.is_connected:
            print("力传感器未连接")
            return None
        
        try:
            # 使用get()方法获取最新数据
            data = self.sensor.get()
            
            if data is None or 'ft' not in data:
                # 降级到read()方法
                force_data = self.sensor.read()
                if force_data is None:
                    return None
            else:
                force_data = data['ft']
            
            return force_data.astype(np.float32)
            
        except Exception as e:
            print(f"读取力传感器数据时发生异常: {e}")
            return None
    
    def disconnect(self) -> bool:
        """
        断开力传感器连接
        
        Returns:
            bool: 断开是否成功
        """
        try:
            if self.sensor is not None:
                # 停止数据流
                self.sensor.stop_stream()
                # 断开连接
                self.sensor.disconnect()
                self.sensor = None
            
            self.is_connected = False
            print("力传感器已断开")
            return True
            
        except Exception as e:
            print(f"断开力传感器时发生异常: {e}")
            return False
