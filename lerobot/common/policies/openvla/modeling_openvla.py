# lerobot/common/policies/openvla/modeling_openvla.py

from typing import Dict
import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from huggingface_hub import PyTorchModelHubMixin
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.utils.utils import get_safe_torch_device
from PIL import Image
import numpy as np  # 确保numpy导入可用

class OpenVLAPolicy(nn.Module, PyTorchModelHubMixin):
    """LeRobot框架的OpenVLA策略封装"""
    
    name = "openvla"

    def __init__(self, config, dataset_stats=None, device=None):
        super().__init__()
        self.config = config

        # 定义SO-100机器人的动作范围
        # 这些值应该根据您的机器人规格进行调整
        self.action_ranges = {
            'position': {  
                'min': np.array([-0.02, -0.02, -0.02]),  # 每步最大2cm移动
                'max': np.array([0.02, 0.02, 0.02])      
            },
            'orientation': {  
                'min': np.array([-0.1, -0.1, -0.1]),     # 每步最大约5.7度旋转
                'max': np.array([0.1, 0.1, 0.1])      
            },
            'gripper': {
                'min': 0.0,  # 闭合
                'max': 1.0   # 打开
            }
        }
        
        # 使用传入的设备或者默认为CPU
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
            
        # 检查是否支持bfloat16
        self.use_bfloat16 = True
        if self.device.type == 'cuda':
            # 检查CUDA设备是否支持bfloat16
            if not torch.cuda.is_bf16_supported():
                print("警告: 当前CUDA设备不支持bfloat16，将使用float32代替")
                self.use_bfloat16 = False
        elif self.device.type == 'cpu':
            # CPU一般支持bfloat16但效率不高
            if not hasattr(torch, 'bfloat16'):
                print("警告: 当前PyTorch版本不支持CPU上的bfloat16，将使用float32代替")
                self.use_bfloat16 = False
                
        # 确定要使用的dtype
        self.dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
        print(f"使用数据类型: {self.dtype}")

        # 初始化OpenVLA模型和处理器
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )

        # 加载模型
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.model.to(self.device)


    def reset(self):
        """环境重置时调用"""
        pass

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """训练前向传递 - OpenVLA未实现，因为我们使用预训练模型"""
        raise NotImplementedError("OpenVLA策略仅用于推理，不支持训练")

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        将OpenVLA归一化动作转换为SO-100机器人范围
        normalized_action: 形状为(7,)的数组，范围在[-1, 1]
        返回: 形状为(7,)的数组，范围在机器人实际范围内
        """

        # 添加安全检查
        if np.any(np.abs(normalized_action) > 1.0):
            print("警告: 输入动作超出[-1,1]范围:", normalized_action)
            normalized_action = np.clip(normalized_action, -1.0, 1.0)

        # 分割成各组成部分
        pos_norm = normalized_action[:3]
        rot_norm = normalized_action[3:6]
        grip_norm = normalized_action[6]

        # 解除位置归一化 (从[-1,1]到实际位置增量)
        pos_range = self.action_ranges['position']
        pos_unnorm = (
            pos_norm + 1  # 从[-1,1]到[0,2]
        ) / 2 * (pos_range['max'] - pos_range['min']) + pos_range['min']

        # 解除方向归一化 (从[-1,1]到实际角度增量)
        rot_range = self.action_ranges['orientation']
        rot_unnorm = (
            rot_norm + 1
        ) / 2 * (rot_range['max'] - rot_range['min']) + rot_range['min']

        # 解除抓取器归一化 (从[-1,1]到[0,1])
        grip_range = self.action_ranges['gripper']
        grip_unnorm = (grip_norm + 1) / 2 * (grip_range['max'] - grip_range['min']) + grip_range['min']

        # 确保grip_unnorm是数组而不是标量
        grip_unnorm_array = np.array([grip_unnorm])
        
        # 打印调试信息
        # print(f"位置解归一化形状: {pos_unnorm.shape}, 类型: {pos_unnorm.dtype}")
        # print(f"方向解归一化形状: {rot_unnorm.shape}, 类型: {rot_unnorm.dtype}")
        # print(f"抓取器解归一化形状: {grip_unnorm_array.shape}, 类型: {grip_unnorm_array.dtype}")

        # 将位置、方向和抓取器动作组合成最终动作
        action = np.concatenate([pos_unnorm, rot_unnorm, grip_unnorm_array])
        
        return action

    @torch.inference_mode()
    def select_action(self, observation: Dict[str, Tensor]) -> Tensor:
        """Run inference to select next action"""
        # 打印观察字典内容和结构
        print("\nObservation Dictionary内容:")
        for key, value in observation.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"{key}: 嵌套字典，包含键 {list(value.keys())}")
            else:
                print(f"{key}: type={type(value)}")

        # 查找图像数据
        image_tensor = None
        image_key = None
        
        # 特殊处理常见的 observation.images.primary 格式
        if "images" in observation and isinstance(observation["images"], dict):
            if "primary" in observation["images"]:
                image_tensor = observation["images"]["primary"]
                image_key = "images.primary"
                print(f"找到图像: images.primary")
        
        # 如果上面没找到，尝试其他方法
        if image_tensor is None:
            # 检查顶层中带有'image'的键
            for key in observation.keys():
                if "image" in key.lower() and isinstance(observation[key], torch.Tensor):
                    image_tensor = observation[key]
                    image_key = key
                    print(f"找到图像: {key}")
                    break
            
            # 检查嵌套字典
            if image_tensor is None:
                for key, value in observation.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if ("image" in sub_key.lower() or "camera" in sub_key.lower()) and isinstance(sub_value, torch.Tensor):
                                image_tensor = sub_value
                                image_key = f"{key}.{sub_key}"
                                print(f"找到图像: {key}.{sub_key}")
                                break
        
        if image_tensor is None:
            raise ValueError(f"无法在观察中找到图像数据。观察键: {list(observation.keys())}")

        # 使用任务指令格式化提示
        prompt = f"In: What action should the robot take to {self.config.instruction}?\nOut:"
        print(f"提示: {prompt}, 图像键: {image_key}")
        
        # 检查图像形状
        print(f"原始图像形状: {image_tensor.shape}")
        
        # 处理图像
        # 如果图像已经是[H, W, C]格式且是uint8类型，直接转换为PIL
        if len(image_tensor.shape) == 3 and image_tensor.shape[2] == 3 and image_tensor.dtype == torch.uint8:
            # 直接转换为PIL图像，不需要额外处理
            image_np = image_tensor.cpu().numpy()
            image_pil = Image.fromarray(image_np)
        else:
            # 如果有batch维度，移除它
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor.squeeze(0)
                
            # 如果通道在前面，调整为通道在后面
            if image_tensor.shape[0] == 3:
                image_tensor = image_tensor.permute(1, 2, 0)
                
            # 如果是浮点数据类型，缩放到0-255
            if image_tensor.dtype == torch.float32 or image_tensor.dtype == torch.float64:
                image_np = (image_tensor.cpu().numpy() * 255).astype('uint8')
            else:
                image_np = image_tensor.cpu().numpy()
                
            try:
                image_pil = Image.fromarray(image_np)
            except Exception as e:
                print(f"创建PIL图像时出错: {e}")
                print(f"图像数组形状: {image_np.shape}, 数据类型: {image_np.dtype}")
                print(f"图像数据范围: min={image_np.min()}, max={image_np.max()}")
                # 尝试转换为RGB
                if len(image_np.shape) == 3 and image_np.shape[2] == 1:
                    # 转换单通道为RGB
                    image_np = np.repeat(image_np, 3, axis=2)
                    image_pil = Image.fromarray(image_np)
                else:
                    raise
        
        print(f"PIL图像大小: {image_pil.size}")

        # 准备模型输入并确保类型正确
        inputs = self.processor(prompt, image_pil)
        
        # 将input_ids转换为long类型并移动到设备上
        inputs["input_ids"] = inputs["input_ids"].long().to(self.device)
        
        # 将attention_mask转换到适当的类型和设备
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
            
        # 将pixel_values转换为相应数据类型并移动到设备
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype, device=self.device)

        # 从OpenVLA获取归一化的动作
        # 预测动作
        actions = self.model.predict_action(**inputs, unnorm_key="tape_eef", do_sample=False)
        
        print("原始归一化动作:", actions)  # 应该在[-1, 1]范围内

        # 为SO-100解除归一化
        # unnorm_actions = self.unnormalize_action(actions)
        #print("解除归一化后的SO-100动作:", unnorm_actions)

        # 将numpy数组转换为torch张量
        actions_tensor = torch.from_numpy(actions)

        print("\n动作张量:", actions_tensor)
        # print("动作形状:", actions_tensor.shape)
        # print("动作数据类型:", actions_tensor.dtype)

        return actions_tensor

    def to(self, device):
        """Override to() to handle device movement"""
        super().to(device)
        self.device = device
        self.model.to(device)
        return self