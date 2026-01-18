# lerobot/common/policies/openpi/modeling_openpi.py

from typing import Dict
import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from huggingface_hub import PyTorchModelHubMixin
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.utils.utils import get_safe_torch_device
from PIL import Image
from collections import deque
import numpy as np  # 确保numpy导入可用

from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).


class OpenPiPolicy(nn.Module, PyTorchModelHubMixin):
    """LeRobot框架的OpenPi策略封装"""
    
    name = "openpi"

    def __init__(self, config, dataset_stats=None, device=None):
        super().__init__()
        self.config = config
        
        # 获取host和port参数
        host = getattr(config, 'host', 'localhost')
        port = getattr(config, 'port', 8000)
        
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        
        # 初始化动作队列
        n_action_steps = getattr(self.config, 'n_action_steps', 25)
        self._action_queue = deque([], maxlen=n_action_steps)

    def reset(self):
        """环境重置时调用"""
        # 清空动作队列
        self._action_queue.clear()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """训练前向传递 - OpenPi未实现，因为我们使用预训练模型"""
        raise NotImplementedError("OpenPi策略仅用于推理，不支持训练")

    
    @torch.inference_mode()
    def select_action(self, batch: Dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps the policy inference in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling the policy when the
        queue is empty.
        """

        

        self.eval()

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            # 从batch中提取图像数据
            # 假设batch包含观察数据
            img = batch.get("observation.images.primary", batch.get("image"))
            wrist_img = batch.get("observation.images.wrist", batch.get("wrist_image"))
            state = batch.get("observation.state", batch.get("state"))
            task_instruction = batch.get("task", "tidy up the items on the table")
                        # 添加调试信息
            print(f"OpenPI输入数据:")
            print(f"  - 图像形状: {img.shape if img is not None else 'None'}")
            print(f"  - 状态形状: {state.shape if state is not None else 'None'}")
            print(f"  - 任务描述: {task_instruction}")
            
            # 检查数据有效性
            if state is not None:
                print(f"  - 状态值范围: {state.min():.4f} ~ {state.max():.4f}")
                print(f"  - 状态是否包含NaN: {torch.isnan(state).any()}")
            
          

            print(f"调试信息:")
            print(f"  - img shape: {img.shape if img is not None else 'None'}")
            print(f"  - wrist_img shape: {wrist_img.shape if wrist_img is not None else 'None'}")
            print(f"  - state shape: {state.shape if state is not None else 'None'}")
            print(f"  - task: {task_instruction}")

            # 转换tensor为numpy数组 (如果需要的话)
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if isinstance(wrist_img, torch.Tensor):
                wrist_img = wrist_img.cpu().numpy()
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()

            # 构建观察字典
            observation = {
                "observation/image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, 224, 224)
                ),
                "observation/wrist_image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, 224, 224)
                ),
                "observation/state": state,
                "prompt": task_instruction,
            }

            # Call the policy server with the current observation.
            # This returns an action chunk of shape (action_horizon, action_dim).
            print('OpenPi starts inference')
            action_chunk = self.client.infer(observation)["actions"]
            
            #print(f"OpenPi返回的action_chunk: {type(action_chunk)}, shape/len: {action_chunk.shape if hasattr(action_chunk, 'shape') else len(action_chunk) if hasattr(action_chunk, '__len__') else 'unknown'}")
            
            # 确保action_chunk是torch tensor
            if not isinstance(action_chunk, torch.Tensor):
                action_chunk = torch.tensor(action_chunk, dtype=torch.float32)
                  # 调用OpenPI服务器

            
            # 检查返回的动作
            print(f"OpenPI返回动作:")
            print(f"  - 动作类型: {type(action_chunk)}")
            print(f"  - 动作形状: {action_chunk.shape if hasattr(action_chunk, 'shape') else len(action_chunk)}")
            print(f"  - 动作值范围: {action_chunk.min():.4f} ~ {action_chunk.max():.4f}")
            print(f"  - 动作是否包含NaN: {torch.isnan(action_chunk).any() if hasattr(action_chunk, 'isnan') else 'Unknown'}")
            
            #print(f"转换后的action_chunk shape: {action_chunk.shape}")
            
            # 如果action_chunk是2D (action_horizon, action_dim)，需要添加batch维度
            if action_chunk.dim() == 2:
                action_chunk = action_chunk.unsqueeze(0)  # (1, action_horizon, action_dim)
            
            #print(f"最终action_chunk shape: {action_chunk.shape}")
            
            # `action_chunk` shape is (batch_size, n_action_steps, action_dim), but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            # `action_chunk` shape is (batch_size, n_action_steps, action_dim), but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.

            n_steps = getattr(self.config, "n_action_steps", action_chunk.shape[1] if action_chunk.dim() >= 2 else None)
            if action_chunk.dim() >= 2 and n_steps is not None and n_steps < action_chunk.shape[1]:
                action_chunk = action_chunk[:, :n_steps, ...]
            self._action_queue.extend(action_chunk.transpose(0, 1))
            
        action = self._action_queue.popleft()
        #print(f"从队列返回的action shape: {action.shape}")
        
        # 确保返回的action是1D tensor
        if action.dim() > 1:
            action = action.squeeze()  # 移除多余的维度
        
        #print(f"最终返回的action shape: {action.shape}")
        return action

    def to(self, device):
        """Override to() to handle device movement"""
        super().to(device)
        self.device = device
        return self