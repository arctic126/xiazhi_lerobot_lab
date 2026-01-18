"""
使用pi0策略控制Piper机器人，每次重置后等待用户输入任务。

示例用法：
```bash
python lerobot/scripts/run_piper_with_pi0.py \
    --policy_path=outputs/train/pi0_61/checkpoints/060000/pretrained_model \
    --device=cuda
```
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Optional

import torch
import cv2

from lerobot.common.robot_devices.control_configs import ControlPipelineConfig
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    reset_environment,
    busy_wait,
    is_headless,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.utils.utils import init_logging, get_safe_torch_device, log_say
from lerobot.configs import parser
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

@dataclass
class PiperPI0Config:
    """Piper PI0运行配置"""
    robot_type: str = "piper"
    policy_path: Optional[str] = None
    device: str = "cuda"
    use_amp: bool = False
    fps: int = 30
    display_cameras: bool = True
    play_sounds: bool = False
    
@safe_disconnect
def run_piper_with_pi0(
    robot: Robot,
    policy,
    device,
    use_amp=False,
    fps=30,
    display_cameras=True,
    play_sounds=False,
):
    """
    运行Piper机器人，使用pi0策略进行控制，每次重置后等待用户输入任务
    """
    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()
    events["stop_running"] = False

    try:
        while not events["stop_running"]:
            # 重置环境
            log_say("重置环境", play_sounds)
            reset_environment(robot, events, reset_time_s=5, fps=fps)
            
            # 等待用户输入任务
            print("\n请输入任务描述（按Enter确认）：")
            task_description = input("> ")
            if not task_description:
                task_description = "place the screwdriver into the box."
                print(f"使用默认任务: {task_description}")
            
            log_say(f"执行任务: {task_description}", play_sounds)
            
            # 执行任务
            run_task(
                robot=robot,
                policy=policy,
                task_description=task_description,
                device=device,
                events=events,
                use_amp=use_amp,
                fps=fps,
                display_cameras=display_cameras,
            )
            
            if events["stop_running"]:
                break
                
            print("\n任务完成！按ESC键退出，或输入新任务继续...")
            
    finally:
        # 清理资源
        if listener is not None:
            listener.stop()
            
        if display_cameras and not is_headless():
            cv2.destroyAllWindows()
            
        robot.disconnect()
        log_say("程序结束", play_sounds)

def predict_action_with_task(observation, task_description, policy, device, use_amp):
    """自定义预测函数，正确处理task字符串"""
    observation_copy = {}
    
    # 拷贝观察数据并准备任务描述
    for name, value in observation.items():
        if "image" in name:
            # 转换图像为浮点数并调整通道顺序
            tensor_value = value.type(torch.float32) / 255
            tensor_value = tensor_value.permute(2, 0, 1).contiguous()
            observation_copy[name] = tensor_value.unsqueeze(0).to(device)
        else:
            # 处理其他张量数据
            observation_copy[name] = value.unsqueeze(0).to(device)
    
    # 单独处理任务字符串，不做张量转换
    observation_copy["task"] = task_description
    
    # 使用torch的推理模式
    with torch.inference_mode():
        # 可选的自动混合精度
        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda"):
                action = policy.select_action(observation_copy)
        else:
            action = policy.select_action(observation_copy)
        
        # 移除批量维度并移至CPU
        action = action.squeeze(0).to("cpu")
    
    return action

def run_task(
    robot,
    policy,
    task_description,
    device,
    events,
    use_amp=False,
    fps=30,
    display_cameras=True,
    episode_time_s=30,
):
    """执行单个任务"""
    # 重置策略
    policy.reset()
    
    start_episode_t = time.perf_counter()
    elapsed_time = 0
    
    while elapsed_time < episode_time_s:
        start_loop_t = time.perf_counter()
        
        # 获取观察
        observation = robot.capture_observation()
        
        # 预测动作（使用自定义的预测函数，正确处理任务描述）
        pred_action = predict_action_with_task(observation, task_description, policy, device, use_amp)
        
        # 发送动作给机器人
        robot.send_action(pred_action)
        
        # 显示相机图像
        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        
        # 维持帧率
        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)
        
        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)
        
        elapsed_time = time.perf_counter() - start_episode_t
        
        # 处理提前退出
        if events.get("exit_early", False):
            events["exit_early"] = False
            break
            
        # 处理停止运行
        if events.get("stop_recording", False):
            events["stop_running"] = True
            break
    
    return

def main():
    """主入口函数，处理命令行参数并启动程序"""
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="使用pi0策略控制Piper机器人")
    parser.add_argument("--policy_path", type=str, required=True, help="pi0策略路径")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("--use_amp", action="store_true", help="是否使用混合精度")
    parser.add_argument("--fps", type=int, default=30, help="控制频率")
    parser.add_argument("--display_cameras", action="store_true", default=True, help="是否显示相机图像")
    
    # 解析参数
    args = parser.parse_args()
    
    # 初始化日志
    init_logging()
    
    # 构建机器人配置 - 提供inference_time=True参数
    from lerobot.common.robot_devices.robots.configs import PiperRobotConfig
    robot_config = PiperRobotConfig(inference_time=True)
    
    # 创建机器人
    robot = make_robot_from_config(robot_config)
    
    # 直接使用from_pretrained加载预训练模型
    policy = PI0Policy.from_pretrained(args.policy_path, map_location=args.device)
    
    # 获取设备
    device = get_safe_torch_device(args.device)
    
    # 运行机器人
    run_piper_with_pi0(
        robot=robot,
        policy=policy,
        device=device,
        use_amp=args.use_amp,
        fps=args.fps,
        display_cameras=args.display_cameras,
    )

if __name__ == "__main__":
    main() 