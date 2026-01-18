"""
使用OpenVLA策略控制Piper机器人，每次重置后等待用户输入任务。

示例用法：
```bash
python lerobot/scripts/run_piper_with_openvla.py \
    --model_name=openvla/openvla-7b \
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
from lerobot.common.policies.openvla.modeling_openvla import OpenVLAPolicy
from lerobot.common.policies.openvla.configuration_openvla import OpenVLAConfig

@dataclass
class PiperOpenVLAConfig:
    """Piper OpenVLA运行配置"""
    robot_type: str = "piper_eef"
    model_name: str = "openvla/openvla-7b"
    device: str = "cuda"
    use_amp: bool = False
    fps: int = 5
    display_cameras: bool = True
    play_sounds: bool = False
    
@safe_disconnect
def run_piper_with_openvla(
    robot: Robot,
    policy,
    device,
    use_amp=False,
    fps=5,
    display_cameras=True,
    play_sounds=False,
):
    """
    运行Piper机器人，使用OpenVLA策略进行控制，每次重置后等待用户输入任务
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
            
            # 更新策略中的任务指令
            policy.config.instruction = task_description
            
            # 执行任务
            run_task(
                robot=robot,
                policy=policy,
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

def run_task(
    robot,
    policy,
    device,
    events,
    use_amp=False,
    fps=5,
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
        
        # 预测动作
        with torch.inference_mode():
            # 可选的自动混合精度
            if device.type == "cuda" and use_amp:
                with torch.autocast(device_type="cuda"):
                    action = policy.select_action(observation)
            else:
                action = policy.select_action(observation)
        
        # 发送动作给机器人
        robot.send_action(action)

        
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
    parser = argparse.ArgumentParser(description="使用OpenVLA策略控制Piper机器人")
    parser.add_argument("--model_name", type=str, default="openvla/openvla-7b", help="OpenVLA模型名称")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("--use_amp", action="store_true", help="是否使用混合精度")
    parser.add_argument("--fps", type=int, default=5, help="控制频率")
    parser.add_argument("--display_cameras", action="store_true", default=False, help="是否显示相机图像")
    
    # 解析参数
    args = parser.parse_args()
    
    # 初始化日志
    init_logging()
    
    # 构建机器人配置 - 提供inference_time=True参数
    from lerobot.common.robot_devices.robots.configs import PiperEefRobotConfig
    robot_config = PiperEefRobotConfig(inference_time=True)
    
    # 创建机器人
    robot = make_robot_from_config(robot_config)
    
    # 获取设备
    device = get_safe_torch_device(args.device)
    
    # 创建OpenVLA配置
    openvla_config = OpenVLAConfig(model_name=args.model_name)
    
    # 创建OpenVLA策略，传入设备参数
    policy = OpenVLAPolicy(config=openvla_config, device=device)
    
    # 运行机器人
    run_piper_with_openvla(
        robot=robot,
        policy=policy,
        device=device,
        use_amp=args.use_amp,
        fps=args.fps,
        display_cameras=args.display_cameras,
    )

if __name__ == "__main__":
    main() 