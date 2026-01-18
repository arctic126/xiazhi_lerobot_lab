from typing import Protocol

from lerobot.common.robot_devices.motors.configs import (
    DynamixelMotorsBusConfig,
    FeetechMotorsBusConfig,
    PiperMotorsBusConfig,
    Piper2MotorsBusConfig,
    PiperRelMotorsBusConfig,
    MotorsBusConfig
)


class MotorsBus(Protocol):
    def motor_names(self): ...
    def set_calibration(self): ...
    def apply_calibration(self): ...
    def revert_calibration(self): ...
    def read(self): ...
    def write(self): ...


def make_motors_buses_from_configs(motors_bus_configs: dict[str, MotorsBusConfig]) -> list[MotorsBus]:
    motors_buses = {}

    for key, cfg in motors_bus_configs.items():
        if cfg.type == "dynamixel":
            from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

            motors_buses[key] = DynamixelMotorsBus(cfg)

        elif cfg.type == "feetech":
            from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

            motors_buses[key] = FeetechMotorsBus(cfg)

        elif cfg.type == "piper":
            from lerobot.common.robot_devices.motors.piper import PiperMotorsBus

            motors_buses[key] = PiperMotorsBus(cfg)

        elif cfg.type == "piper2":
            from lerobot.common.robot_devices.motors.piper2 import Piper2MotorsBus

            motors_buses[key] = Piper2MotorsBus(cfg)

        elif cfg.type == "piper_eef":
            from lerobot.common.robot_devices.motors.piper_eef import PiperEefMotorsBus

            motors_buses[key] = PiperEefMotorsBus(cfg)

        elif cfg.type == "piper_eef1":
            from lerobot.common.robot_devices.motors.piper_eef1 import PiperEef1MotorsBus

            motors_buses[key] = PiperEef1MotorsBus(cfg)

        elif cfg.type == "piper_eef2":
            from lerobot.common.robot_devices.motors.piper_eef2 import PiperEef2MotorsBus

            motors_buses[key] = PiperEef2MotorsBus(cfg)

        elif cfg.type == "piper_rel":
            from lerobot.common.robot_devices.motors.piper_rel import PiperRelMotorsBus

            motors_buses[key] = PiperRelMotorsBus(cfg)


        elif cfg.type == "flexiv":
            from lerobot.common.robot_devices.motors.flexiv import FlexivMotorsBus

            motors_buses[key] = FlexivMotorsBus(cfg)

        elif cfg.type == "jaka":
            from lerobot.common.robot_devices.motors.jaka import JakaMotorsBus

            motors_buses[key] = JakaMotorsBus(cfg)

        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return motors_buses


def make_motors_bus(motor_type: str, **kwargs) -> MotorsBus:
    if motor_type == "dynamixel":
        from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

        config = DynamixelMotorsBusConfig(**kwargs)
        return DynamixelMotorsBus(config)

    elif motor_type == "feetech":
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

        config = FeetechMotorsBusConfig(**kwargs)
        return FeetechMotorsBus(config)
    
    elif motor_type == "piper":
        from lerobot.common.robot_devices.motors.piper import PiperMotorsBus

        config = PiperMotorsBusConfig(**kwargs)
        return PiperMotorsBus(config)
    
    elif motor_type == "piper2":
        from lerobot.common.robot_devices.motors.piper2 import Piper2MotorsBus

        config = Piper2MotorsBusConfig(**kwargs)
        return Piper2MotorsBus(config)
    
    elif motor_type == "piper_rel":
        from lerobot.common.robot_devices.motors.piper_rel import PiperRelMotorsBus

        config = PiperRelMotorsBusConfig(**kwargs)
        return PiperRelMotorsBus(config)

    else:
        raise ValueError(f"The motor type '{motor_type}' is not valid.")

def get_motor_names(arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]
