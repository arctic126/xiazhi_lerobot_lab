import abc
from dataclasses import dataclass

import draccus


@dataclass
class MotorsBusConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@MotorsBusConfig.register_subclass("dynamixel")
@dataclass
class DynamixelMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False


@MotorsBusConfig.register_subclass("feetech")
@dataclass
class FeetechMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False


@MotorsBusConfig.register_subclass("piper")
@dataclass
class PiperMotorsBusConfig(MotorsBusConfig):
    can_name: str
    motors: dict[str, tuple[int, str]]


@MotorsBusConfig.register_subclass("piper2")
@dataclass
class Piper2MotorsBusConfig(MotorsBusConfig):
    can_name: str
    motors: dict[str, tuple[int, str]]

@MotorsBusConfig.register_subclass("piper_eef")
@dataclass
class PiperEefMotorsBusConfig(MotorsBusConfig):
    can_name: str
    motors: dict[str, tuple[int, str]]

@MotorsBusConfig.register_subclass("piper_eef1")
@dataclass
class PiperEef1MotorsBusConfig(MotorsBusConfig):
    can_name: str
    motors: dict[str, tuple[int, str]]


@MotorsBusConfig.register_subclass("piper_eef2")
@dataclass
class PiperEef2MotorsBusConfig(MotorsBusConfig):
    can_name: str
    motors: dict[str, tuple[int, str]]

@MotorsBusConfig.register_subclass("piper_rel")
@dataclass
class PiperRelMotorsBusConfig(MotorsBusConfig):
    can_name: str
    motors: dict[str, tuple[int, str]]


@MotorsBusConfig.register_subclass("flexiv")
@dataclass
class FlexivMotorsBusConfig(MotorsBusConfig):
    serial_port: str
    gripper_name: str
    motors: dict[str, tuple[int, str]]


@MotorsBusConfig.register_subclass("jaka")
@dataclass
class JakaMotorsBusConfig(MotorsBusConfig):
    robot_ip: str
    # 末端执行器自由度配置 [x, y, z, rx, ry, rz, gripper]
    end_effector_dof: dict[str, tuple[int, str]]
