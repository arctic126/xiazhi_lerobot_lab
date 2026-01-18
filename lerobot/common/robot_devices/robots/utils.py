from typing import Protocol

from scipy.spatial.transform import Rotation as R

from lerobot.common.robot_devices.robots.configs import (
    AlohaRobotConfig,
    KochBimanualRobotConfig,
    KochRobotConfig,
    LeKiwiRobotConfig,
    ManipulatorRobotConfig,
    MossRobotConfig,
    RobotConfig,
    So100RobotConfig,
    StretchRobotConfig,
    PiperRobotConfig,
    Piper2RobotConfig,
    FlexivRobotConfig,
    PiperEefRobotConfig,
    PiperEef1RobotConfig,
    PiperEef2RobotConfig,
    PiperRelRobotConfig,
    JakaRobotConfig,
    JakaRobotVLAConfig,
)


def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"


class Robot(Protocol):
    # TODO(rcadene, aliberts): Add unit test checking the protocol is implemented in the corresponding classes
    robot_type: str
    features: dict

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    if robot_type == "aloha":
        return AlohaRobotConfig(**kwargs)
    elif robot_type == "koch":
        return KochRobotConfig(**kwargs)
    elif robot_type == "koch_bimanual":
        return KochBimanualRobotConfig(**kwargs)
    elif robot_type == "moss":
        return MossRobotConfig(**kwargs)
    elif robot_type == "so100":
        return So100RobotConfig(**kwargs)
    elif robot_type == "stretch":
        return StretchRobotConfig(**kwargs)
    elif robot_type == "lekiwi":
        return LeKiwiRobotConfig(**kwargs)
    elif robot_type == 'piper':
        return PiperRobotConfig(**kwargs)
    elif robot_type == 'piper2':
        return Piper2RobotConfig(**kwargs)
    elif robot_type == 'piper_eef':
        return PiperEefRobotConfig(**kwargs)
    elif robot_type == 'piper_eef1':
        return PiperEef1RobotConfig(**kwargs)
    elif robot_type == 'piper_eef2':
        return PiperEef2RobotConfig(**kwargs)
    elif robot_type == 'flexiv':
        return FlexivRobotConfig(**kwargs)
    elif robot_type == 'piper_rel':
        return PiperRelRobotConfig(**kwargs)
    elif robot_type == 'jaka':
        return JakaRobotConfig(**kwargs)
    else:
        raise ValueError(f"Robot type '{robot_type}' is not available.")


def make_robot_from_config(config: RobotConfig):
    if isinstance(config, ManipulatorRobotConfig):
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

        return ManipulatorRobot(config)
    elif isinstance(config, LeKiwiRobotConfig):
        from lerobot.common.robot_devices.robots.mobile_manipulator import MobileManipulator

        return MobileManipulator(config)
    elif isinstance(config, PiperRobotConfig):
        from lerobot.common.robot_devices.robots.piper import PiperRobot

        return PiperRobot(config)
    elif isinstance(config, Piper2RobotConfig):
        from lerobot.common.robot_devices.robots.piper2 import Piper2Robot

        return Piper2Robot(config)
    elif isinstance(config, PiperEefRobotConfig):
        from lerobot.common.robot_devices.robots.piper_eef import PiperEefRobot

        return PiperEefRobot(config)
    elif isinstance(config, PiperEef1RobotConfig):
        from lerobot.common.robot_devices.robots.piper_eef1 import PiperEef1Robot

        return PiperEef1Robot(config)
    elif isinstance(config, PiperEef2RobotConfig):
        from lerobot.common.robot_devices.robots.piper_eef2 import PiperEef2Robot

        return PiperEef2Robot(config)
    elif isinstance(config, PiperRelRobotConfig):
        from lerobot.common.robot_devices.robots.piper_rel import PiperRelRobot

        return PiperRelRobot(config)
    elif isinstance(config, FlexivRobotConfig):
        from lerobot.common.robot_devices.robots.flexiv import FlexivRobot

        return FlexivRobot(config)
    elif isinstance(config, JakaRobotVLAConfig):
        from lerobot.common.robot_devices.robots.jaka_vla import JakaRobotVLA

        return JakaRobotVLA(config)
    elif isinstance(config, JakaRobotConfig):
        from lerobot.common.robot_devices.robots.jaka import JakaRobot

        return JakaRobot(config)
    else:
        from lerobot.common.robot_devices.robots.stretch import StretchRobot

        return StretchRobot(config)


def make_robot(robot_type: str, **kwargs) -> Robot:
    config = make_robot_config(robot_type, **kwargs)
    return make_robot_from_config(config)

def quat2eulerZYX(quat, degree=False):
    """
    Convert quaternion to Euler angles with ZYX axis rotations.

    Parameters
    ----------
    quat : float list
        Quaternion input in [w,x,y,z] order.
    degree : bool
        Return values in degrees, otherwise in radians.

    Returns
    ----------
    float list
        Euler angles in [x,y,z] order, radian by default unless specified otherwise.
    """

    # Convert target quaternion to Euler ZYX using scipy package's 'xyz' extrinsic rotation
    # NOTE: scipy uses [x,y,z,w] order to represent quaternion
    eulerZYX = (
        R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        .as_euler("xyz", degrees=degree)
        .tolist()
    )

    return eulerZYX

def eulerZYX2quat(eulerZYX, degree=False):
    """
    Convert Euler angles to quaternion with ZYX axis rotations.

    Parameters
    ----------
    eulerZYX : float list
        Euler angles in [x,y,z] order.
    degree : bool
        Return values in degrees, otherwise in radians.

    Returns
    ----------
    float list
        Quaternion in [w,x,y,z] order.
    """
    quat = R.from_euler("xyz", eulerZYX, degrees=degree).as_quat().tolist()

    return [quat[3], quat[0], quat[1], quat[2]]
