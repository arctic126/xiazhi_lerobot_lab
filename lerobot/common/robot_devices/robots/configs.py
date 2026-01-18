import abc
from dataclasses import dataclass, field
from typing import Sequence

import draccus

from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import (
    DynamixelMotorsBusConfig,
    FeetechMotorsBusConfig,
    MotorsBusConfig,
    PiperMotorsBusConfig,
    PiperEefMotorsBusConfig,
    FlexivMotorsBusConfig,
    Piper2MotorsBusConfig,
    PiperEef1MotorsBusConfig,
    PiperEef2MotorsBusConfig,
    PiperRelMotorsBusConfig,
    JakaMotorsBusConfig
)


@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


# TODO(rcadene, aliberts): remove ManipulatorRobotConfig abstraction
@dataclass
class ManipulatorRobotConfig(RobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle. This makes it
    # possible to squeeze the gripper and have it spring back to an open position on its own. If None, the
    # gripper is not put in torque mode.
    gripper_open_degree: float | None = None

    mock: bool = False

    def __post_init__(self):
        if self.mock:
            for arm in self.leader_arms.values():
                if not arm.mock:
                    arm.mock = True
            for arm in self.follower_arms.values():
                if not arm.mock:
                    arm.mock = True
            for cam in self.cameras.values():
                if not cam.mock:
                    cam.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )


@RobotConfig.register_subclass("aloha")
@dataclass
class AlohaRobotConfig(ManipulatorRobotConfig):
    # Specific to Aloha, LeRobot comes with default calibration files. Assuming the motors have been
    # properly assembled, no manual calibration step is expected. If you need to run manual calibration,
    # simply update this path to ".cache/calibration/aloha"
    calibration_dir: str = ".cache/calibration/aloha_default"

    # /!\ FOR SAFETY, READ THIS /!\
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    # For Aloha, for every goal position request, motor rotations are capped at 5 degrees by default.
    # When you feel more confident with teleoperation or running the policy, you can extend
    # this safety limit and even removing it by setting it to `null`.
    # Also, everything is expected to work safely out-of-the-box, but we highly advise to
    # first try to teleoperate the grippers only (by commenting out the rest of the motors in this yaml),
    # then to gradually add more motors (by uncommenting), until you can teleoperate both arms fully
    max_relative_target: int | None = 5

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                # window_x
                port="/dev/ttyDXL_leader_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm430-w350"],
                    "shoulder": [2, "xm430-w350"],
                    "shoulder_shadow": [3, "xm430-w350"],
                    "elbow": [4, "xm430-w350"],
                    "elbow_shadow": [5, "xm430-w350"],
                    "forearm_roll": [6, "xm430-w350"],
                    "wrist_angle": [7, "xm430-w350"],
                    "wrist_rotate": [8, "xl430-w250"],
                    "gripper": [9, "xc430-w150"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                # window_x
                port="/dev/ttyDXL_leader_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm430-w350"],
                    "shoulder": [2, "xm430-w350"],
                    "shoulder_shadow": [3, "xm430-w350"],
                    "elbow": [4, "xm430-w350"],
                    "elbow_shadow": [5, "xm430-w350"],
                    "forearm_roll": [6, "xm430-w350"],
                    "wrist_angle": [7, "xm430-w350"],
                    "wrist_rotate": [8, "xl430-w250"],
                    "gripper": [9, "xc430-w150"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
        }
    )

    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_high": IntelRealSenseCameraConfig(
                serial_number=128422271347,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_low": IntelRealSenseCameraConfig(
                serial_number=130322270656,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": IntelRealSenseCameraConfig(
                serial_number=218622272670,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_right_wrist": IntelRealSenseCameraConfig(
                serial_number=130322272300,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("koch")
@dataclass
class KochRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@RobotConfig.register_subclass("koch_bimanual")
@dataclass
class KochBimanualRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch_bimanual"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem575E0031751",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem575E0032081",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@RobotConfig.register_subclass("moss")
@dataclass
class MossRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/moss"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("so100")
@dataclass
class So100RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so100"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("stretch")
@dataclass
class StretchRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "navigation": OpenCVCameraConfig(
                camera_index="/dev/hello-nav-head-camera",
                fps=10,
                width=1280,
                height=720,
                rotation=-90,
            ),
            "head": IntelRealSenseCameraConfig(
                name="Intel RealSense D435I",
                fps=30,
                width=640,
                height=480,
                rotation=90,
            ),
            "wrist": IntelRealSenseCameraConfig(
                name="Intel RealSense D405",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Network Configuration
    ip: str = "192.168.0.193"
    port: int = 5555
    video_port: int = 5556

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                camera_index="/dev/video0", fps=30, width=640, height=480, rotation=90
            ),
            "wrist": OpenCVCameraConfig(
                camera_index="/dev/video2", fps=30, width=640, height=480, rotation=180
            ),
        }
    )

    calibration_dir: str = ".cache/calibration/lekiwi"

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0077581",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                    "left_wheel": (7, "sts3215"),
                    "back_wheel": (8, "sts3215"),
                    "right_wheel": (9, "sts3215"),
                },
            ),
        }
    )

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig):
    inference_time: bool
    
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": PiperMotorsBusConfig(
                can_name="can_left",
                motors={
                    # name: (index, model)
                    "joint_1": [1, "agilex_piper"],
                    "joint_2": [2, "agilex_piper"],
                    "joint_3": [3, "agilex_piper"],
                    "joint_4": [4, "agilex_piper"],
                    "joint_5": [5, "agilex_piper"],
                    "joint_6": [6, "agilex_piper"],
                    "gripper": (7, "agilex_piper"),
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                # camera_index=10,
                camera_index=16, 
                # camera_index=22,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                # camera_index=22,
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            # "front": OpenCVCameraConfig(
            #     camera_index=10,
            #    # camera_index=28,
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),
        }
    )


@RobotConfig.register_subclass("piper2")
@dataclass
class Piper2RobotConfig(RobotConfig):
    inference_time: bool
    
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": Piper2MotorsBusConfig(
                can_name="can_right",
                motors={
                    # name: (index, model)
                    "joint_1": [1, "agilex_piper"],
                    "joint_2": [2, "agilex_piper"],
                    "joint_3": [3, "agilex_piper"],
                    "joint_4": [4, "agilex_piper"],
                    "joint_5": [5, "agilex_piper"],
                    "joint_6": [6, "agilex_piper"],
                    "gripper": (7, "agilex_piper"),
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                # camera_index=10,
                camera_index=4,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                # camera_index=22,
                camera_index=16,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )


@RobotConfig.register_subclass("piper_rel")
@dataclass
class PiperRelRobotConfig(RobotConfig):
    inference_time: bool
    
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": PiperRelMotorsBusConfig(
                can_name="can_right",
                motors={
                    # name: (index, model)
                    "joint_1": [1, "agilex_piper"],
                    "joint_2": [2, "agilex_piper"],
                    "joint_3": [3, "agilex_piper"],
                    "joint_4": [4, "agilex_piper"],
                    "joint_5": [5, "agilex_piper"],
                    "joint_6": [6, "agilex_piper"],
                    "gripper": (7, "agilex_piper"),
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                # camera_index=10,
                camera_index=10,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                # camera_index=22,
                camera_index=22,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )



@RobotConfig.register_subclass("piper_eef")
@dataclass
class PiperEefRobotConfig(RobotConfig):
    inference_time: bool
    
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": PiperEefMotorsBusConfig(
                can_name="can_right",
                motors={
                    # name: (index, model)
                    "x": [1, "agilex_piper"],
                    "y": [2, "agilex_piper"],
                    "z": [3, "agilex_piper"],
                    "rx": [4, "agilex_piper"],
                    "ry": [5, "agilex_piper"],
                    "rz": [6, "agilex_piper"],
                    "gripper": (7, "agilex_piper"),
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                camera_index=10,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                camera_index=22,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

@RobotConfig.register_subclass("piper_eef1")
@dataclass
class PiperEef1RobotConfig(RobotConfig):
    inference_time: bool
    
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": PiperEef1MotorsBusConfig(
                can_name="can_left",
                motors={
                    # name: (index, model)
                    "x": [1, "agilex_piper"],
                    "y": [2, "agilex_piper"],
                    "z": [3, "agilex_piper"],
                    "roll": [4, "agilex_piper"],
                    "yaw": [5, "agilex_piper"],
                    "pitch": [6, "agilex_piper"],
                    "gripper": (7, "agilex_piper"),
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                camera_index=4,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                camera_index=16,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

@RobotConfig.register_subclass("piper_eef2")
@dataclass
class PiperEef2RobotConfig(RobotConfig):
    inference_time: bool
    
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": PiperEef2MotorsBusConfig(
                can_name="can_right",
                motors={
                    # name: (index, model)
                    "x": [1, "agilex_piper"],
                    "y": [2, "agilex_piper"],
                    "z": [3, "agilex_piper"],
                    "roll": [4, "agilex_piper"],
                    "yaw": [5, "agilex_piper"],
                    "pitch": [6, "agilex_piper"],
                    "gripper": (7, "agilex_piper"),
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                camera_index=10,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                camera_index=22,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )


@RobotConfig.register_subclass("flexiv")
@dataclass
class FlexivRobotConfig(RobotConfig):
    """Flexiv机器人配置类"""
    
    # 是否启用推理时间
    inference_time: bool
    
    # 机械臂配置
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FlexivMotorsBusConfig(
                serial_port="/dev/tty.usbmodem585A0076891",
                gripper_name="gripper",
                motors={
                    # name: (index, model)
                    "x": [1, "flexiv_rizon"],
                    "y": [2, "flexiv_rizon"],
                    "z": [3, "flexiv_rizon"],
                    "rx": [4, "flexiv_rizon"],
                    "ry": [5, "flexiv_rizon"],
                    "rz": [6, "flexiv_rizon"],
                    "gripper": [7, "flexiv_gripper"],
                },
            ),
        }
    )
    
    # 相机配置
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                camera_index=12,
                fps=50,
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                camera_index=6,
                fps=50,
                width=640,
                height=480,
            ),
        }
    )
    
    # 是否使用模拟模式
    mock: bool = False


@RobotConfig.register_subclass("jaka")
@dataclass
class JakaRobotConfig(RobotConfig):
    """JAKA机器人配置类 - 末端位姿控制模式"""
    
    # 推理时间标志（必须为True）
    inference_time: bool = True
    
    # 机械臂配置（末端位姿控制）
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": JakaMotorsBusConfig(
                robot_ip="192.168.31.112",  # JAKA机器人IP地址，请根据实际修改
                end_effector_dof={
                    # name: (index, model)
                    # 末端自由度：[x, y, z, yaw, pitch, roll, gripper]
                    "x": [0, "jaka_s5"],
                    "y": [1, "jaka_s5"],
                    "z": [2, "jaka_s5"],
                    "yaw": [3, "jaka_s5"],
                    "pitch": [4, "jaka_s5"],
                    "roll": [5, "jaka_s5"],
                    "gripper": [6, "jaka_s5"],  # JAKA无夹爪，值固定为0
                },
            ),
        }
    )
    
    # 相机配置
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                camera_index=1,  # 相机编号，请根据实际修改
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
    
    # 力传感器配置
    force_sensor: dict = field(
        default_factory=lambda: {
            "enabled": True,  # 是否启用力传感器
            "ip_addr": "192.168.0.108",  # 宇立力传感器IP地址
            "port": 4008  # 宇立力传感器端口
        }
    )
    
    # 模拟模式（推理时建议关闭）
    mock: bool = False


@RobotConfig.register_subclass("jaka_vla")
@dataclass
class JakaRobotVLAConfig(RobotConfig):
    """JAKA机器人配置类 - VLA版本（使用简单加法计算绝对位置）"""
    
    # 推理时间标志（必须为True）
    inference_time: bool = True
    
    # 机械臂配置（末端位姿控制）
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": JakaMotorsBusConfig(
                robot_ip="192.168.2.64",  # JAKA机器人IP地址，请根据实际修改
                end_effector_dof={
                    # name: (index, model)
                    # 末端自由度：[x, y, z, rx, ry, rz, gripper]
                    "x": [0, "jaka_s5"],
                    "y": [1, "jaka_s5"],
                    "z": [2, "jaka_s5"],
                    "rx": [3, "jaka_s5"],
                    "ry": [4, "jaka_s5"],
                    "rz": [5, "jaka_s5"],
                    "gripper": [6, "jaka_s5"],  # JAKA无夹爪，值固定为0
                },
            ),
        }
    )
    
    # 相机配置
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                camera_index=1,  # 相机编号，请根据实际修改
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
    
    # 力传感器配置（VLA需要20帧历史）
    force_sensor: dict = field(
        default_factory=lambda: {
            "enabled": True,  # 是否启用力传感器
            "ip_addr": "192.168.0.108",  # 宇立力传感器IP地址
            "port": 4008  # 宇立力传感器端口
        }
    )
    
    # 模拟模式（推理时建议关闭）
    mock: bool = False
