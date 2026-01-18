from dataclasses import dataclass

@dataclass 
class OpenVLAConfig:
    """Configuration for OpenVLA policy"""
    model_name: str = "openvla/openvla-7b"
    instruction: str = "place the tape into the box" # Task instruction for prompting