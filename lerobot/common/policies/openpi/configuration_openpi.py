from dataclasses import dataclass

@dataclass 
class OpenPiConfig:
    """Configuration for OpenVLA policy"""
    n_action_steps: int = 25
    name: str = "openpi"