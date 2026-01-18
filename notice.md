When you use master-slave teleop, comment out the belows:
    `lerobot/common/robot_devices/motors/piper.py` line 130-132

When you inference with Pi0, modify line 392 in `lerobot/common/policies/pi0/modeling_pi0.py`

If you use relative control, modify line ~256 in `lerobot/common/robot_devices/control_utils.py`

currently finished flexiv control adaptation. TODO: delta ee pose data collection for flexiv.