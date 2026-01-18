from kinematics.piper_fk import C_PiperForwardKinematics as piper_fk
from piper_sdk import *
import time

piper = C_PiperInterface_V2()
piper.ConnectPort()
piper_fk = piper_fk()

while True:
    eef1 = piper.GetArmEndPoseMsgs()
    print(eef1)
    current_joints = piper.GetArmJointMsgs()
    joint_angles = [
        current_joints.joint_state.joint_1 / 1000,
        current_joints.joint_state.joint_2 / 1000,
        current_joints.joint_state.joint_3 / 1000,
        current_joints.joint_state.joint_4 / 1000,
        current_joints.joint_state.joint_5 / 1000,
        current_joints.joint_state.joint_6 / 1000
    ]
    # print(f"joint_angles:{joint_angles}")
    # current_eef_pose = piper_fk.CalFK(joint_angles)
    # print(f"current_eef_pose:{current_eef_pose}")
    current_eef_pose_feedback = piper.GetFK('feedback')
    current_eef_pose_ctrl = piper.GetFK('control')
    print(f"current_eef_pose_feedback:{current_eef_pose_feedback}")
    print('\n')
    print(f"current_eef_pose_ctrl:{current_eef_pose_ctrl}")
    time.sleep(0.1)




# current_eef_pose = piper_fk.CalFK(current_joints)[5]
# print(current_eef_pose)