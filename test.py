from typing import (
    Optional,
)
import time
from piper_sdk import *

# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_left",False)
    piper.ConnectPort()
    # piper.CrashProtectionConfig(1,1,1,1,1,1)
    piper.CrashProtectionConfig(0,0,0,0,0,0)
    while True:
        piper.ArmParamEnquiryAndConfig(0x02, 0x00, 0x00, 0x00, 0x03)

        print(piper.GetCrashProtectionLevelFeedback())
        time.sleep(0.01)