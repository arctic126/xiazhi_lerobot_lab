# 霞智机器人数据采集

---

## 安装

### 虚拟环境

```bash
conda create -n jakalerobot python=3.10
conda activate jakalerobot
```

### Lerobot安装

```bash
git clone https://github.com/arctic126/xiazhi_lerobot_lab.git
cd xiazhi_lerobot_lab
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

pip uninstall numpy
pip install numpy==1.26.0
pip install pynput
```

对于Linux系统，额外运行：

```
conda install -c conda-forge ffmpeg
pip uninstall opencv-python
conda install "opencv>=4.10.0"
```

### Jaka安装

下载路径https://www.jaka.com/prod-api/common/download/resource?resource=%2Fprofile%2Fupload%2F2024%2F11%2F21%2F20241121111728A003.tar

Linux需要将libjakaAPI.so和jkrc.so 放在同一个文件夹下，并添加当前文件夹路径到环境变量，

```bash
export LD_LIBRARY_PATH=/xx/xx/
```

同时将so文件放到conda对应env路径下。

### ForceUMI安装

```bash
git clone https://github.com/arctic126/ForceUMI.git
cd ForceUMI
pip install -e .
```

---

### PyTracker安装

```bash
git clone https://github.com/arctic126/PyTracker.git
cd PyTracker
pip install -e .
cd ..
```

**说明：**

需要安装 **SteamVR**

后修改配置文件

```
gedit ~/.steam/steam/steamapps/common/SteamVR/resources/settings/default.vrsettings
```

将第三行的 `"requireHmd" : true,` 改为 `"requireHmd" : false,` 保存并退出设置文件。

---

### PyForce安装

```bash
git clone https://github.com/arctic126/PyForce.git
cd PyForce
pip install -e .
cd ..
```

**说明：**

* 需要 Sunrise（宇立）六轴力/力矩传感器
* 通过 TCP/IP 与主机通信

---

### TAVLA安装

1. 克隆仓库（包含子模块）

```
git clone --recurse-submodules https://github.com/arctic126/TA-VLA.git
cd TA-VLA
```

2. 安装依赖

该项目使用 `uv` 进行环境与依赖管理，并默认跳过 Git LFS 自动拉取大文件：

```
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

需要注意，该虚拟环境中同时需要再次Lerobot安装

```bash
cd TA-VLA
source .venv/bin/activate
```

进入虚拟环境后再次执行Lerobot的安装

## 快速开始

### 1. 启动 GUI 数据采集程序

```bash
python -m forceumi.gui.cv_main_window
```

或使用示例启动脚本：

```bash
python examples/launch_gui.py
```

#### 键盘快捷键说明

* `C`：连接设备
* `D`：断开设备
* `S`：开始采集
* `E`：停止并保存当前 episode
* `Q`：退出程序

---

### 2.模型运行方式

先在Lerobot的config中调整为力传感器、机器人IP以及摄像头编号。

模型运行方式：

开启模型：

```bash
cd TA-VLA
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi0_lora_favla \
    --policy.dir /path/to/ckpt
```

另外打开终端

```
cd xiazhi_lerobot_lab
python eval_jaka_tavla.py \
    --server_host localhost \
    --server_port 8000 \
    --robot_ip <robotip> \
    --force_ip 192.168.0.108 \
    --camera_index <cameraindex> \
    --task_prompt "clean the basin" \
    --max_steps 500 \
    --frequency 10
```

