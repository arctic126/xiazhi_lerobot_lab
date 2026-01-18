# JAKA + FADP 推理使用指南

使用训练好的Force-Aware Diffusion Policy (FADP)模型控制JAKA机器人

## 📋 前置条件

### 1. 硬件设备
- ✅ JAKA S5机器人（已上电并连接到网络）
- ✅ 宇立(Sunrise)力传感器（已连接到网络）
- ✅ USB相机（已连接到控制电脑）
- ✅ 控制电脑（安装有CUDA的GPU，推荐）

### 2. 软件环境
```bash
# 确保已安装以下Python包
pip install torch torchvision  # PyTorch
pip install opencv-python      # 图像处理
pip install hydra-core         # FADP配置管理
pip install diffusers          # Diffusion模型
pip install git+https://github.com/Elycyx/PyForce.git  # 力传感器
```

### 3. 模型文件
- 训练好的FADP checkpoint文件（`.ckpt`格式）
- 位置：`/home/hyx/xiazhi/model/fadp_ckpt/2025.10.28/16.14.32_train_force_aware_diffusion_policy_real_forceumi1/checkpoints/latest.ckpt`

## 🚀 快速开始

### 基本使用

```bash
cd /home/hyx/xiazhi/jaka-Lerobot/lerobot_lab

python eval_jaka_fadp.py \
    --checkpoint /home/hyx/xiazhi/model/fadp_ckpt/2025.10.28/16.14.32_train_force_aware_diffusion_policy_real_forceumi1/checkpoints/latest.ckpt \
    --robot_ip 192.168.2.64 \
    --force_ip 192.168.0.108 \
    --camera_index 0 \
    --max_steps 500 \
    --frequency 10
```

### 参数说明

| 参数 | 短参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `--checkpoint` | `-c` | FADP模型checkpoint路径 | **必需** |
| `--robot_ip` | `-r` | JAKA机器人IP地址 | 192.168.2.64 |
| `--force_ip` | `-f` | 宇立力传感器IP地址 | 192.168.0.108 |
| `--camera_index` | `-cam` | 相机索引（通常是0） | 0 |
| `--max_steps` | `-m` | 每个episode最大步数 | 500 |
| `--frequency` | `-hz` | 控制频率（Hz） | 10.0 |
| `--device` | `-d` | 计算设备（cuda/cpu） | cuda |

## 🎮 运行时控制

### 键盘控制
- **'q'键**: 退出程序
- **'s'键**: 停止当前episode

### OpenCV窗口显示
推理时会实时显示：
- 📷 相机画面
- 📍 当前末端位置 (x, y, z)
- 💪 当前力大小 (总力)
- 🎯 预测的action步数

## 📊 数据流说明

### Observation（输入）
```
{
    'camera_0': (8, 3, 240, 320)  # 8帧RGB图像历史
    'force': (8, 6)                # 8帧力/力矩历史 [fx,fy,fz,mx,my,mz]
}
```

### Action（输出）
```
(16, 7)  # 16步预测，每步7维
# [Δx, Δy, Δz, Δyaw, Δpitch, Δroll, Δgripper]
# 注意：这是相对于episode第一帧的增量
```

### Action执行
脚本目前只执行每次预测的第一步action。如果需要执行多步，可以修改代码中的：
```python
action_to_execute = action[0]  # 只执行第一步
```

## ⚙️ 工作流程

1. **加载模型**
   - 加载FADP checkpoint
   - 设置为评估模式
   - 配置DDIM推理步数（16步）

2. **连接设备**
   - 连接JAKA机器人
   - 连接力传感器并校零
   - 连接相机

3. **预热**
   - 采集一帧dummy observation
   - 运行一次推理预热GPU

4. **推理循环**
   ```
   while not done:
       1. capture_observation()    # 采集obs（含force历史）
       2. policy.predict_action()  # FADP推理
       3. robot.send_action()      # 发送action到机器人
       4. visualize()              # 可视化
       5. check_keyboard()         # 检查用户输入
   ```

5. **清理**
   - 断开所有设备
   - 关闭显示窗口

## 🔧 故障排查

### 问题1: 模型加载失败
```
FileNotFoundError: checkpoint not found
```
**解决**：
- 检查checkpoint路径是否正确
- 确保`.ckpt`文件存在
- 确保有读取权限

### 问题2: 机器人连接失败
```
ConnectionError: JAKA机器人连接失败
```
**解决**：
- 检查机器人IP地址
- 确保机器人已上电
- 检查网络连接
- 尝试ping机器人IP

### 问题3: 力传感器连接失败
```
警告: 力传感器连接失败
```
**解决**：
- 检查力传感器IP和端口
- 确保PyForce已正确安装
- 检查传感器电源和网络

### 问题4: 相机打不开
```
警告: 相机 primary 连接失败
```
**解决**：
- 检查camera_index（尝试0, 2, 4等）
- 使用`v4l2-ctl --list-devices`查看可用相机
- 确保相机未被其他程序占用

### 问题5: CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决**：
- 使用CPU模式：`--device cpu`
- 关闭其他占用GPU的程序
- 减小batch size（修改代码）

### 问题6: 机器人动作异常
**症状**：机器人运动不平滑或抖动

**可能原因**：
1. **频率设置过高**：降低`--frequency`（如从10降到5）
2. **Force数据噪声**：检查力传感器校零
3. **Reference state问题**：重启程序以重置reference

## 📈 性能优化

### 提高推理速度
1. **使用GPU**：`--device cuda`
2. **降低DDIM步数**：修改代码中的`policy.num_inference_steps`
3. **减少observation历史**：修改`n_obs_steps`（需重新训练）

### 提高控制精度
1. **提高控制频率**：增加`--frequency`（推荐10-30Hz）
2. **执行多步action**：修改代码执行action[0:N]而不是只执行action[0]
3. **优化force校零**：在稳定状态下重新校零

## 🔍 调试技巧

### 打印详细信息
脚本已包含详细的打印输出：
- ✅ 推理时间
- ✅ 预测的action步数
- ✅ 实际控制频率
- ✅ 当前位置和力

### 保存运行日志
```bash
python eval_jaka_fadp.py [参数] 2>&1 | tee inference.log
```

### 检查数据维度
在脚本中添加打印：
```python
print(f"Policy obs keys: {policy_obs.keys()}")
print(f"Camera shape: {policy_obs['camera_0'].shape}")  # 应该是(1,8,3,240,320)
print(f"Force shape: {policy_obs['force'].shape}")      # 应该是(1,8,6)
print(f"Action shape: {action.shape}")                  # 应该是(16,7)
```

## ⚠️ 安全注意事项

1. **急停准备**：随时准备按下机器人急停按钮
2. **工作空间检查**：确保机器人周围无障碍物
3. **初始位置**：确保机器人初始位置安全
4. **力监控**：观察force数值，异常大的力表示可能碰撞
5. **速度限制**：首次运行建议降低频率（如5Hz）

## 📚 相关文档

- [JAKA_README.md](lerobot/common/robot_devices/robots/JAKA_README.md) - JAKA机器人详细说明
- [TEST_GUIDE.md](TEST_GUIDE.md) - 集成测试指南
- [FADP文档](../../Force-Aware-Diffusion-Policy/README.md) - FADP模型说明

## 🆘 获取帮助

如果遇到问题：
1. 检查上述故障排查部分
2. 查看相关文档
3. 检查终端输出的错误信息
4. 使用`python eval_jaka_fadp.py --help`查看帮助

## 🎉 成功运行的标志

如果一切正常，您应该看到：
```
================================================================================
JAKA机器人 + FADP模型推理
================================================================================
...
✓ 模型加载成功
✓ 所有设备已连接
✓ 预热完成

================================================================================
开始推理...
================================================================================
Step 0:
  推理时间: 50.2ms
  预测了 16 步action
  Action: x=0.123, y=-0.045, z=0.067
  实际频率: 9.8 Hz
...
```

祝您使用愉快！🚀
