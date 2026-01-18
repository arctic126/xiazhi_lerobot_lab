#!/usr/bin/env python3
"""
环境检查脚本 - 诊断CUDA和FADP环境问题
"""

import sys
import os
from pathlib import Path

print("="*80)
print("环境诊断工具")
print("="*80)

# 1. Python版本
print("\n1. Python环境")
print(f"   Python版本: {sys.version}")
print(f"   Python路径: {sys.executable}")

# 2. 检查PyTorch
print("\n2. PyTorch环境")
try:
    import torch
    print(f"   ✓ PyTorch已安装")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   PyTorch路径: {torch.__file__}")
    
    # 检查CUDA
    print("\n3. CUDA检查")
    print(f"   CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA可用")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"   ✗ CUDA不可用")
        print("\n   可能的原因:")
        print("   1. PyTorch为CPU版本 (需要重装CUDA版本)")
        print("   2. NVIDIA驱动未安装")
        print("   3. CUDA toolkit未安装")
        print("   4. CUDA_VISIBLE_DEVICES环境变量被设置为空")
        
        # 检查环境变量
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"\n   CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # 检查nvidia-smi
        print("\n   尝试运行nvidia-smi:")
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   ✓ nvidia-smi可用（驱动已安装）")
                print("\n" + result.stdout)
            else:
                print("   ✗ nvidia-smi失败")
        except FileNotFoundError:
            print("   ✗ nvidia-smi未找到（驱动可能未安装）")
        except Exception as e:
            print(f"   ✗ nvidia-smi错误: {e}")
            
        print("\n   解决方案:")
        print("   如果nvidia-smi可用但torch.cuda不可用，请重装PyTorch CUDA版本:")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
except ImportError:
    print("   ✗ PyTorch未安装")
    print("   请安装: pip install torch torchvision")

# 4. 检查FADP
print("\n4. FADP模块检查")
fadp_paths = [
    Path(__file__).parent.parent.parent / "Force-Aware-Diffusion-Policy",
    Path("/home/hyx/xiazhi/Force-Aware-Diffusion-Policy"),
]

fadp_found = False
for fadp_path in fadp_paths:
    print(f"   检查路径: {fadp_path}")
    if fadp_path.exists():
        print(f"   ✓ FADP目录存在")
        fadp_found = True
        
        # 检查fadp模块
        fadp_module = fadp_path / "fadp"
        if fadp_module.exists():
            print(f"   ✓ fadp模块目录存在: {fadp_module}")
            
            # 尝试导入
            sys.path.insert(0, str(fadp_path))
            try:
                import fadp
                print(f"   ✓ fadp模块可以导入")
                print(f"   fadp路径: {fadp.__file__}")
            except ImportError as e:
                print(f"   ✗ fadp模块导入失败: {e}")
        else:
            print(f"   ✗ fadp模块目录不存在: {fadp_module}")
        break
    else:
        print(f"   ✗ 目录不存在")

if not fadp_found:
    print("\n   ✗ 未找到FADP目录")
    print("   请确保Force-Aware-Diffusion-Policy目录存在于:")
    for p in fadp_paths:
        print(f"     - {p}")

# 5. 检查其他依赖
print("\n5. 其他依赖检查")
packages = {
    'cv2': 'opencv-python',
    'hydra': 'hydra-core',
    'diffusers': 'diffusers',
    'dill': 'dill',
}

for module_name, package_name in packages.items():
    try:
        __import__(module_name)
        print(f"   ✓ {package_name}")
    except ImportError:
        print(f"   ✗ {package_name} - 需要安装: pip install {package_name}")

# 6. 检查模型文件
print("\n6. 模型文件检查")
model_path = Path("/home/hyx/xiazhi/model/fadp_ckpt/2025.10.28/16.14.32_train_force_aware_diffusion_policy_real_forceumi1/checkpoints/latest.ckpt")
print(f"   模型路径: {model_path}")
if model_path.exists():
    print(f"   ✓ 模型文件存在")
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   文件大小: {size_mb:.1f} MB")
else:
    print(f"   ✗ 模型文件不存在")

print("\n" + "="*80)
print("诊断完成")
print("="*80)

# 7. 给出建议
print("\n建议:")
if not torch.cuda.is_available():
    print("⚠ CUDA不可用 - 推理会非常慢")
    print("   1. 检查nvidia-smi输出确认GPU可用")
    print("   2. 重装PyTorch CUDA版本")
    print("   3. 清除CUDA_VISIBLE_DEVICES环境变量")
else:
    print("✓ CUDA环境正常")

if not fadp_found:
    print("⚠ FADP模块未找到")
    print("   1. 确认Force-Aware-Diffusion-Policy目录存在")
    print("   2. 运行脚本会自动添加路径，但需要目录存在")
else:
    print("✓ FADP模块可用")

print("\n如果所有检查都通过，可以运行:")
print("python eval_jaka_fadp.py -c <checkpoint_path>")
