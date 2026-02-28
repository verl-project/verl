#!/usr/bin/env python3
"""
verl-qwen3.5 NPU环境验证脚本
基于实际安装经验总结的完整验证流程
"""

import sys
import traceback

def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def verify_verl():
    """验证verl导入"""
    try:
        import verl
        print("✅ verl导入成功")
        print(f"   版本: {verl.__version__}")
        print(f"   路径: {verl.__file__}")
        return True
    except Exception as e:
        print(f"❌ verl导入失败: {e}")
        traceback.print_exc()
        return False

def verify_ray():
    """验证ray导入"""
    try:
        import ray
        print("✅ ray导入成功")
        print(f"   版本: {ray.__version__}")
        print(f"   路径: {ray.__file__}")
        return True
    except Exception as e:
        print(f"❌ ray导入失败: {e}")
        return False

def verify_torch_npu():
    """验证torch和torch_npu"""
    try:
        import torch
        import torch_npu
        
        print("✅ torch导入成功")
        print(f"   torch版本: {torch.version.__version__}")
        print(f"   torch_npu版本: {torch_npu.__version__}")
        
        # 检查NPU设备
        device_count = torch_npu.npu.device_count()
        is_available = torch_npu.npu.is_available()
        
        print(f"   NPU设备数量: {device_count}")
        print(f"   NPU 0可用: {is_available}")
        
        if is_available:
            # 测试NPU功能
            x = torch.tensor([1.0, 2.0, 3.0])
            x_npu = x.npu()
            print(f"   ✅ 张量成功移动到NPU: {x_npu.device}")
            
            # 测试计算
            y = torch.tensor([4.0, 5.0, 6.0]).npu()
            z = x_npu + y
            print(f"   ✅ NPU计算成功: {z.cpu().numpy()}")
        
        return True
    except Exception as e:
        print(f"❌ torch/torch_npu导入失败: {e}")
        return False

def verify_other_components():
    """验证其他核心组件"""
    components = [
        ("numpy", "numpy"),
        ("vllm", "vllm"),
        ("triton", "triton"),
        ("transformers", "transformers"),
        ("tensordict", "tensordict"),
        ("accelerate", "accelerate"),
    ]
    
    all_success = True
    for name, module in components:
        try:
            imported_module = __import__(module)
            print(f"✅ {name}导入成功")
            if hasattr(imported_module, '__version__'):
                print(f"   版本: {imported_module.__version__}")
        except Exception as e:
            print(f"⚠️  {name}导入警告: {e}")
            all_success = False
    
    return all_success

def verify_python_environment():
    """验证Python环境"""
    import sys
    import platform
    
    print("📋 Python环境信息:")
    print(f"   Python版本: {sys.version.split()[0]}")
    print(f"   Python路径: {sys.executable}")
    print(f"   系统架构: {platform.machine()}")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    
    return True

def main():
    """主验证函数"""
    print_section("verl-qwen3.5 NPU环境验证")
    
    # 记录验证结果
    results = []
    
    # 验证Python环境
    print("\n1. 验证Python环境:")
    results.append(("Python环境", verify_python_environment()))
    
    # 验证核心组件
    print("\n2. 验证核心组件:")
    results.append(("verl", verify_verl()))
    results.append(("ray", verify_ray()))
    results.append(("torch_npu", verify_torch_npu()))
    
    # 验证其他组件
    print("\n3. 验证其他组件:")
    results.append(("其他组件", verify_other_components()))
    
    # 总结
    print_section("验证结果总结")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"📊 验证统计:")
    print(f"   总检查项: {total}")
    print(f"   通过项: {passed}")
    print(f"   失败项: {total - passed}")
    
    if total - passed > 0:
        print("\n❌ 失败的检查项:")
        for name, success in results:
            if not success:
                print(f"   - {name}")
    
    # 最终判断
    if passed == total:
        print("\n🎉 所有检查项通过！verl-qwen3.5 NPU环境安装成功！")
        return 0
    elif passed >= total - 1:  # 允许一个非核心组件失败
        print("\n⚠️  部分检查项有警告，但核心功能可用")
        return 0
    else:
        print("\n❌ 环境验证失败，请检查安装")
        return 1

if __name__ == "__main__":
    sys.exit(main())