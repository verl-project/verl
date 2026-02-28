#!/usr/bin/env python3
import sys
import os

# 清理可能的问题路径
sys.path = [p for p in sys.path if 'site-packages' in p or p == '']

print("Python版本:", sys.version)
print("Python路径前5个:")
for p in sys.path[:5]:
    print("  ", p)

print("\n尝试导入包...")
try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ torch: {e}")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ transformers: {e}")

try:
    import verl
    print(f"✓ verl 导入成功")
    if hasattr(verl, '__version__'):
        print(f"  verl版本: {verl.__version__}")
except Exception as e:
    print(f"✗ verl: {e}")

print("\n环境测试完成")
