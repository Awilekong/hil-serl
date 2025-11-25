#!/usr/bin/env python3
"""测试SpaceMouseExpert初始化"""

import sys
import signal
import time

# 设置超时处理
def timeout_handler(signum, frame):
    print("\n✗ 超时！SpaceMouseExpert 初始化超过15秒")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(15)  # 15秒超时

try:
    print("1. 添加路径并导入...")
    sys.path.insert(0, '/home/dexfranka/ws_zpw/hil-serl/serl_robot_infra')
    from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
    print("   ✓ 导入成功")
    
    print("\n2. 初始化 SpaceMouseExpert...")
    expert = SpaceMouseExpert()
    print("   ✓ 初始化成功")
    
    print("\n3. 测试获取动作...")
    for i in range(5):
        action, buttons = expert.get_action()
        print(f"   读取 {i+1}: action shape={len(action)}, buttons={buttons}")
        time.sleep(0.2)
    
    print("\n✓ SpaceMouseExpert 工作正常！")
    signal.alarm(0)  # 取消超时
    
except KeyboardInterrupt:
    print("\n用户中断")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
