#!/usr/bin/env python3
"""测试SpaceMouse初始化"""

import sys
import signal
import time

# 设置超时处理
def timeout_handler(signum, frame):
    print("\n✗ 超时！SpaceMouse初始化超过10秒")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10秒超时

try:
    print("1. 开始导入 pyspacemouse...")
    sys.path.insert(0, '/home/dexfranka/ws_zpw/hil-serl/serl_robot_infra')
    from franka_env.spacemouse import pyspacemouse
    print("   ✓ 导入成功")
    
    print("\n2. 调用 pyspacemouse.open()...")
    pyspacemouse.open()
    print("   ✓ open() 成功")
    
    print("\n3. 尝试读取 SpaceMouse 数据...")
    for i in range(5):
        state = pyspacemouse.read_all()
        print(f"   读取 {i+1}: {len(state)} 个设备")
        if state:
            print(f"     - 第一个设备状态: x={state[0].x:.3f}, y={state[0].y:.3f}, z={state[0].z:.3f}")
        time.sleep(0.1)
    
    print("\n✓ SpaceMouse 工作正常！")
    signal.alarm(0)  # 取消超时
    
except KeyboardInterrupt:
    print("\n用户中断")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
