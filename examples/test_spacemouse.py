"""
SpaceMouse 测试和调试脚本

此脚本用于测试 SpaceMouse 的控制映射，帮助诊断遥操问题。
运行此脚本可以实时查看 SpaceMouse 的原始输出和经过变换后的动作。
"""

import sys
import os
import numpy as np
import time

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'serl_robot_infra'))

from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert


def main():
    print("=" * 60)
    print("SpaceMouse 测试脚本")
    print("=" * 60)
    print("\n初始化 SpaceMouse...")
    
    expert = SpaceMouseExpert()
    print("SpaceMouse 已连接!")
    print("\n请移动 SpaceMouse 来测试各个方向的响应:")
    print("- 前后移动 (Y轴)")
    print("- 左右移动 (X轴)")
    print("- 上下移动 (Z轴)")
    print("- 旋转 (Roll, Pitch, Yaw)")
    print("\n按 Ctrl+C 退出\n")
    
    try:
        while True:
            action, buttons = expert.get_action()
            
            # 只显示有明显移动的数据
            if np.linalg.norm(action) > 0.01:
                print("\n" + "=" * 60)
                print(f"原始 SpaceMouse 输出:")
                print(f"  位置: X={action[0]:+.3f}, Y={action[1]:+.3f}, Z={action[2]:+.3f}")
                print(f"  旋转: Roll={action[3]:+.3f}, Pitch={action[4]:+.3f}, Yaw={action[5]:+.3f}")
                print(f"  按钮: {buttons}")
                
                # 模拟应用 ACTION_SCALE
                pos_scale = 0.01  # 当前配置
                rot_scale = 0.06  # 当前配置
                scaled_pos = action[:3] * pos_scale
                scaled_rot = action[3:6] * rot_scale
                
                print(f"\n应用当前 ACTION_SCALE (pos={pos_scale}, rot={rot_scale}) 后:")
                print(f"  位置: X={scaled_pos[0]:+.5f}, Y={scaled_pos[1]:+.5f}, Z={scaled_pos[2]:+.5f}")
                print(f"  旋转: Roll={scaled_rot[0]:+.5f}, Pitch={scaled_rot[1]:+.5f}, Yaw={scaled_rot[2]:+.5f}")
                
                # 建议更好的缩放值
                better_pos_scale = 0.05  # 5倍于当前
                better_rot_scale = 0.08  # 稍微增加
                better_scaled_pos = action[:3] * better_pos_scale
                better_scaled_rot = action[3:6] * better_rot_scale
                
                print(f"\n建议的 ACTION_SCALE (pos={better_pos_scale}, rot={better_rot_scale}) 后:")
                print(f"  位置: X={better_scaled_pos[0]:+.5f}, Y={better_scaled_pos[1]:+.5f}, Z={better_scaled_pos[2]:+.5f}")
                print(f"  旋转: Roll={better_scaled_rot[0]:+.5f}, Pitch={better_scaled_rot[1]:+.5f}, Yaw={better_scaled_rot[2]:+.5f}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n退出...")
        expert.close()
        
    print("\n建议修改:")
    print("1. 在 config.py 中将 ACTION_SCALE 从 (0.01, 0.06, 1) 改为 (0.05, 0.08, 1)")
    print("2. 如果某个轴的方向反了，在 spacemouse_expert.py 中调整符号")
    print("3. 如果旋转控制困难，可以进一步增大旋转缩放值")


if __name__ == "__main__":
    main()
