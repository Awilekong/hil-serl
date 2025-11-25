"""
SpaceMouse 实时诊断脚本 - 查找无法移动的方向

此脚本实时显示：
1. SpaceMouse 的原始输入
2. 机械臂的当前位置
3. 安全空间的限制
4. 是否触碰到限制
"""

import sys
import os
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'serl_robot_infra'))
sys.path.insert(0, os.path.join(project_root, 'serl_launcher'))

from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from experiments.mappings import CONFIG_MAPPING


def main():
    print("=" * 80)
    print("SpaceMouse 实时诊断脚本")
    print("=" * 80)
    
    # 获取环境
    exp_name = 'ram_insertion'
    config = CONFIG_MAPPING[exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    
    # 获取配置
    env_config = env.unwrapped.config
    action_scale = env.unwrapped.action_scale
    
    print(f"\n当前 ACTION_SCALE: {action_scale}")
    print(f"\n安全空间限制:")
    print(f"  X: [{env_config.ABS_POSE_LIMIT_LOW[0]:.3f}, {env_config.ABS_POSE_LIMIT_HIGH[0]:.3f}] (范围: {env_config.ABS_POSE_LIMIT_HIGH[0] - env_config.ABS_POSE_LIMIT_LOW[0]:.3f}m)")
    print(f"  Y: [{env_config.ABS_POSE_LIMIT_LOW[1]:.3f}, {env_config.ABS_POSE_LIMIT_HIGH[1]:.3f}] (范围: {env_config.ABS_POSE_LIMIT_HIGH[1] - env_config.ABS_POSE_LIMIT_LOW[1]:.3f}m)")
    print(f"  Z: [{env_config.ABS_POSE_LIMIT_LOW[2]:.3f}, {env_config.ABS_POSE_LIMIT_HIGH[2]:.3f}] (范围: {env_config.ABS_POSE_LIMIT_HIGH[2] - env_config.ABS_POSE_LIMIT_LOW[2]:.3f}m)")
    
    # 初始化 SpaceMouse
    print("\n初始化 SpaceMouse...")
    expert = SpaceMouseExpert()
    print("SpaceMouse 已连接!")
    
    # Reset 环境
    obs, _ = env.reset()
    
    print("\n开始实时监控...")
    print("请移动 SpaceMouse，查看各个方向的响应")
    print("按 Ctrl+C 退出\n")
    
    try:
        step = 0
        while True:
            step += 1
            
            # 获取 SpaceMouse 输入
            spacemouse_action, buttons = expert.get_action()
            
            # 执行动作
            action = np.zeros(env.action_space.sample().shape)
            next_obs, _, _, _, info = env.step(action)
            
            if "intervene_action" in info:
                action = info["intervene_action"]
            
            # 获取当前位置
            tcp_pose = env.unwrapped.currpos
            tcp_pos = tcp_pose[:3]
            tcp_euler = R.from_quat(tcp_pose[3:]).as_euler('xyz')
            
            # 检查是否有明显的 SpaceMouse 输入
            if np.linalg.norm(spacemouse_action) > 0.05:
                print("\n" + "=" * 80)
                print(f"Step {step}")
                print("-" * 80)
                
                # SpaceMouse 输入
                print(f"SpaceMouse 输入:")
                print(f"  位置: X={spacemouse_action[0]:+.3f}, Y={spacemouse_action[1]:+.3f}, Z={spacemouse_action[2]:+.3f}")
                print(f"  旋转: Roll={spacemouse_action[3]:+.3f}, Pitch={spacemouse_action[4]:+.3f}, Yaw={spacemouse_action[5]:+.3f}")
                
                # 实际动作（应用 ACTION_SCALE）
                print(f"\n实际动作 (经过 ACTION_SCALE):")
                if len(action) >= 6:
                    print(f"  位置: X={action[0]:+.5f}, Y={action[1]:+.5f}, Z={action[2]:+.5f}")
                    print(f"  旋转: Roll={action[3]:+.5f}, Pitch={action[4]:+.5f}, Yaw={action[5]:+.5f}")
                
                # 当前位置
                print(f"\n当前末端位置:")
                print(f"  位置: X={tcp_pos[0]:.4f}, Y={tcp_pos[1]:.4f}, Z={tcp_pos[2]:.4f}")
                print(f"  姿态: Roll={tcp_euler[0]:.4f}, Pitch={tcp_euler[1]:.4f}, Yaw={tcp_euler[2]:.4f}")
                
                # 检查是否触碰边界
                print(f"\n距离边界:")
                margin_low_x = tcp_pos[0] - env_config.ABS_POSE_LIMIT_LOW[0]
                margin_high_x = env_config.ABS_POSE_LIMIT_HIGH[0] - tcp_pos[0]
                margin_low_y = tcp_pos[1] - env_config.ABS_POSE_LIMIT_LOW[1]
                margin_high_y = env_config.ABS_POSE_LIMIT_HIGH[1] - tcp_pos[1]
                margin_low_z = tcp_pos[2] - env_config.ABS_POSE_LIMIT_LOW[2]
                margin_high_z = env_config.ABS_POSE_LIMIT_HIGH[2] - tcp_pos[2]
                
                def format_margin(margin, direction):
                    if margin < 0.005:  # 小于5mm，红色警告
                        return f"  {direction}: {margin*1000:.1f}mm ⚠️  触碰边界!"
                    elif margin < 0.02:  # 小于20mm，黄色警告
                        return f"  {direction}: {margin*1000:.1f}mm ⚠️"
                    else:
                        return f"  {direction}: {margin*1000:.1f}mm"
                
                print(format_margin(margin_low_x, "X 负方向 (左)"))
                print(format_margin(margin_high_x, "X 正方向 (右)"))
                print(format_margin(margin_low_y, "Y 负方向 (后)"))
                print(format_margin(margin_high_y, "Y 正方向 (前)"))
                print(format_margin(margin_low_z, "Z 负方向 (下)"))
                print(format_margin(margin_high_z, "Z 正方向 (上)"))
            
            obs = next_obs
            time.sleep(0.02)  # 50Hz
            
    except KeyboardInterrupt:
        print("\n\n退出...")
        expert.close()
    
    print("\n诊断完成!")


if __name__ == "__main__":
    main()
