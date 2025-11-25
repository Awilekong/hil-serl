#!/usr/bin/env python3
"""检查RealSense相机连接状态"""

import pyrealsense2 as rs

def check_cameras():
    print("=" * 60)
    print("RealSense 相机诊断工具")
    print("=" * 60)
    
    try:
        # 创建context
        ctx = rs.context()
        devices = ctx.query_devices()
        
        print(f"\n✓ 检测到 {len(devices)} 个 RealSense 设备\n")
        
        if len(devices) == 0:
            print("⚠️  警告: 没有检测到任何RealSense相机！\n")
            print("请检查:")
            print("  1. 相机USB线是否已连接")
            print("  2. USB端口是否正常工作")
            print("  3. 是否有权限访问USB设备 (可能需要 sudo)")
            print("  4. RealSense驱动是否正确安装")
            return
        
        # 配置文件中期望的相机序列号
        expected_cameras = {
            "wrist_1": "323622271399",
            "wrist_2": "323622271298"
        }
        
        print("已连接的相机:")
        print("-" * 60)
        detected_serials = []
        for i, dev in enumerate(devices):
            serial = dev.get_info(rs.camera_info.serial_number)
            detected_serials.append(serial)
            name = dev.get_info(rs.camera_info.name)
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
            
            print(f"\n设备 {i+1}:")
            print(f"  型号: {name}")
            print(f"  序列号: {serial}")
            print(f"  USB类型: {usb_type}")
            
            # 检查是否匹配配置
            matched = False
            for cam_name, expected_serial in expected_cameras.items():
                if serial == expected_serial:
                    print(f"  ✓ 匹配配置: {cam_name}")
                    matched = True
                    break
            if not matched:
                print(f"  ⚠️  未在配置中找到此序列号")
        
        print("\n" + "=" * 60)
        print("配置文件中期望的相机:")
        print("-" * 60)
        
        all_found = True
        for cam_name, expected_serial in expected_cameras.items():
            if expected_serial in detected_serials:
                print(f"✓ {cam_name}: {expected_serial} - 已找到")
            else:
                print(f"✗ {cam_name}: {expected_serial} - 未找到")
                all_found = False
        
        print("\n" + "=" * 60)
        if all_found:
            print("✓ 所有相机都已正确连接！")
        else:
            print("⚠️  部分相机未连接或序列号不匹配")
            print("\n建议:")
            print("  1. 检查相机USB连接")
            print("  2. 更新config.py中的序列号以匹配实际设备")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        print("\n可能的原因:")
        print("  1. pyrealsense2 未正确安装")
        print("  2. RealSense SDK 未安装")
        print("  3. 权限不足")

if __name__ == "__main__":
    check_cameras()
