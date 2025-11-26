#!/usr/bin/env python3
"""
相机诊断工具 - 帮助诊断 RealSense 相机连接问题
"""

import sys
import os
import time

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'serl_robot_infra'))

import pyrealsense2 as rs
from franka_env.camera.rs_capture import RSCapture

def check_cameras():
    """检查所有连接的相机"""
    print("=" * 60)
    print("RealSense 相机诊断工具")
    print("=" * 60)
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    print(f"\n检测到 {len(devices)} 个 RealSense 设备:")
    
    if len(devices) == 0:
        print("  ❌ 未检测到任何相机！")
        print("  请检查:")
        print("    - 相机是否正确连接到 USB 端口")
        print("    - USB 线缆是否正常")
        print("    - 是否安装了 RealSense SDK")
        return []
    
    camera_info = []
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        fw_version = dev.get_info(rs.camera_info.firmware_version)
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
        
        print(f"\n相机 {i+1}:")
        print(f"  型号: {name}")
        print(f"  序列号: {serial}")
        print(f"  固件版本: {fw_version}")
        print(f"  USB 类型: {usb_type}")
        
        # Get supported streams
        print(f"  支持的流:")
        for sensor in dev.sensors:
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"    - {sensor_name}")
            
            # Get supported resolutions
            profiles = sensor.profiles
            color_resolutions = set()
            for profile in profiles:
                if profile.stream_type() == rs.stream.color and profile.format() == rs.format.bgr8:
                    vp = profile.as_video_stream_profile()
                    color_resolutions.add((vp.width(), vp.height(), vp.fps()))
            
            if color_resolutions:
                print(f"      彩色流支持的分辨率:")
                for width, height, fps in sorted(color_resolutions, reverse=True):
                    print(f"        {width}x{height} @ {fps}fps")
        
        camera_info.append({
            'serial': serial,
            'name': name,
            'fw_version': fw_version,
            'usb_type': usb_type
        })
    
    return camera_info


def test_camera(serial_number, resolution=(1280, 720), exposure=3000):
    """测试单个相机"""
    print("\n" + "=" * 60)
    print(f"测试相机: {serial_number}")
    print("=" * 60)
    
    try:
        print(f"初始化相机...")
        print(f"  分辨率: {resolution}")
        print(f"  曝光: {exposure}")
        
        camera = RSCapture(
            name="test",
            serial_number=serial_number,
            dim=resolution,
            fps=15,
            depth=False,
            exposure=exposure
        )
        
        print("✓ 相机初始化成功")
        
        # Try to capture frames
        print("\n尝试捕获 10 帧...")
        success_count = 0
        for i in range(10):
            try:
                ret, img = camera.read()
                if ret:
                    success_count += 1
                    print(f"  帧 {i+1}/10: ✓ (尺寸: {img.shape})")
                else:
                    print(f"  帧 {i+1}/10: ✗ (返回 False)")
                time.sleep(0.1)
            except Exception as e:
                print(f"  帧 {i+1}/10: ✗ (错误: {e})")
        
        print(f"\n成功率: {success_count}/10")
        
        camera.close()
        print("✓ 相机已关闭")
        
        return success_count == 10
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Check all cameras
    camera_info = check_cameras()
    
    if not camera_info:
        return
    
    # Test each camera
    print("\n" + "=" * 60)
    print("开始逐个测试相机")
    print("=" * 60)
    
    results = {}
    for info in camera_info:
        serial = info['serial']
        success = test_camera(serial)
        results[serial] = success
        time.sleep(1)  # Wait between cameras
    
    # Summary
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for serial, success in results.items():
        status = "✓ 正常" if success else "✗ 失败"
        print(f"  {serial}: {status}")
    
    failed_cameras = [s for s, success in results.items() if not success]
    if failed_cameras:
        print("\n失败的相机:")
        for serial in failed_cameras:
            print(f"  - {serial}")
        
        print("\n故障排除建议:")
        print("  1. 尝试重新插拔该相机的 USB 线")
        print("  2. 尝试连接到不同的 USB 端口")
        print("  3. 确保没有其他程序在使用该相机")
        print("  4. 检查 USB 线缆质量")
        print("  5. 如果使用 USB Hub，尝试直接连接到电脑")
        print("  6. 重启电脑")


if __name__ == "__main__":
    main()
