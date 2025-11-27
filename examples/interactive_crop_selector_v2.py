#!/usr/bin/env python3
"""
Interactive crop selector for RealSense cameras.
Uses the existing RSCapture class from the project.
"""

import cv2
import numpy as np
import sys
import os
import re
import time
from pathlib import Path

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'serl_robot_infra'))

from franka_env.camera.rs_capture import RSCapture

class CropSelector:
    def __init__(self, camera_name, serial_number, dimensions, exposure):
        self.camera_name = camera_name
        self.serial_number = serial_number
        self.dimensions = dimensions
        self.exposure = exposure
        self.image = None
        self.crop_coords = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_image = None
        self.camera = None
        
    def initialize_camera(self, max_retries=3):
        """Initialize the RealSense camera using RSCapture with retry logic"""
        print(f"\n初始化相机 {self.camera_name} (序列号: {self.serial_number})...")
        print(f"分辨率: {self.dimensions}, 曝光: {self.exposure}")
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"重试 {attempt}/{max_retries-1}...")
                    time.sleep(2)  # Wait before retry
                
                self.camera = RSCapture(
                    name=self.camera_name,
                    serial_number=self.serial_number,
                    dim=self.dimensions,
                    fps=15,
                    depth=False,
                    exposure=self.exposure
                )
                print(f"✓ 相机 {self.camera_name} 初始化成功!")
                
                # Wait for camera to stabilize
                print("等待相机稳定...")
                stable_frames = 0
                for i in range(10):  # Try up to 10 frames
                    ret, _ = self.camera.read()
                    if ret:
                        stable_frames += 1
                        if stable_frames <= 5:  # Only print first 5
                            print(f"  帧 {stable_frames}/5 成功")
                        if stable_frames >= 5:
                            break
                    time.sleep(0.2)
                
                if stable_frames >= 5:
                    return True
                else:
                    print(f"⚠️  只获取到 {stable_frames} 帧，继续尝试...")
                    if self.camera:
                        self.camera.close()
                        self.camera = None
                        time.sleep(1)
                
            except Exception as e:
                print(f"✗ 初始化失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if self.camera:
                    try:
                        self.camera.close()
                    except:
                        pass
                    self.camera = None
                
                if attempt < max_retries - 1:
                    print("等待后重试...")
                    time.sleep(3)
                else:
                    import traceback
                    traceback.print_exc()
        
        return False
    
    def capture_image(self):
        """Capture a single image from the camera"""
        if self.camera is None:
            return None
        
        ret, image = self.camera.read()
        if ret:
            self.image = image
            return image
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing crop rectangle"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # Calculate crop coordinates (y1:y2, x1:x2)
            x1 = min(self.start_point[0], self.end_point[0])
            x2 = max(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            y2 = max(self.start_point[1], self.end_point[1])
            
            self.crop_coords = (y1, y2, x1, x2)
            print(f"\n选中区域: img[{y1}:{y2}, {x1}:{x2}]")
    
    def select_crop_region(self):
        """Interactive crop region selection"""
        if self.image is None:
            print("错误: 没有图像可用")
            return None
        
        window_name = f"{self.camera_name}_crop"
        
        # Create window and show initial image first
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, self.image)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.waitKey(100)  # Wait for window to be created
        
        # Now set the mouse callback
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print(f"\n{self.camera_name}:")
        print("  - 用鼠标框选要裁剪的区域")
        print("  - 按 'r' 重置选择")
        print("  - 按 'q' 确认并继续")
        print("  - 按 's' 显示实时画面（持续更新）")
        
        live_mode = False
        
        while True:
            # In live mode, continuously capture new frames
            if live_mode:
                self.capture_image()
            
            if self.image is None:
                print("无法获取图像")
                break
            
            # Create a copy for display
            self.temp_image = self.image.copy()
            
            # Draw the rectangle if we're drawing or have selected
            if self.start_point and self.end_point:
                cv2.rectangle(
                    self.temp_image,
                    self.start_point,
                    self.end_point,
                    (0, 255, 0),
                    2
                )
                
                # Show coordinates
                if self.crop_coords:
                    y1, y2, x1, x2 = self.crop_coords
                    text = f"[{y1}:{y2}, {x1}:{x2}]"
                    cv2.putText(
                        self.temp_image,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
            
            # Show mode indicator
            mode_text = "LIVE" if live_mode else "FROZEN"
            cv2.putText(
                self.temp_image,
                mode_text,
                (10, self.temp_image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255) if live_mode else (128, 128, 128),
                2
            )
            
            cv2.imshow(window_name, self.temp_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                self.start_point = None
                self.end_point = None
                self.crop_coords = None
                print("已重置选择")
            
            elif key == ord('s'):  # Toggle live mode
                live_mode = not live_mode
                print(f"{'启用' if live_mode else '禁用'}实时画面")
            
            elif key == ord('q'):  # Confirm
                if self.crop_coords:
                    cv2.destroyWindow(window_name)
                    return self.crop_coords
                else:
                    print("请先框选区域!")
        
    def cleanup(self):
        """Stop the camera"""
        try:
            if self.camera is not None:
                print(f"正在关闭相机 {self.camera_name}...")
                self.camera.close()
                print(f"✓ 相机 {self.camera_name} 已关闭")
                self.camera = None
                
                # Important: Wait for the camera to fully release
                print("等待相机资源完全释放...")
                time.sleep(2)  # Give time for camera to fully disconnect
                
        except Exception as e:
            print(f"关闭相机时出错: {e}")
        
        cv2.destroyAllWindows()
        # Additional wait after closing windows
        time.sleep(0.5)


def parse_config_file(config_path):
    """Parse the config file to extract camera configurations"""
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find REALSENSE_CAMERAS dict
    cameras = {}
    camera_pattern = r'"(\w+)":\s*\{[^}]*"serial_number":\s*"(\d+)"[^}]*"dim":\s*\((\d+),\s*(\d+)\)[^}]*"exposure":\s*(\d+)'
    
    for match in re.finditer(camera_pattern, content):
        camera_name = match.group(1)
        serial_number = match.group(2)
        width = int(match.group(3))
        height = int(match.group(4))
        exposure = int(match.group(5))
        
        cameras[camera_name] = {
            'serial_number': serial_number,
            'dim': (width, height),
            'exposure': exposure
        }
    
    return cameras, content


def update_config_file(config_path, content, crop_results, all_cameras):
    """Update the config file with new crop coordinates"""
    # Find the IMAGE_CROP section
    image_crop_pattern = r'IMAGE_CROP\s*=\s*\{([^}]+)\}'
    match = re.search(image_crop_pattern, content, re.DOTALL)
    
    if not match:
        print("错误: 未找到 IMAGE_CROP 配置")
        return False
    
    # Extract existing crop values
    existing_crops = {}
    existing_section = match.group(1)
    existing_pattern = r'"(\w+)":\s*lambda\s+img:\s*img\[(\d+):(\d+),\s*(\d+):(\d+)\]'
    for match in re.finditer(existing_pattern, existing_section):
        cam_name = match.group(1)
        y1, y2, x1, x2 = int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))
        existing_crops[cam_name] = (y1, y2, x1, x2)
    
    # Build new IMAGE_CROP dict, keeping old values for cameras that weren't updated
    new_crop_lines = []
    for camera_name in all_cameras.keys():
        if camera_name in crop_results:
            # Use new crop
            y1, y2, x1, x2 = crop_results[camera_name]
            new_crop_lines.append(f'        "{camera_name}": lambda img: img[{y1}:{y2}, {x1}:{x2}],')
        elif camera_name in existing_crops:
            # Keep old crop
            y1, y2, x1, x2 = existing_crops[camera_name]
            new_crop_lines.append(f'        "{camera_name}": lambda img: img[{y1}:{y2}, {x1}:{x2}],')
            print(f"  {camera_name}: 保留原有裁剪 img[{y1}:{y2}, {x1}:{x2}]")
    
    new_crop_section = "IMAGE_CROP = {\n" + "\n".join(new_crop_lines) + "\n    }"
    
    # Replace the old IMAGE_CROP section
    new_content = re.sub(
        r'IMAGE_CROP\s*=\s*\{[^}]+\}',
        new_crop_section,
        content,
        flags=re.DOTALL
    )
    
    # Write back to file
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True


def main():
    print("=" * 60)
    print("RealSense 相机裁剪区域交互选择工具 v2")
    print("=" * 60)
    print("\n⚠️  重要提示:")
    print("  - 请确保没有其他程序在使用相机（如 realsense-viewer）")
    print("  - 如果相机被占用，请先关闭其他程序\n")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        config_path = os.path.join(
            os.path.dirname(__file__),
            'experiments',
            exp_name,
            'config.py'
        )
    else:
        # Default to ram_insertion
        config_path = os.path.join(
            os.path.dirname(__file__),
            'experiments',
            'ram_insertion',
            'config.py'
        )
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        print("用法: python interactive_crop_selector_v2.py [exp_name]")
        print("例如: python interactive_crop_selector_v2.py ram_insertion")
        return
    
    print(f"配置文件: {config_path}")
    
    # Parse config
    cameras, original_content = parse_config_file(config_path)
    print(f"\n找到 {len(cameras)} 个相机配置:")
    for name, config in cameras.items():
        print(f"  - {name}: {config['serial_number']}")
    
    # Check which cameras are actually connected
    print("\n检查连接的相机...")
    try:
        available_serials = RSCapture.get_device_serial_numbers()
        print(f"检测到 {len(available_serials)} 个RealSense设备:")
        for sn in available_serials:
            print(f"  ✓ {sn}")
        
        # Check which configured cameras are available
        missing_cameras = []
        for name, config in cameras.items():
            if config['serial_number'] not in available_serials:
                missing_cameras.append(name)
        
        if missing_cameras:
            print(f"\n⚠️  警告: 以下相机未连接:")
            for name in missing_cameras:
                print(f"  - {name} (序列号: {cameras[name]['serial_number']})")
            print("\n将只处理已连接的相机。")
            
            user_input = input("\n是否继续? (y/n): ").strip().lower()
            if user_input != 'y':
                print("已取消")
                return
    except Exception as e:
        print(f"⚠️  无法检查相机: {e}")
        print("将尝试直接初始化相机...")
    
    # Process each camera
    crop_results = {}
    camera_count = len(cameras)
    
    for idx, (camera_name, camera_config) in enumerate(cameras.items(), 1):
        print("\n" + "=" * 60)
        print(f"处理相机 {idx}/{camera_count}: {camera_name}")
        print("=" * 60)
        
        selector = CropSelector(
            camera_name,
            camera_config['serial_number'],
            camera_config['dim'],
            camera_config['exposure']
        )
        
        try:
            # Initialize camera
            if not selector.initialize_camera():
                print(f"⚠️  跳过相机 {camera_name} (初始化失败)")
                continue
            
            # Capture initial images to ensure stability
            print("捕获初始图像...")
            success_count = 0
            for i in range(5):
                if selector.capture_image() is not None:
                    success_count += 1
                    if success_count <= 3:  # Only print first 3
                        print(f"  ✓ 图像 {success_count}/3")
                    if success_count >= 3:
                        break
                time.sleep(0.15)
            
            if success_count < 3:
                print(f"⚠️  警告: 只捕获到 {success_count} 张图像，可能不稳定")
                user_input = input("是否继续? (y/n): ").strip().lower()
                if user_input != 'y':
                    print(f"跳过相机 {camera_name}")
                    continue
            
            # Interactive selection
            crop_coords = selector.select_crop_region()
            
            if crop_coords:
                crop_results[camera_name] = crop_coords
                y1, y2, x1, x2 = crop_coords
                print(f"✓ {camera_name} 裁剪区域已确认: img[{y1}:{y2}, {x1}:{x2}]")
                
                # Show cropped preview
                if selector.image is not None:
                    cropped = selector.image[y1:y2, x1:x2]
                    print(f"  裁剪后尺寸: {cropped.shape[1]}x{cropped.shape[0]}")
            
        except Exception as e:
            print(f"错误: 处理相机 {camera_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Make sure to cleanup before moving to next camera
            selector.cleanup()
            
            # Add extra delay between cameras (important for USB bandwidth)
            if idx < camera_count:  # Not the last camera
                print(f"\n等待 3 秒后处理下一个相机...")
                time.sleep(3)
    
    # Update config file
    if crop_results:
        print("\n" + "=" * 60)
        print("更新配置文件...")
        print("=" * 60)
        
        for camera_name, (y1, y2, x1, x2) in crop_results.items():
            print(f'  {camera_name}: lambda img: img[{y1}:{y2}, {x1}:{x2}] (新)')
        
        if update_config_file(config_path, original_content, crop_results, cameras):
            print(f"\n✓ 配置文件已更新: {config_path}")
        else:
            print("\n✗ 更新配置文件失败")
    else:
        print("\n未选择任何裁剪区域，配置文件未更新")
    
    # Print summary
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    success_count = len(crop_results)
    total_count = len(cameras)
    print(f"成功处理: {success_count}/{total_count} 个相机")
    
    if success_count < total_count:
        print("\n失败的相机:")
        for cam_name in cameras.keys():
            if cam_name not in crop_results:
                print(f"  - {cam_name} (序列号: {cameras[cam_name]['serial_number']})")
        
        print("\n可能的原因:")
        print("  1. 相机被其他程序占用（如 realsense-viewer）")
        print("  2. USB 带宽不足（尝试只连接一个相机）")
        print("  3. USB 线缆或连接不稳定")
        print("  4. 相机驱动或固件问题")
        
        print("\n建议:")
        print("  - 确保关闭所有使用相机的程序")
        print("  - 尝试重新插拔相机 USB 线")
        print("  - 尝试单独运行脚本处理失败的相机")
        print("  - 检查 'rs-enumerate-devices' 命令的输出")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
