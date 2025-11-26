#!/usr/bin/env python3
"""
Interactive crop selector for RealSense cameras.
Displays images from all cameras defined in config and allows manual selection of crop regions.
Automatically updates the config file with the new crop values.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import sys
import os
import re
from pathlib import Path

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'serl_robot_infra'))

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
        
    def initialize_camera(self):
        """Initialize the RealSense camera"""
        print(f"\n初始化相机 {self.camera_name} (序列号: {self.serial_number})...")
        
        # First, check if the camera is available
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            raise RuntimeError("未检测到任何 RealSense 相机！请检查相机连接。")
        
        print(f"检测到 {len(devices)} 个 RealSense 设备:")
        device_found = None
        for dev in devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            print(f"  - {name} (序列号: {serial})")
            if serial == self.serial_number:
                device_found = dev
        
        if not device_found:
            raise RuntimeError(f"未找到序列号为 {self.serial_number} 的相机！")
        
        # Query supported stream profiles using pipeline config
        print(f"查询相机支持的分辨率...")
        temp_pipeline = rs.pipeline()
        temp_config = rs.config()
        temp_config.enable_device(self.serial_number)
        
        # Get pipeline profile to query available streams
        pipeline_profile = temp_config.resolve(temp_pipeline)
        
        supported_resolutions = set()
        for stream in pipeline_profile.get_streams():
            if stream.stream_type() == rs.stream.color:
                video_stream = stream.as_video_stream_profile()
                supported_resolutions.add((video_stream.width(), video_stream.height()))
        
        # Also query from device sensors
        for sensor in device_found.sensors:
            for profile in sensor.profiles:
                if profile.stream_type() == rs.stream.color and profile.format() == rs.format.bgr8:
                    video_profile = profile.as_video_stream_profile()
                    supported_resolutions.add((video_profile.width(), video_profile.height()))
        
        if supported_resolutions:
            print(f"支持的分辨率: {sorted(supported_resolutions, reverse=True)}")
            
            # Try to use the configured resolution, or fall back to the highest available
            if self.dimensions in supported_resolutions:
                use_width, use_height = self.dimensions
                print(f"使用配置的分辨率: {use_width}x{use_height}")
            else:
                # Use the highest resolution available
                use_width, use_height = max(supported_resolutions)
                print(f"配置的分辨率 {self.dimensions} 不支持，使用最高分辨率: {use_width}x{use_height}")
                self.dimensions = (use_width, use_height)
        else:
            # Fallback to common resolutions for D405
            print("无法查询支持的分辨率，尝试常用分辨率...")
            common_resolutions = [(1280, 720), (848, 480), (640, 480), (640, 360)]
            use_width, use_height = common_resolutions[0]
            print(f"使用默认分辨率: {use_width}x{use_height}")
            self.dimensions = (use_width, use_height)
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable the specific camera by serial number
        self.config.enable_device(self.serial_number)
        self.config.enable_stream(
            rs.stream.color,
            use_width,
            use_height,
            rs.format.bgr8,
            15  # Use lower FPS like in the project
        )
        
        # Start pipeline
        print(f"启动相机流...")
        profile = self.pipeline.start(self.config)
        
        # Set exposure using the project's method
        try:
            # Use query_sensors()[0] like in the project code
            sensor = profile.get_device().query_sensors()[0]
            sensor.set_option(rs.option.exposure, self.exposure)
            print(f"已设置曝光值: {self.exposure}")
        except Exception as e:
            print(f"设置曝光失败，使用默认设置: {e}")
        
        print(f"相机 {self.camera_name} 初始化成功!")
        
        # Wait for camera to stabilize (reduce wait time)
        print(f"等待相机稳定...")
        for i in range(10):
            try:
                self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print(f"等待第 {i+1} 帧失败: {e}")
                if i < 3:  # Only retry a few times
                    continue
                else:
                    raise
    
    def capture_image(self):
        """Capture a single image from the camera"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None
        
        # Convert to numpy array
        self.image = np.asanyarray(color_frame.get_data())
        return self.image
    
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
        
        window_name = f"{self.camera_name} - 框选裁剪区域 (按 'r' 重置, 'q' 确认)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print(f"\n{self.camera_name}:")
        print("  - 用鼠标框选要裁剪的区域")
        print("  - 按 'r' 重置选择")
        print("  - 按 'q' 确认并继续")
        
        while True:
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
            
            cv2.imshow(window_name, self.temp_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                self.start_point = None
                self.end_point = None
                self.crop_coords = None
                print("已重置选择")
            
            elif key == ord('q'):  # Confirm
                if self.crop_coords:
                    cv2.destroyWindow(window_name)
                    return self.crop_coords
                else:
                    print("请先框选区域!")
        
    def cleanup(self):
        """Stop the camera pipeline"""
        try:
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
                print(f"相机 {self.camera_name} 已关闭")
        except Exception as e:
            print(f"关闭相机时出错: {e}")
        cv2.destroyAllWindows()


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


def update_config_file(config_path, content, crop_results):
    """Update the config file with new crop coordinates"""
    # Find the IMAGE_CROP section
    image_crop_pattern = r'IMAGE_CROP\s*=\s*\{([^}]+)\}'
    match = re.search(image_crop_pattern, content, re.DOTALL)
    
    if not match:
        print("错误: 未找到 IMAGE_CROP 配置")
        return False
    
    # Build new IMAGE_CROP dict
    new_crop_lines = []
    for camera_name, (y1, y2, x1, x2) in crop_results.items():
        new_crop_lines.append(f'        "{camera_name}": lambda img: img[{y1}:{y2}, {x1}:{x2}],')
    
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
        print("用法: python interactive_crop_selector.py [exp_name]")
        print("例如: python interactive_crop_selector.py ram_insertion")
        return
    
    print("=" * 60)
    print("RealSense 相机裁剪区域交互选择工具")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    
    # Parse config
    cameras, original_content = parse_config_file(config_path)
    print(f"\n找到 {len(cameras)} 个相机:")
    for name, config in cameras.items():
        print(f"  - {name}: {config['serial_number']}")
    
    # Process each camera
    crop_results = {}
    
    for camera_name, camera_config in cameras.items():
        selector = CropSelector(
            camera_name,
            camera_config['serial_number'],
            camera_config['dim'],
            camera_config['exposure']
        )
        
        try:
            # Initialize and capture
            selector.initialize_camera()
            
            # Capture a few frames to stabilize
            for _ in range(10):
                selector.capture_image()
            
            # Interactive selection
            crop_coords = selector.select_crop_region()
            
            if crop_coords:
                crop_results[camera_name] = crop_coords
                print(f"✓ {camera_name} 裁剪区域已确认: img[{crop_coords[0]}:{crop_coords[1]}, {crop_coords[2]}:{crop_coords[3]}]")
            
        except Exception as e:
            print(f"错误: 处理相机 {camera_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Make sure to cleanup before moving to next camera
            selector.cleanup()
            # Add a small delay between cameras
            import time
            time.sleep(1)
    
    # Update config file
    if crop_results:
        print("\n" + "=" * 60)
        print("更新配置文件...")
        print("=" * 60)
        
        for camera_name, (y1, y2, x1, x2) in crop_results.items():
            print(f'{camera_name}: lambda img: img[{y1}:{y2}, {x1}:{x2}]')
        
        if update_config_file(config_path, original_content, crop_results):
            print(f"\n✓ 配置文件已更新: {config_path}")
        else:
            print("\n✗ 更新配置文件失败")
    else:
        print("\n未选择任何裁剪区域，配置文件未更新")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
