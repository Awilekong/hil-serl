#!/usr/bin/env python3
"""
Auto-configure safety limits by recording teleoperation workspace.

Usage:
    python auto_configure_safety_limits.py --exp_name ram_insertion

This script will:
1. Initialize the robot environment with NO safety limits
2. Wait for SPACE key press to start recording
3. Record robot TCP poses while you teleoperate (free movement)
4. Wait for SPACE key press to stop recording
5. Calculate the workspace bounding box
6. Compute ABS_POSE_LIMIT_LOW/HIGH relative to TARGET_POSE
7. Automatically update the config.py file and exit

Author: Auto-generated for hil-serl workspace
"""

import argparse
import numpy as np
import time
import sys
import os
from pathlib import Path
from pynput import keyboard
import re
import inspect

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'serl_robot_infra'))
sys.path.insert(0, str(project_root / 'serl_launcher'))

from experiments.mappings import CONFIG_MAPPING


class SafetyLimitRecorder:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.recording = False
        self.poses = []
        self.space_pressed = False
        
        # Load experiment config
        if exp_name not in CONFIG_MAPPING:
            raise ValueError(f"Unknown experiment: {exp_name}. Available: {list(CONFIG_MAPPING.keys())}")
        
        train_config_class = CONFIG_MAPPING[exp_name]
        
        # Get the module where the TrainConfig class is defined
        config_module = sys.modules[train_config_class.__module__]
        self.config_module = config_module
        
        # Get the file path of the module
        self.config_path = Path(inspect.getfile(config_module))
        
        print(f"[INFO] Loaded config from: {self.config_path}")
        print(f"[INFO] Target pose: {config_module.EnvConfig.TARGET_POSE}")
        
        # Temporarily DISABLE safety limits for free recording
        # Save original limits
        self.original_low = config_module.EnvConfig.ABS_POSE_LIMIT_LOW.copy()
        self.original_high = config_module.EnvConfig.ABS_POSE_LIMIT_HIGH.copy()
        
        # Set very permissive limits (effectively disabled)
        very_large = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        config_module.EnvConfig.ABS_POSE_LIMIT_LOW = config_module.EnvConfig.TARGET_POSE - very_large
        config_module.EnvConfig.ABS_POSE_LIMIT_HIGH = config_module.EnvConfig.TARGET_POSE + very_large
        
        print(f"\n[INFO] ⚠️  Safety limits TEMPORARILY DISABLED for free movement")
        print(f"[INFO] Original LOW:  TARGET_POSE - {self.original_low - config_module.EnvConfig.TARGET_POSE}")
        print(f"[INFO] Original HIGH: TARGET_POSE + {self.original_high - config_module.EnvConfig.TARGET_POSE}")
        print(f"[INFO] Current:       ±10m (effectively unlimited)")
        print(f"[INFO] BE CAREFUL during teleoperation!\n")
        
        # Initialize environment
        print("[INFO] Initializing environment...")
        train_config = train_config_class()
        self.env = train_config.get_environment(fake_env=False, save_video=False, classifier=False)
        
        print("[INFO] Environment initialized successfully!")
        print("\n" + "="*60)
        print("INSTRUCTIONS:")
        print("1. Press SPACE to START recording")
        print("2. Teleoperate the robot to explore the desired workspace")
        print("3. Press SPACE to STOP recording")
        print("4. The script will calculate and update safety limits")
        print("5. Script will automatically exit after update")
        print("="*60 + "\n")
        
    def on_press(self, key):
        """Keyboard callback for space key detection."""
        try:
            if key == keyboard.Key.space:
                if not self.space_pressed:
                    self.space_pressed = True
                    if not self.recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
        except Exception as e:
            print(f"[ERROR] Keyboard callback error: {e}")
    
    def on_release(self, key):
        """Reset space key flag on release."""
        try:
            if key == keyboard.Key.space:
                self.space_pressed = False
        except:
            pass
    
    def start_recording(self):
        """Start recording TCP poses."""
        self.recording = True
        self.poses = []
        print("\n" + "🔴 " * 20)
        print("RECORDING STARTED - Move the robot around!")
        print("🔴 " * 20 + "\n")
    
    def stop_recording(self):
        """Stop recording and process data."""
        self.recording = False
        print("\n" + "🟢 " * 20)
        print("RECORDING STOPPED")
        print("🟢 " * 20 + "\n")
        
        if len(self.poses) < 10:
            print(f"[WARNING] Only {len(self.poses)} poses recorded. Need at least 10.")
            print("[INFO] Press SPACE to start recording again.")
            return
        
        print(f"[INFO] Recorded {len(self.poses)} poses")
        self.process_and_update()
    
    def run(self):
        """Main recording loop."""
        # Reset environment to TARGET_POSE
        print("[INFO] Resetting robot to TARGET_POSE...")
        obs, _ = self.env.reset()
        print(f"[INFO] Robot reset complete. Current position should be at TARGET_POSE:")
        print(f"       {self.config_module.EnvConfig.TARGET_POSE}")
        
        # Debug: Check what's in obs to find TCP pose
        print("\n[DEBUG] Checking observation keys...")
        if isinstance(obs, dict):
            print(f"[DEBUG] Available keys in obs: {list(obs.keys())}")
            if 'state' in obs:
                print(f"[DEBUG] obs['state'] shape: {obs['state'].shape}")
                print(f"[DEBUG] obs['state'] first few values: {obs['state'][:10] if len(obs['state']) > 10 else obs['state']}")
        
        # Start keyboard listener
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        print("\n[INFO] Ready! Press SPACE to start recording...")
        
        try:
            while True:
                # Step environment with zero action to get current state
                action = np.zeros(self.env.action_space.shape)
                obs, reward, done, truncated, info = self.env.step(action)
                
                # Record pose if recording
                if self.recording:
                    # Get ABSOLUTE TCP pose from the base environment
                    tcp_pose = None
                    
                    # Method 1: Try to get from info dict
                    if 'tcp_pose' in info:
                        tcp_pose = info['tcp_pose']
                    
                    # Method 2: Try to access base environment directly
                    if tcp_pose is None:
                        try:
                            # Navigate through wrappers to find base FrankaEnv
                            current_env = self.env
                            while hasattr(current_env, 'env'):
                                current_env = current_env.env
                            
                            # Get absolute pose from robot state
                            if hasattr(current_env, '_get_obs'):
                                base_obs = current_env._get_obs()
                                if 'tcp_pose' in base_obs:
                                    tcp_pose = base_obs['tcp_pose']
                            elif hasattr(current_env, 'robot_state'):
                                tcp_pose = current_env.robot_state[:6]  # x,y,z,roll,pitch,yaw
                        except Exception as e:
                            print(f"\n[DEBUG] Error accessing base env: {e}")
                    
                    # Method 3: Fallback - but warn user this might be relative coordinates
                    if tcp_pose is None and 'state' in obs:
                        state = obs['state']
                        if hasattr(state, 'shape'):
                            state = np.squeeze(state)
                        if len(state) >= 6:
                            tcp_pose = state[:6]
                            if len(self.poses) == 0:
                                print("\n[WARNING] Using obs['state'] - these might be RELATIVE coordinates!")
                                print("[WARNING] If final limits look wrong, the script needs adjustment.")
                    
                    if tcp_pose is not None:
                        self.poses.append(tcp_pose.copy())
                        
                        # Visual feedback
                        if len(self.poses) % 10 == 0:
                            print(f"[RECORDING] {len(self.poses)} poses | Current: X={tcp_pose[0]:.3f}, Y={tcp_pose[1]:.3f}, Z={tcp_pose[2]:.3f}     ", end='\r')
                    else:
                        if len(self.poses) % 50 == 0:
                            print(f"\n[ERROR] Cannot find TCP pose! Check observation structure.")
                
                time.sleep(0.01)  # 100 Hz recording rate
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl+C)")
            sys.exit(0)
        finally:
            listener.stop()
            print("[INFO] Cleaning up...")
    
    def process_and_update(self):
        """Calculate workspace bounds and update config file."""
        poses = np.array(self.poses)
        target_pose = self.config_module.EnvConfig.TARGET_POSE
        
        print("\n" + "="*60)
        print("WORKSPACE ANALYSIS")
        print("="*60)
        
        # Calculate min/max for each dimension
        min_pose = poses.min(axis=0)
        max_pose = poses.max(axis=0)
        
        print(f"\nRecorded workspace bounds:")
        labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        for i, label in enumerate(labels):
            print(f"  {label:6s}: [{min_pose[i]:7.4f}, {max_pose[i]:7.4f}]  "
                  f"(range: {max_pose[i] - min_pose[i]:.4f})")
        
        print(f"\nTarget pose: {target_pose}")
        
        # Sanity check: verify poses are in reasonable range for absolute coordinates
        # Franka workspace is roughly X:[0.3-0.8], Y:[-0.4, 0.4], Z:[0.0, 0.6]
        if abs(min_pose[0]) > 2.0 or abs(max_pose[0]) > 2.0:
            print("\n" + "⚠️ " * 20)
            print("[ERROR] Recorded X coordinates seem unreasonable!")
            print(f"[ERROR] X range: [{min_pose[0]:.3f}, {max_pose[0]:.3f}]")
            print("[ERROR] This might be RELATIVE coordinates, not ABSOLUTE!")
            print("[ERROR] Cannot safely update config. Exiting...")
            print("⚠️ " * 20 + "\n")
            sys.exit(1)
        
        # No safety margin - use exact recorded bounds
        abs_low = min_pose
        abs_high = max_pose
        
        # Calculate relative to TARGET_POSE
        relative_low = target_pose - abs_low
        relative_high = abs_high - target_pose
        
        print(f"\nCalculated limits (absolute):")
        print(f"  ABS_POSE_LIMIT_LOW  = {abs_low}")
        print(f"  ABS_POSE_LIMIT_HIGH = {abs_high}")
        
        print(f"\nCalculated limits (relative to TARGET_POSE):")
        print(f"  ABS_POSE_LIMIT_LOW  = TARGET_POSE - {relative_low}")
        print(f"  ABS_POSE_LIMIT_HIGH = TARGET_POSE + {relative_high}")
        
        # Another sanity check for relative values
        if np.any(np.abs(relative_low) > 1.0) or np.any(np.abs(relative_high) > 1.0):
            print("\n" + "⚠️ " * 20)
            print("[WARNING] Relative limits seem very large!")
            for i, label in enumerate(labels):
                if abs(relative_low[i]) > 1.0 or abs(relative_high[i]) > 1.0:
                    print(f"[WARNING] {label}: -{relative_low[i]:.3f} / +{relative_high[i]:.3f}")
            print("\n[PROMPT] Do you want to continue with these limits? (y/n): ", end='', flush=True)
            
            # Wait for user confirmation
            response = input().strip().lower()
            if response != 'y':
                print("[INFO] Update cancelled by user. Exiting...")
                sys.exit(0)
        
        # Format arrays for config file
        low_array_str = self._format_array(relative_low)
        high_array_str = self._format_array(relative_high)
        
        print("\n" + "="*60)
        print("UPDATING CONFIG FILE")
        print("="*60)
        
        # Read current config file
        with open(self.config_path, 'r') as f:
            content = f.read()
        
        # Find and replace ABS_POSE_LIMIT_LOW (no backup)
        pattern_low = r'ABS_POSE_LIMIT_LOW\s*=\s*TARGET_POSE\s*-\s*np\.array\([^\)]+\)'
        replacement_low = 'ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array(' + low_array_str + ')'
        content_new, n_low = re.subn(pattern_low, replacement_low, content)
        
        if n_low == 0:
            print("[ERROR] Could not find ABS_POSE_LIMIT_LOW in config file")
            print("[INFO] Exiting without changes...")
            sys.exit(1)
        
        # Find and replace ABS_POSE_LIMIT_HIGH
        pattern_high = r'ABS_POSE_LIMIT_HIGH\s*=\s*TARGET_POSE\s*\+\s*np\.array\([^\)]+\)'
        replacement_high = 'ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array(' + high_array_str + ')'
        content_new, n_high = re.subn(pattern_high, replacement_high, content_new)
        
        if n_high == 0:
            print("[ERROR] Could not find ABS_POSE_LIMIT_HIGH in config file")
            print("[INFO] Exiting without changes...")
            sys.exit(1)
        
        # Write updated content (no backup, direct update)
        with open(self.config_path, 'w') as f:
            f.write(content_new)
        
        print(f"[SUCCESS] Updated {n_low} ABS_POSE_LIMIT_LOW line(s)")
        print(f"[SUCCESS] Updated {n_high} ABS_POSE_LIMIT_HIGH line(s)")
        print(f"[SUCCESS] Config file updated: {self.config_path}")
        
        print("\n" + "="*60)
        print("UPDATED CONFIG LINES:")
        print("="*60)
        print(f"ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array({low_array_str})")
        print(f"ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array({high_array_str})")
        print("="*60 + "\n")
        
        print("[INFO] ✅ Done! Config file updated with new safety limits.")
        print("[INFO] 🛑 Exiting safely...")
        
        # Exit after successful update
        sys.exit(0)
    
    def _format_array(self, arr):
        """Format numpy array for config file with proper precision."""
        # Round to 4 decimal places for cleaner output
        arr_rounded = np.round(arr, 4)
        formatted = [f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}" for x in arr_rounded]
        return "[" + ", ".join(formatted) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Auto-configure safety limits by recording workspace"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name (e.g., ram_insertion, usb_pickup_insertion)",
    )
    
    args = parser.parse_args()
    
    try:
        recorder = SafetyLimitRecorder(args.exp_name)
        recorder.run()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
