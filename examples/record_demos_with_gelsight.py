import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
from pynput import keyboard
import gymnasium as gym

from experiments.mappings import CONFIG_MAPPING


import cv2
import sys  # Ensure sys is imported
import os

# Add the project root to sys.path to fix import issues
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the GelSight module path to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
gelmini_path = os.path.abspath(os.path.join(script_dir, "../VTLA_Data_Collect-main"))
if gelmini_path not in sys.path:
    sys.path.insert(0, gelmini_path)

from gelmini import gsdevice  # Import GelSight SDK

# Initialize GelSight after environment setup
print("[INFO] Initializing GelSight sensors...")
gelsight_1 = gsdevice.Camera(0)
gelsight_1.connect()
gelsight_2 = gsdevice.Camera(0)
gelsight_2.connect()
print("[INFO] GelSight sensors initialized.")

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

# State management
recording = False
reset_requested = False
space_pressed = False
enter_pressed = False
waiting_for_save_decision = False
save_decision = None  # Will be 'y' or 'n'

def on_press(key):
    global recording, reset_requested, space_pressed, enter_pressed
    global waiting_for_save_decision, save_decision
    
    try:
        # If waiting for save decision, only accept y/n
        if waiting_for_save_decision:
            if hasattr(key, 'char'):
                if key.char == 'y':
                    save_decision = 'y'
                    print("Y")
                elif key.char == 'n':
                    save_decision = 'n'
                    print("N")
            return  # Don't process other keys while waiting for decision
        
        # Space key toggles recording
        if str(key) == 'Key.space':
            if not space_pressed:
                recording = not recording
                if recording:
                    print("\n[INFO] Recording STARTED - collecting trajectory...")
                else:
                    print("\n[INFO] Recording STOPPED - trajectory complete")
                space_pressed = True
        # Enter key triggers reset
        elif str(key) == 'Key.enter':
            if not enter_pressed:
                reset_requested = True
                print("\n[INFO] Reset requested")
                enter_pressed = True
    except AttributeError:
        pass

def on_release(key):
    global space_pressed, enter_pressed
    if str(key) == 'Key.space':
        space_pressed = False
    elif str(key) == 'Key.enter':
        enter_pressed = False

def main(_):
    global recording, reset_requested
    global waiting_for_save_decision, save_decision
    
    # Video recording variables
    video_writer = None
    video_filename = None
    
    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
    # HACK: Enable gripper control for demo collection only
    # Find SpacemouseIntervention wrapper and enable gripper
    current_env = env
    spacemouse_wrapper = None
    while hasattr(current_env, 'env'):
        if current_env.__class__.__name__ == 'SpacemouseIntervention':
            spacemouse_wrapper = current_env
            break
        current_env = current_env.env
    
    if spacemouse_wrapper is not None:
        spacemouse_wrapper.gripper_enabled = False
        print("[INFO] Gripper control DISABLED for demo collection")
        print("[INFO] Gripper will remain in fixed position")
    else:
        print("[WARN] SpacemouseIntervention wrapper not found, gripper control may not work")
    
    # Temporarily increase ACTION_SCALE for better teleoperation
    if hasattr(env.unwrapped, 'action_scale'):
        original_action_scale = env.unwrapped.action_scale
        env.unwrapped.action_scale = (0.03, 0.5, 1)
        print(f"[INFO] ACTION_SCALE temporarily increased for teleoperation:")
        print(f"      Original: {original_action_scale}")
        print(f"      Current:  {env.unwrapped.action_scale}")
    
    # Set high resolution for RealSense images in video recording
    env.unwrapped.observation_space["images"]["side_policy"] = gym.spaces.Box(0, 255, shape=(1080, 1920, 3), dtype=np.uint8)
    print("[INFO] Set RealSense observation space to high resolution")
    
    # Reset only once at the beginning
    obs, info = env.reset()


    print("\n" + "="*60)
    print("[INFO] Environment initialized. Ready to collect demos.")
    print("="*60)
    print("[CONTROLS] Press SPACE to start/stop recording a trajectory")
    print("[CONTROLS] Press ENTER to reset the environment")
    print("[CONTROLS] Ctrl+C to finish and save collected demos")
    print("           ⚠️  Incomplete trajectories will be discarded!")
    print("="*60 + "\n")
    
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed, desc="Demos collected")
    trajectory = []
    
    step_count = 0
    trajectory_reward_sum = 0
    
    try:
        while success_count < success_needed:
            step_count += 1
            
            # Execute step
            actions = np.zeros(env.action_space.sample().shape) 
            next_obs, rew, done, truncated, info = env.step(actions)
            gelsight_img_1 = gelsight_1.get_image()
            gelsight_img_2 = gelsight_2.get_image()
            combined_gelsight = np.hstack((gelsight_img_1, gelsight_img_2))
            
            # Convert GelSight images to RGB if they are BGR
            if combined_gelsight.ndim == 3 and combined_gelsight.shape[2] == 3:
                combined_gelsight = cv2.cvtColor(combined_gelsight, cv2.COLOR_BGR2RGB)
            
            # Extract RealSense image from obs
            realsense_img = obs["side_policy"]
            if realsense_img.ndim == 4:
                realsense_img = realsense_img.squeeze(0)  # Remove batch/time dimension
            
            # Resize combined_gelsight to match RealSense height, limiting width to 1920
            gelsight_height = combined_gelsight.shape[0]
            realsense_height = realsense_img.shape[0]
            realsense_width = realsense_img.shape[1]
            scale_factor = realsense_height / gelsight_height
            original_new_width = int(combined_gelsight.shape[1] * scale_factor)
            new_width = min(original_new_width, realsense_width)
            combined_gelsight_resized = cv2.resize(combined_gelsight, (new_width, realsense_height))
            
            # Pad GelSight to match RealSense width with black borders
            gelsight_padded = np.zeros((realsense_height, realsense_width, 3), dtype=np.uint8)
            gelsight_padded[:, :new_width] = combined_gelsight_resized
            
            # Concatenate RealSense and padded GelSight images
            combined_img = np.hstack((realsense_img, gelsight_padded))
            
            # Video recording logic
            if recording:
                if video_writer is None:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    video_filename = f"./demo_data/{FLAGS.exp_name}_combined_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    height, width = combined_img.shape[:2]
                    video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height))
                    print(f"[VIDEO] Started recording combined video: {video_filename}")
                # Ensure image is in correct format for OpenCV
                frame = combined_img.astype(np.uint8)
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
            
            if "intervene_action" in info:
                actions = info["intervene_action"]
            
            # Print current reward in real-time
            print(f"\r[STEP {step_count}] Reward: {rew:.3f} | Recording: {'YES' if recording else 'NO '} | Trajectory length: {len(trajectory):3d}", end='', flush=True)
            
            # If recording, add to trajectory
            if recording:
                transition = copy.deepcopy(
                    dict(
                        observations=obs,
                        actions=actions,
                        next_observations=next_obs,
                        rewards=rew,
                        masks=1.0,
                        dones=False,
                        infos=info,
                    )
                )
                trajectory.append(transition)
                trajectory_reward_sum += rew
            
            # If recording just stopped (trajectory complete), ask user to save or discard
            if not recording and len(trajectory) > 0:
                # Save video immediately when recording stops
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print(f"[VIDEO] Saved combined video: {video_filename}")
                
                print(f"\n[TRAJECTORY] Collected {len(trajectory)} transitions with total reward: {trajectory_reward_sum:.2f}")
                print("[PROMPT] Save this trajectory? Press Y (save) or N (discard): ", end='', flush=True)
                
                # Wait for user decision using keyboard listener
                waiting_for_save_decision = True
                save_decision = None
                
                # Wait for y or n key press
                while save_decision is None:
                    time.sleep(0.05)  # Small delay to reduce CPU usage
                
                waiting_for_save_decision = False
                
                if save_decision == 'y':
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                    print(f"\n[SAVE] Trajectory saved! Total demos: {success_count}/{success_needed}")
                else:
                    print("\n[DISCARD] Trajectory discarded.")
                
                # Clear trajectory
                trajectory = []
                trajectory_reward_sum = 0
            
            # Handle reset request
            if reset_requested:
                print("\n[RESET] Resetting environment...")
                obs, info = env.reset()
                trajectory = []
                trajectory_reward_sum = 0
                recording = False
                reset_requested = False
                step_count = 0
                print("[RESET] Environment reset complete. Ready to record.")
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    if os.path.exists(video_filename):
                        os.remove(video_filename)
                    video_filename = None
            else:
                obs = next_obs
                
    except KeyboardInterrupt:
        print("\n\n[INFO] Data collection interrupted by user (Ctrl+C)")
        
        # Handle incomplete trajectory
        if recording and len(trajectory) > 0:
            print(f"[WARNING] Current trajectory in progress ({len(trajectory)} steps) will be DISCARDED")
            print("[INFO] Only completed and saved trajectories will be kept")
            trajectory = []  # Discard incomplete trajectory
        elif len(trajectory) > 0 and not recording:
            # Trajectory was just completed but user hasn't decided yet
            print(f"\n[TRAJECTORY] Collected {len(trajectory)} transitions with total reward: {trajectory_reward_sum:.2f}")
            print("[PROMPT] Save this trajectory before exit? Press Y (save) or N (discard): ", end='', flush=True)
            
            # Wait for user decision
            waiting_for_save_decision = True
            save_decision = None
            
            # Give user 5 seconds to decide, otherwise discard
            timeout_start = time.time()
            while save_decision is None and (time.time() - timeout_start) < 5.0:
                time.sleep(0.05)
            
            waiting_for_save_decision = False
            
            if save_decision == 'y':
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                print(f"\n[SAVE] Final trajectory saved! Total demos: {success_count}/{success_needed}")
            else:
                print("\n[DISCARD] Final trajectory discarded (timeout or user choice).")
            
            trajectory = []
    finally:
        # Cleanup
        print("\n[INFO] Cleaning up...")
        current_env = env
        spacemouse_closed = False
        while hasattr(current_env, 'env'):
            if hasattr(current_env, 'expert') and not spacemouse_closed:
                print("[INFO] Closing SpaceMouse...")
                try:
                    if hasattr(current_env.expert, 'process'):
                        current_env.expert.process.terminate()
                        current_env.expert.process.join(timeout=1)
                        if current_env.expert.process.is_alive():
                            current_env.expert.process.kill()
                    spacemouse_closed = True
                except Exception as e:
                    print(f"[WARN] Error closing SpaceMouse: {e}")
                break
            current_env = current_env.env
        
        # Restore settings
        if hasattr(env.unwrapped, 'action_scale'):
            env.unwrapped.action_scale = original_action_scale
            print("[INFO] ACTION_SCALE restored")
        
        print("[INFO] Safety limits kept at original values (not modified)")
        
        if video_writer is not None:
            video_writer.release()
            if os.path.exists(video_filename):
                os.remove(video_filename)
            print("[VIDEO] Cleanup: discarded incomplete video")
        
        try:
            listener.stop()
        except:
            pass
        print("[INFO] Cleanup complete")
    
    # Save collected data
    if len(transitions) > 0:
        if not os.path.exists("./demo_data"):
            os.makedirs("./demo_data")
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"./demo_data/{FLAGS.exp_name}_{success_count}_demos_{uuid}.pkl"
        
        print("\n" + "="*60)
        print(f"[SAVING] Writing {len(transitions)} transitions to disk...")
        print(f"[FILE] {file_name}")
        
        try:
            with open(file_name, "wb") as f:
                pkl.dump(transitions, f)
            print(f"[SUCCESS] ✓ Data saved successfully!")
            print(f"[SUMMARY] {success_count} trajectories, {len(transitions)} total transitions")
            print("="*60)
        except Exception as e:
            print(f"[ERROR] ✗ Failed to save data: {e}")
            print("="*60)
    else:
        print("\n" + "="*60)
        print("[INFO] No demos collected, nothing to save.")
        print("="*60)
    
    print("\n[INFO] All done! Exiting...")
    import sys
    sys.exit(0)

if __name__ == "__main__":
    app.run(main)