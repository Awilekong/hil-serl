import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard
import cv2

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")


# Recording state management
recording_mode = None  # None, 'success', or 'failure'
toggle_key_pressed = False

def on_press(key):
    global recording_mode, toggle_key_pressed
    try:
        # Space key toggles success recording
        if str(key) == 'Key.space':
            if recording_mode != 'failure':  # Only toggle if not recording failure
                if not toggle_key_pressed:
                    if recording_mode == 'success':
                        recording_mode = None
                        print("\n[INFO] Success recording stopped")
                    else:
                        recording_mode = 'success'
                        print("\n[INFO] Success recording started")
                    toggle_key_pressed = True
        # Enter key toggles failure recording
        elif str(key) == 'Key.enter':
            if recording_mode != 'success':  # Only toggle if not recording success
                if not toggle_key_pressed:
                    if recording_mode == 'failure':
                        recording_mode = None
                        print("\n[INFO] Failure recording stopped")
                    else:
                        recording_mode = 'failure'
                        print("\n[INFO] Failure recording started")
                    toggle_key_pressed = True
    except AttributeError:
        pass

def on_release(key):
    global toggle_key_pressed
    toggle_key_pressed = False

def main(_):
    global recording_mode
    
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()
    
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # Get environment
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    
    # HACK: Enable gripper control for data collection only
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
        print("[INFO] Gripper control DISABLED for data collection")
        print("[INFO] Gripper will remain in fixed position")
    else:
        print("[WARN] SpacemouseIntervention wrapper not found, gripper control may not work")
    
    # Temporarily increase ACTION_SCALE for better teleoperation
    # This only affects this script, not training
    if hasattr(env.unwrapped, 'action_scale'):
        original_action_scale = env.unwrapped.action_scale
        env.unwrapped.action_scale = (0.03, 0.5, 1)  # 5x position, 1.33x rotation
        print(f"[INFO] ACTION_SCALE temporarily increased for teleoperation:")
        print(f"      Original: {original_action_scale}")
        print(f"      Current:  {env.unwrapped.action_scale}")
    
    # Completely disable XYZ safety box clipping for data collection
    if hasattr(env.unwrapped, 'xyz_bounding_box'):
        original_xyz_box = (env.unwrapped.xyz_bounding_box.low.copy(), 
                           env.unwrapped.xyz_bounding_box.high.copy())
        # Set to very large range to effectively disable XYZ limits
        env.unwrapped.xyz_bounding_box.low = np.array([-10.0, -10.0, -10.0])
        env.unwrapped.xyz_bounding_box.high = np.array([10.0, 10.0, 10.0])
        print("[INFO] XYZ safety box limits DISABLED for data collection")
        print(f"      Original X: [{original_xyz_box[0][0]:.3f}, {original_xyz_box[1][0]:.3f}]")
        print(f"      Original Y: [{original_xyz_box[0][1]:.3f}, {original_xyz_box[1][1]:.3f}]")
        print(f"      Original Z: [{original_xyz_box[0][2]:.3f}, {original_xyz_box[1][2]:.3f}]")
        print(f"      Current: [-10.0, 10.0] for all axes")
    
    # Also update ABS_POSE_LIMIT for consistency (but xyz_bounding_box is what matters)
    if hasattr(env.unwrapped, 'config'):
        env_config = env.unwrapped.config
        original_low = env_config.ABS_POSE_LIMIT_LOW.copy()
        original_high = env_config.ABS_POSE_LIMIT_HIGH.copy()
        # Set very permissive limits (mainly for rotation, XYZ already disabled above)
        env_config.ABS_POSE_LIMIT_LOW[:3] = np.array([-10.0, -10.0, -10.0])
        env_config.ABS_POSE_LIMIT_HIGH[:3] = np.array([10.0, 10.0, 10.0])
        env_config.ABS_POSE_LIMIT_LOW[3:] = np.array([-np.pi, -np.pi, -np.pi])
        env_config.ABS_POSE_LIMIT_HIGH[3:] = np.array([np.pi, np.pi, np.pi])

    # Reset only once at the beginning
    obs, _ = env.reset()
    print("[INFO] Environment initialized. Ready to collect data.")
    print("[CONTROLS] Press SPACE to start/stop success recording")
    print("[CONTROLS] Press ENTER to start/stop failure recording")
    
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed, desc="Success samples")
    
    step_count = 0
    try:
        while len(successes) < success_needed:
            step_count += 1
            actions = np.zeros(env.action_space.sample().shape) 
            next_obs, rew, done, truncated, info = env.step(actions)
            if "intervene_action" in info:
                actions = info["intervene_action"]

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0,  # Always 1.0 since we don't reset
                    dones=False,  # Never done since we don't reset
                )
            )
            obs = next_obs
            
            # Record based on current mode
            if recording_mode == 'success':
                successes.append(transition)
                pbar.update(1)
                # Display recording status on image window
                if hasattr(env.unwrapped, 'img_queue') and not env.unwrapped.img_queue.empty():
                    # Add text overlay: "Recording SUCCESS data..."
                    pass  # Text overlay handled by window title or separate display
            elif recording_mode == 'failure':
                failures.append(transition)
            
            # Update window title or display status
            status = "IDLE"
            if recording_mode == 'success':
                status = f"Recording SUCCESS data... ({len(successes)}/{success_needed})"
            elif recording_mode == 'failure':
                status = f"Recording FAILURE data... ({len(failures)} samples)"
            
            if step_count % 10 == 0:  # Print status every 10 steps
                print(f"\r[STATUS] {status} | Total steps: {step_count}", end='')
            
            # Note: We removed the auto-reset logic (done or truncated)
            # The environment will continue running indefinitely
            
    except KeyboardInterrupt:
        print("\n[INFO] Data collection interrupted by user")
    finally:
        # Close SpaceMouse to prevent BrokenPipeError
        # Find SpacemouseIntervention wrapper and close its expert
        print("\n[INFO] Cleaning up...")
        current_env = env
        spacemouse_closed = False
        while hasattr(current_env, 'env'):
            if hasattr(current_env, 'expert') and not spacemouse_closed:
                print("[INFO] Closing SpaceMouse...")
                try:
                    # Terminate the process more forcefully
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
        
        # Restore original ACTION_SCALE
        if hasattr(env.unwrapped, 'action_scale'):
            env.unwrapped.action_scale = original_action_scale
            print("[INFO] ACTION_SCALE restored")
        
        # Restore original XYZ safety box
        if hasattr(env.unwrapped, 'xyz_bounding_box'):
            env.unwrapped.xyz_bounding_box.low = original_xyz_box[0]
            env.unwrapped.xyz_bounding_box.high = original_xyz_box[1]
            print("[INFO] XYZ safety box limits restored")
        
        # Restore original pose limits
        if hasattr(env.unwrapped, 'config'):
            env.unwrapped.config.ABS_POSE_LIMIT_LOW = original_low
            env.unwrapped.config.ABS_POSE_LIMIT_HIGH = original_high
            print("[INFO] Pose limits restored")
        
        # Stop keyboard listener
        try:
            listener.stop()
        except:
            pass
        print("[INFO] Cleanup complete")

    # Save collected data
    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if len(successes) > 0:
        file_name = f"./classifier_data/{FLAGS.exp_name}_{len(successes)}_success_images_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(successes, f)
            print(f"\n[SAVE] Saved {len(successes)} successful transitions to {file_name}")

    if len(failures) > 0:
        file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(failures, f)
            print(f"[SAVE] Saved {len(failures)} failure transitions to {file_name}")
    
    print("\n[INFO] All done! Exiting...")
    
    # Force exit to ensure all processes terminate
    import sys
    sys.exit(0)
        
if __name__ == "__main__":
    app.run(main)
