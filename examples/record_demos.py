import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

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
    
    # Disable XYZ safety box clipping
    if hasattr(env.unwrapped, 'xyz_bounding_box'):
        original_xyz_box = (env.unwrapped.xyz_bounding_box.low.copy(), 
                           env.unwrapped.xyz_bounding_box.high.copy())
        env.unwrapped.xyz_bounding_box.low = np.array([-10.0, -10.0, -10.0])
        env.unwrapped.xyz_bounding_box.high = np.array([10.0, 10.0, 10.0])
        print("[INFO] XYZ safety box limits DISABLED for data collection")
    
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
                
                # Check if reward indicates success (reward == 1)
                if rew >= 1.0:
                    print(f"\n[SUCCESS] Reward reached {rew:.1f}! Trajectory complete with {len(trajectory)} steps.")
                    recording = False
            
            # If recording just stopped (trajectory complete), ask user to save or discard
            if not recording and len(trajectory) > 0:
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
        
        if hasattr(env.unwrapped, 'xyz_bounding_box'):
            env.unwrapped.xyz_bounding_box.low = original_xyz_box[0]
            env.unwrapped.xyz_bounding_box.high = original_xyz_box[1]
            print("[INFO] XYZ safety box limits restored")
        
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