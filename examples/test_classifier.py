"""
Real-time Reward Classifier Testing Script

This script allows you to test the trained reward classifier in real-time.
It displays:
- Current reward/probability from the classifier
- Success/Failure prediction
- Live camera feed
- Manual labels for accuracy checking

Controls:
- SPACE: Mark current state as SUCCESS (ground truth)
- ENTER: Mark current state as FAILURE (ground truth)
- Q: Quit and show accuracy statistics
"""

import os
from tqdm import tqdm
import numpy as np
import copy
import datetime
from absl import app, flags
import time
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")

# State management
manual_label = None  # Will be 'success' or 'failure'
space_pressed = False
enter_pressed = False
quit_requested = False

# Statistics tracking
predictions = []
ground_truths = []
prediction_history = []

def on_press(key):
    global manual_label, space_pressed, enter_pressed, quit_requested
    
    try:
        # Space key: mark as SUCCESS
        if str(key) == 'Key.space':
            if not space_pressed:
                manual_label = 'success'
                print("\n[MANUAL LABEL] ✓ SUCCESS")
                space_pressed = True
        # Enter key: mark as FAILURE
        elif str(key) == 'Key.enter':
            if not enter_pressed:
                manual_label = 'failure'
                print("\n[MANUAL LABEL] ✗ FAILURE")
                enter_pressed = True
        # Q key: quit
        elif hasattr(key, 'char') and key.char == 'q':
            quit_requested = True
            print("\n[INFO] Quit requested...")
    except AttributeError:
        pass

def on_release(key):
    global space_pressed, enter_pressed
    if str(key) == 'Key.space':
        space_pressed = False
    elif str(key) == 'Key.enter':
        enter_pressed = False

def calculate_accuracy(predictions, ground_truths):
    """Calculate accuracy from predictions and ground truths"""
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return correct / len(predictions)

def main(_):
    global manual_label, quit_requested, predictions, ground_truths, prediction_history
    
    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # Must use classifier=True to load the reward classifier
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
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
        print("[INFO] XYZ safety box limits DISABLED")
    
    # Reset environment
    obs, info = env.reset()
    
    print("\n" + "="*70)
    print("       REWARD CLASSIFIER REAL-TIME TESTING")
    print("="*70)
    print("[CONTROLS] Press SPACE to label current state as SUCCESS (✓)")
    print("[CONTROLS] Press ENTER to label current state as FAILURE (✗)")
    print("[CONTROLS] Press Q to quit and show accuracy statistics")
    print("="*70)
    print("\nTeleoperate the robot and press SPACE/ENTER to label states.")
    print("The classifier will predict success/failure in real-time.\n")
    
    step_count = 0
    correct_count = 0
    total_labeled = 0
    
    try:
        while not quit_requested:
            step_count += 1
            
            # Execute step
            actions = np.zeros(env.action_space.sample().shape) 
            next_obs, rew, done, truncated, info = env.step(actions)
            if "intervene_action" in info:
                actions = info["intervene_action"]
            
            # Get classifier prediction
            predicted_success = rew >= 1.0  # Classifier returns 1 for success, 0 for failure
            predicted_label = "SUCCESS" if predicted_success else "FAILURE"
            
            # Color codes for terminal output
            if predicted_success:
                status_color = "✓"
                status_emoji = "🟢"
            else:
                status_color = "✗"
                status_emoji = "🔴"
            
            # Calculate running accuracy if we have labels
            accuracy_str = ""
            if total_labeled > 0:
                accuracy = (correct_count / total_labeled) * 100
                accuracy_str = f"| Accuracy: {accuracy:.1f}% ({correct_count}/{total_labeled})"
            
            # Print real-time status
            print(f"\r[STEP {step_count:4d}] {status_emoji} Classifier: {predicted_label:7s} (reward={rew:.3f}) {accuracy_str}", 
                  end='', flush=True)
            
            # Store prediction for history
            prediction_history.append({
                'step': step_count,
                'reward': float(rew),
                'predicted': predicted_success,
                'timestamp': time.time()
            })
            
            # If user provided a manual label, record it and compare with prediction
            if manual_label is not None:
                ground_truth_success = (manual_label == 'success')
                
                # Store for statistics
                predictions.append(predicted_success)
                ground_truths.append(ground_truth_success)
                total_labeled += 1
                
                # Check if prediction matches ground truth
                is_correct = (predicted_success == ground_truth_success)
                if is_correct:
                    correct_count += 1
                    print(f"\n[RESULT] ✓ CORRECT! Classifier predicted {predicted_label}, you labeled {manual_label.upper()}")
                else:
                    print(f"\n[RESULT] ✗ WRONG! Classifier predicted {predicted_label}, but you labeled {manual_label.upper()}")
                
                # Clear manual label
                manual_label = None
            
            obs = next_obs
            
            # Small delay to prevent too fast updates
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n\n[INFO] Testing interrupted by user")
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
    
    # Print final statistics
    print("\n" + "="*70)
    print("                    TESTING STATISTICS")
    print("="*70)
    print(f"Total steps executed: {step_count}")
    print(f"Total states labeled: {total_labeled}")
    
    if total_labeled > 0:
        accuracy = (correct_count / total_labeled) * 100
        print(f"\nClassifier Accuracy: {accuracy:.2f}% ({correct_count}/{total_labeled})")
        
        # Breakdown by class
        success_predictions = sum(1 for p in predictions if p)
        failure_predictions = len(predictions) - success_predictions
        success_ground_truth = sum(1 for g in ground_truths if g)
        failure_ground_truth = len(ground_truths) - success_ground_truth
        
        print(f"\nPredicted SUCCESS: {success_predictions}/{total_labeled}")
        print(f"Predicted FAILURE: {failure_predictions}/{total_labeled}")
        print(f"\nActual SUCCESS: {success_ground_truth}/{total_labeled}")
        print(f"Actual FAILURE: {failure_ground_truth}/{total_labeled}")
        
        # Calculate confusion matrix
        true_positive = sum(1 for p, g in zip(predictions, ground_truths) if p and g)
        true_negative = sum(1 for p, g in zip(predictions, ground_truths) if not p and not g)
        false_positive = sum(1 for p, g in zip(predictions, ground_truths) if p and not g)
        false_negative = sum(1 for p, g in zip(predictions, ground_truths) if not p and g)
        
        print("\nConfusion Matrix:")
        print("                  Predicted")
        print("                SUCCESS  FAILURE")
        print(f"Actual SUCCESS     {true_positive:3d}      {false_negative:3d}")
        print(f"       FAILURE     {false_positive:3d}      {true_negative:3d}")
        
        # Additional metrics
        if (true_positive + false_positive) > 0:
            precision = true_positive / (true_positive + false_positive)
            print(f"\nPrecision: {precision:.2%}")
        
        if (true_positive + false_negative) > 0:
            recall = true_positive / (true_positive + false_negative)
            print(f"Recall: {recall:.2%}")
    else:
        print("\nNo states were labeled. Run again and press SPACE/ENTER to label states.")
    
    print("\n" + "="*70)
    print("[INFO] All done! Exiting...")
    import sys
    sys.exit(0)

if __name__ == "__main__":
    app.run(main)
