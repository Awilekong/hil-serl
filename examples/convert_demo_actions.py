#!/usr/bin/env python3
"""
Data conversion script: Convert 7D demo actions to 6D by removing gripper dimension.

Usage:
    python convert_demo_actions.py --input <input.pkl> --output <output.pkl>
    
Example:
    python convert_demo_actions.py \
        --input demo_data/ram_insertion_49_demos_2025-11-25_17-58-16.pkl \
        --output demo_data/ram_insertion_49_demos_6d.pkl
"""
import argparse
import pickle as pkl
import numpy as np
import os
from pathlib import Path


def convert_demo_actions(input_path, output_path, target_action_dim=6):
    """
    Convert demo data from 7D actions to 6D by removing the gripper dimension.
    
    Args:
        input_path: Path to input pickle file with 7D actions
        output_path: Path to output pickle file with 6D actions
        target_action_dim: Target action dimension (default: 6)
    """
    print(f"\n{'='*70}")
    print(f"Demo Action Converter: 7D → {target_action_dim}D")
    print(f"{'='*70}\n")
    
    # Load input data
    print(f"[1/4] Loading input data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, "rb") as f:
        transitions = pkl.load(f)
    
    print(f"      ✓ Loaded {len(transitions)} transitions")
    
    # Check first transition
    first_transition = transitions[0]
    
    # Check which key is used: 'action' or 'actions'
    if 'actions' in first_transition:
        action_key = 'actions'
    elif 'action' in first_transition:
        action_key = 'action'
    else:
        raise ValueError("Transition data does not contain 'action' or 'actions' key")
    
    original_action_dim = first_transition[action_key].shape[0]
    print(f"      ✓ Original action dimension: {original_action_dim}")
    
    if original_action_dim <= target_action_dim:
        print(f"\n⚠️  Warning: Original action dimension ({original_action_dim}) <= target ({target_action_dim})")
        print(f"    No conversion needed. Exiting.")
        return
    
    # Convert actions
    print(f"\n[2/4] Converting actions: {original_action_dim}D → {target_action_dim}D")
    print(f"      Removing dimension(s): [{target_action_dim}:{original_action_dim}]")
    
    converted_transitions = []
    removed_values_stats = []
    
    for i, transition in enumerate(transitions):
        converted_transition = transition.copy()
        
        # Convert action (use the key we detected: 'action' or 'actions')
        if action_key in transition:
            original_action = transition[action_key]
            converted_action = original_action[:target_action_dim]
            converted_transition[action_key] = converted_action
            
            # Track removed values for statistics
            removed_values = original_action[target_action_dim:]
            removed_values_stats.append(removed_values)
        
        # Convert 'next_action' if it exists
        if 'next_action' in transition:
            original_next_action = transition['next_action']
            converted_next_action = original_next_action[:target_action_dim]
            converted_transition['next_action'] = converted_next_action
        
        converted_transitions.append(converted_transition)
        
        # Progress indicator
        if (i + 1) % 500 == 0 or i == len(transitions) - 1:
            print(f"      Progress: {i+1}/{len(transitions)} transitions", end='\r')
    
    print(f"\n      ✓ Converted {len(converted_transitions)} transitions")
    
    # Statistics
    print(f"\n[3/4] Statistics of removed values (gripper actions):")
    removed_array = np.array(removed_values_stats)
    
    if removed_array.shape[1] == 1:
        # Single gripper dimension
        print(f"      Gripper dimension [index {target_action_dim}]:")
        print(f"        Min:    {removed_array.min():.4f}")
        print(f"        Max:    {removed_array.max():.4f}")
        print(f"        Mean:   {removed_array.mean():.4f}")
        print(f"        Std:    {removed_array.std():.4f}")
    else:
        # Multiple removed dimensions
        for dim_idx in range(removed_array.shape[1]):
            print(f"      Dimension [index {target_action_dim + dim_idx}]:")
            print(f"        Min:    {removed_array[:, dim_idx].min():.4f}")
            print(f"        Max:    {removed_array[:, dim_idx].max():.4f}")
            print(f"        Mean:   {removed_array[:, dim_idx].mean():.4f}")
            print(f"        Std:    {removed_array[:, dim_idx].std():.4f}")
    
    # Save output
    print(f"\n[4/4] Saving converted data to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"      ✓ Created output directory: {output_dir}")
    
    with open(output_path, "wb") as f:
        pkl.dump(converted_transitions, f)
    
    # Verify saved data
    with open(output_path, "rb") as f:
        verify_data = pkl.load(f)
    
    verify_action_dim = verify_data[0][action_key].shape[0]
    print(f"      ✓ Saved {len(verify_data)} transitions")
    print(f"      ✓ Verified action dimension: {verify_action_dim}")
    
    # File size comparison
    input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    size_reduction = ((input_size - output_size) / input_size) * 100
    
    print(f"\n{'='*70}")
    print(f"Conversion Summary:")
    print(f"  Input file:         {input_path}")
    print(f"  Output file:        {output_path}")
    print(f"  Transitions:        {len(transitions)}")
    print(f"  Action conversion:  {original_action_dim}D → {verify_action_dim}D")
    print(f"  Input file size:    {input_size:.2f} MB")
    print(f"  Output file size:   {output_size:.2f} MB")
    print(f"  Size reduction:     {size_reduction:.1f}%")
    print(f"{'='*70}\n")
    print(f"✓ Conversion completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert demo actions from 7D to 6D by removing gripper dimension",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python convert_demo_actions.py \\
      --input demo_data/ram_insertion_49_demos_2025-11-25_17-58-16.pkl \\
      --output demo_data/ram_insertion_49_demos_6d.pkl
  
  # Specify custom output dimension
  python convert_demo_actions.py \\
      --input my_demos.pkl \\
      --output my_demos_6d.pkl \\
      --target-dim 6
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input pickle file with 7D actions"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output pickle file for 6D actions"
    )
    
    parser.add_argument(
        "--target-dim", "-d",
        type=int,
        default=6,
        help="Target action dimension (default: 6)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_demo_actions(args.input, args.output, args.target_dim)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        raise


if __name__ == "__main__":
    main()
