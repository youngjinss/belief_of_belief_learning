#!/usr/bin/env python3
"""
Script to find trajectory files with trajectory length over 50
in AchieverBlocker 9x9 environments.
"""

import os
import re
import glob
from pathlib import Path


def extract_trajectory_length(file_path):
    """
    Extract trajectory length from a trajectory file.

    Args:
        file_path: Path to the trajectory file

    Returns:
        int: Trajectory length, or None if not found
    """
    try:
        with open(file_path, "r") as f:
            for line in f:
                # Look for "Trajectory length: XXX" pattern
                match = re.search(r"Trajectory length:\s*(\d+)", line)
                if match:
                    return int(match.group(1))
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def find_long_trajectories(base_dir, threshold=50):
    """
    Find all trajectory files with length over the threshold.

    Args:
        base_dir: Base directory containing the data
        threshold: Minimum trajectory length to report

    Returns:
        dict: Dictionary mapping directory to list of (filename, length) tuples
    """
    # Dynamically find all subdirectories under "data/MiniGrid-AchieverBlocker-9x9-v1/"
    data_root = os.path.join(base_dir, "data", "MiniGrid-AchieverBlocker-9x9-v1")
    if not os.path.exists(data_root):
        print(f"Data root directory does not exist: {data_root}")
        search_dirs = []
    else:
        # Only extract subdirectories (exclude files)
        search_dirs = [
            os.path.relpath(
                os.path.join("data", "MiniGrid-AchieverBlocker-9x9-v1", d), base_dir
            )
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ]

    results = {}

    for search_dir in search_dirs:
        full_dir_path = os.path.join(base_dir, search_dir)

        if not os.path.exists(full_dir_path):
            print(f"Directory not found: {full_dir_path}")
            continue

        print(f"Searching in: {full_dir_path}")

        # Find all .txt files in the directory
        txt_files = glob.glob(os.path.join(full_dir_path, "*.txt"))

        long_trajectories = []

        for txt_file in txt_files:
            trajectory_length = extract_trajectory_length(txt_file)

            if trajectory_length is not None and trajectory_length > threshold:
                filename = os.path.basename(txt_file)
                long_trajectories.append((filename, trajectory_length))

        if long_trajectories:
            # Sort by trajectory length (descending)
            long_trajectories.sort(key=lambda x: x[1], reverse=True)
            results[search_dir] = long_trajectories

    return results


def main():
    """Main function to run the trajectory length analysis."""
    # Get the base directory (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..", "..")  # Go up to project root
    base_dir = os.path.abspath(base_dir)

    print(f"Base directory: {base_dir}")
    print(f"Looking for trajectory files with length > 50...")
    print("=" * 60)

    # Find long trajectories
    threshold = 40
    results = find_long_trajectories(base_dir, threshold=threshold)

    if not results:
        print(f"No trajectory files with length > {threshold} found in any directory.")
        return

    # Print results
    total_files = 0
    for dir_name, trajectories in results.items():
        print(f"\n{dir_name}:")
        print(f"Found {len(trajectories)} files with trajectory length > 50")
        print("-" * 40)

        for filename, length in trajectories:
            print(f"  {filename}: {length} steps")

        total_files += len(trajectories)

    print("=" * 60)
    print(f"Total files with long trajectories: {total_files}")

    # Optionally, write results to a file
    output_file = os.path.join(base_dir, "long_trajectory_files.txt")
    with open(output_file, "w") as f:
        f.write("Files with trajectory length > 50\n")
        f.write("=" * 40 + "\n\n")

        for dir_name, trajectories in results.items():
            f.write(f"{dir_name}:\n")
            for filename, length in trajectories:
                f.write(f"  {filename}: {length} steps\n")
            f.write("\n")

    print(f"Results also saved to: {output_file}")


if __name__ == "__main__":
    main()
