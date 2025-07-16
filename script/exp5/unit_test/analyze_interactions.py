#!/usr/bin/env python3
"""
Unit test script to analyze interactions and win rates for AchieverBlocker combinations.

Analyzes interaction patterns and calculates win rates from trajectory data files
in data/MiniGrid-AchieverBlocker-9x9-v1/ directories.
"""

import os
import re
import glob
from collections import defaultdict, Counter
from pathlib import Path


class InteractionAnalyzer:
    """Analyzer for AchieverBlocker interaction patterns and win rates."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(
            base_dir, "data", "MiniGrid-AchieverBlocker-9x9-v1"
        )

        # Define interaction types
        self.achiever_interactions = {"A", "B", "C", "D", "a", "b", "c", "d", "X"}
        self.blocker_interactions = {"0", "1", "X"}

        # Define win conditions
        self.achiever_win_interactions = {"a", "b", "c", "d"}  # Door opening
        self.blocker_win_interactions = {"1"}  # Successful block
        
        # Define simultaneous success interactions
        self.simultaneous_success_patterns = {
            # Achiever opens door AND blocker succeeds  
            ("a", "1"), ("b", "1"), ("c", "1"), ("d", "1")
        }

    def parse_trajectory_file(self, file_path):
        """
        Parse a single trajectory file to extract interaction data.

        Args:
            file_path: Path to the trajectory file

        Returns:
            dict: Parsed data including interactions and game outcome
        """
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        data = {
            "trajectory_length": None,
            "achiever_interactions": [],
            "blocker_interactions": [],
            "last_achiever_interaction": None,
            "last_blocker_interaction": None,
            "winner": None,
            "filename": os.path.basename(file_path),
            "simultaneous_interactions": [],  # Track simultaneous interactions
            "has_simultaneous_success": False,  # Flag for simultaneous success
        }

        # Extract trajectory length
        for line in lines:
            match = re.search(r"Trajectory length:\s*(\d+)", line)
            if match:
                data["trajectory_length"] = int(match.group(1))
                break

        # Extract interactions from trajectory lines
        # Format: [x1, y1][x2, y2] : action1,action2 : interaction1,interaction2
        interaction_pattern = r"\[(\d+),\s*(\d+)\]\[(\d+),\s*(\d+)\]\s*:\s*(\d+),(\d+)\s*:\s*([^,]+),([^\n]+)"

        for line in lines:
            match = re.search(interaction_pattern, line)
            if match:
                achiever_interaction = match.group(7).strip()
                blocker_interaction = match.group(8).strip()

                data["achiever_interactions"].append(achiever_interaction)
                data["blocker_interactions"].append(blocker_interaction)
                
                # Check for simultaneous interactions (both agents not "X")
                if achiever_interaction != "X" and blocker_interaction != "X":
                    simultaneous_pair = (achiever_interaction, blocker_interaction)
                    data["simultaneous_interactions"].append(simultaneous_pair)
                    
                    # Check if this is a simultaneous success
                    if simultaneous_pair in self.simultaneous_success_patterns:
                        data["has_simultaneous_success"] = True

        # Get last interactions (determine winner)
        if data["achiever_interactions"]:
            data["last_achiever_interaction"] = data["achiever_interactions"][-1]
        if data["blocker_interactions"]:
            data["last_blocker_interaction"] = data["blocker_interactions"][-1]

        # Determine winner based on last interactions and simultaneous success
        data["winner"] = self._determine_winner(
            data["last_achiever_interaction"], 
            data["last_blocker_interaction"],
            data["has_simultaneous_success"]
        )

        return data

    def _determine_winner(self, last_achiever_interaction, last_blocker_interaction, has_simultaneous_success=False):
        """
        Determine the winner based on last interactions and simultaneous success.

        Args:
            last_achiever_interaction: Last interaction by achiever
            last_blocker_interaction: Last interaction by blocker
            has_simultaneous_success: Whether there was simultaneous success during the game

        Returns:
            str: 'achiever', 'blocker', 'simultaneous_success', or 'need_to_examine'
        """
        # Check for simultaneous success first (both agents succeed at the same time)
        if has_simultaneous_success:
            last_pair = (last_achiever_interaction, last_blocker_interaction)
            if last_pair in self.simultaneous_success_patterns:
                return "simultaneous_success"
        
        # Check individual wins
        if last_achiever_interaction in self.achiever_win_interactions:
            return "achiever"
        elif last_blocker_interaction in self.blocker_win_interactions:
            return "blocker"
        else:
            return "need_to_examine"

    def analyze_directory(self, dir_name):
        """
        Analyze all trajectory files in a specific directory.

        Args:
            dir_name: Directory name (e.g., 'astar_random')

        Returns:
            dict: Analysis results for the directory
        """
        dir_path = os.path.join(self.data_dir, dir_name)

        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            return None

        # Find all .txt files
        txt_files = glob.glob(os.path.join(dir_path, "*.txt"))

        results = {
            "directory": dir_name,
            "total_files": len(txt_files),
            "achiever_interaction_counts": Counter(),
            "blocker_interaction_counts": Counter(),
            "winner_counts": Counter(),
            "trajectory_lengths": [],
            "parsed_files": 0,
            "failed_files": 0,
            "need_to_examine_files": [],  # Store files that need examination
            "min_trajectory_files": [],  # Store files with minimum trajectory length
            "simultaneous_interaction_counts": Counter(),  # Track simultaneous interactions
            "simultaneous_success_files": [],  # Store files with simultaneous success
        }

        print(f"Analyzing {dir_name}...")

        for txt_file in txt_files:
            data = self.parse_trajectory_file(txt_file)

            if data is None:
                results["failed_files"] += 1
                continue

            results["parsed_files"] += 1

            # Count interactions
            for interaction in data["achiever_interactions"]:
                results["achiever_interaction_counts"][interaction] += 1

            for interaction in data["blocker_interactions"]:
                results["blocker_interaction_counts"][interaction] += 1

            # Count winners
            results["winner_counts"][data["winner"]] += 1

            # Track simultaneous interactions
            for interaction_pair in data["simultaneous_interactions"]:
                results["simultaneous_interaction_counts"][interaction_pair] += 1
            
            # Track files with simultaneous success
            if data["has_simultaneous_success"]:
                results["simultaneous_success_files"].append(
                    {
                        "filename": data["filename"],
                        "last_achiever": data["last_achiever_interaction"],
                        "last_blocker": data["last_blocker_interaction"],
                        "trajectory_length": data["trajectory_length"],
                        "simultaneous_interactions": data["simultaneous_interactions"],
                    }
                )

            # Track files that need examination
            if data["winner"] == "need_to_examine":
                results["need_to_examine_files"].append(
                    {
                        "filename": data["filename"],
                        "last_achiever": data["last_achiever_interaction"],
                        "last_blocker": data["last_blocker_interaction"],
                        "trajectory_length": data["trajectory_length"],
                    }
                )

            # Track trajectory lengths
            if data["trajectory_length"] is not None:
                results["trajectory_lengths"].append(data["trajectory_length"])

        # Find files with minimum trajectory length
        if results["trajectory_lengths"]:
            min_length = min(results["trajectory_lengths"])

            # Second pass to find all files with minimum length
            for txt_file in txt_files:
                data = self.parse_trajectory_file(txt_file)
                if (
                    data is not None
                    and data["trajectory_length"] is not None
                    and data["trajectory_length"] == min_length
                ):
                    results["min_trajectory_files"].append(
                        {
                            "filename": data["filename"],
                            "trajectory_length": data["trajectory_length"],
                            "winner": data["winner"],
                            "last_achiever": data["last_achiever_interaction"],
                            "last_blocker": data["last_blocker_interaction"],
                        }
                    )

        return results

    def analyze_all_combinations(self):
        """
        Analyze all achiever-blocker combinations in the data directory.

        Returns:
            dict: Complete analysis results
        """
        # Find all subdirectories
        subdirs = []
        if os.path.exists(self.data_dir):
            subdirs = [
                d
                for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))
            ]

        if not subdirs:
            print(f"No subdirectories found in {self.data_dir}")
            return {}

        print(f"Found directories: {subdirs}")
        print("=" * 60)

        all_results = {}

        for subdir in sorted(subdirs):
            results = self.analyze_directory(subdir)
            if results:
                all_results[subdir] = results

        return all_results

    def print_summary(self, all_results):
        """
        Print a comprehensive summary of all results.

        Args:
            all_results: Dictionary of analysis results for all directories
        """
        print("\n" + "=" * 80)
        print("INTERACTION ANALYSIS SUMMARY")
        print("=" * 80)

        for dir_name, results in all_results.items():
            print(f"\n{dir_name.upper()}:")
            print("-" * 50)

            # Basic stats
            print(f"Total files: {results['total_files']}")
            print(f"Successfully parsed: {results['parsed_files']}")
            print(f"Failed to parse: {results['failed_files']}")

            if results["trajectory_lengths"]:
                avg_length = sum(results["trajectory_lengths"]) / len(
                    results["trajectory_lengths"]
                )
                print(f"Average trajectory length: {avg_length:.1f}")
                print(
                    f"Min/Max trajectory length: {min(results['trajectory_lengths'])}/{max(results['trajectory_lengths'])}"
                )

            # Interaction counts
            print(f"\nAchiever interactions:")
            for interaction in sorted(self.achiever_interactions):
                count = results["achiever_interaction_counts"][interaction]
                if count > 0:
                    print(f"  {interaction}: {count}")

            print(f"\nBlocker interactions:")
            for interaction in sorted(self.blocker_interactions):
                count = results["blocker_interaction_counts"][interaction]
                if count > 0:
                    print(f"  {interaction}: {count}")

            # Win rates
            print(f"\nWin rates:")
            total_games = results["parsed_files"]
            if total_games > 0:
                for winner, count in results["winner_counts"].items():
                    rate = (count / total_games) * 100
                    print(f"  {winner}: {count}/{total_games} ({rate:.1f}%)")
            
            # Simultaneous interactions
            if results["simultaneous_interaction_counts"]:
                print(f"\nSimultaneous interactions:")
                for interaction_pair, count in results["simultaneous_interaction_counts"].most_common():
                    print(f"  {interaction_pair}: {count}")
            
            # Files with simultaneous success
            if results["simultaneous_success_files"]:
                print(f"\nFiles with simultaneous success:")
                for file_info in results["simultaneous_success_files"]:
                    print(
                        f"  {file_info['filename']} (length: {file_info['trajectory_length']}, "
                        f"last achiever: {file_info['last_achiever']}, "
                        f"last blocker: {file_info['last_blocker']}, "
                        f"simultaneous: {file_info['simultaneous_interactions']})"
                    )

            # Print files that need examination
            if results["need_to_examine_files"]:
                print(f"\nFiles that need examination:")
                for file_info in results["need_to_examine_files"]:
                    print(
                        f"  {file_info['filename']} (length: {file_info['trajectory_length']}, "
                        f"last achiever: {file_info['last_achiever']}, "
                        f"last blocker: {file_info['last_blocker']})"
                    )

            # Print files with minimum trajectory length
            if results["min_trajectory_files"]:
                min_length = results["min_trajectory_files"][0]["trajectory_length"]
                print(f"\nFiles with minimum trajectory length ({min_length} steps):")
                for file_info in results["min_trajectory_files"]:
                    print(
                        f"  {file_info['filename']} (winner: {file_info['winner']}, "
                        f"last achiever: {file_info['last_achiever']}, "
                        f"last blocker: {file_info['last_blocker']})"
                    )

            print()

    def save_detailed_report(self, all_results, output_file):
        """
        Save a detailed report to a file.

        Args:
            all_results: Dictionary of analysis results
            output_file: Path to output file
        """
        with open(output_file, "w") as f:
            f.write("AchieverBlocker Interaction Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            for dir_name, results in all_results.items():
                f.write(f"{dir_name}:\n")
                f.write("-" * 30 + "\n")

                # Summary stats
                f.write(
                    f"Files: {results['parsed_files']}/{results['total_files']} parsed successfully\n"
                )

                if results["trajectory_lengths"]:
                    avg_length = sum(results["trajectory_lengths"]) / len(
                        results["trajectory_lengths"]
                    )
                    f.write(f"Average trajectory length: {avg_length:.1f}\n")

                # Detailed interaction counts
                f.write("\nAchiever interactions:\n")
                for interaction, count in results[
                    "achiever_interaction_counts"
                ].most_common():
                    f.write(f"  {interaction}: {count}\n")

                f.write("\nBlocker interactions:\n")
                for interaction, count in results[
                    "blocker_interaction_counts"
                ].most_common():
                    f.write(f"  {interaction}: {count}\n")

                # Win rates
                f.write("\nOutcomes:\n")
                total_games = results["parsed_files"]
                for winner, count in results["winner_counts"].items():
                    if total_games > 0:
                        rate = (count / total_games) * 100
                        f.write(f"  {winner}: {count}/{total_games} ({rate:.1f}%)\n")
                    else:
                        f.write(f"  {winner}: {count}/0 (N/A%)\n")
                
                # Simultaneous interactions
                if results["simultaneous_interaction_counts"]:
                    f.write("\nSimultaneous interactions:\n")
                    for interaction_pair, count in results["simultaneous_interaction_counts"].most_common():
                        f.write(f"  {interaction_pair}: {count}\n")
                
                # Files with simultaneous success
                if results["simultaneous_success_files"]:
                    f.write("\nFiles with simultaneous success:\n")
                    for file_info in results["simultaneous_success_files"]:
                        f.write(
                            f"  {file_info['filename']} (length: {file_info['trajectory_length']}, "
                            f"last achiever: {file_info['last_achiever']}, "
                            f"last blocker: {file_info['last_blocker']}, "
                            f"simultaneous: {file_info['simultaneous_interactions']})\n"
                        )

                # List files that need examination
                if results["need_to_examine_files"]:
                    f.write("\nFiles that need examination:\n")
                    for file_info in results["need_to_examine_files"]:
                        f.write(
                            f"  {file_info['filename']} (length: {file_info['trajectory_length']}, "
                            f"last achiever: {file_info['last_achiever']}, "
                            f"last blocker: {file_info['last_blocker']})\n"
                        )

                # List files with minimum trajectory length
                if results["min_trajectory_files"]:
                    min_length = results["min_trajectory_files"][0]["trajectory_length"]
                    f.write(
                        f"\nFiles with minimum trajectory length ({min_length} steps):\n"
                    )
                    for file_info in results["min_trajectory_files"]:
                        f.write(
                            f"  {file_info['filename']} (winner: {file_info['winner']}, "
                            f"last achiever: {file_info['last_achiever']}, "
                            f"last blocker: {file_info['last_blocker']})\n"
                        )

                f.write("\n" + "=" * 50 + "\n\n")


def main():
    """Main function to run the interaction analysis."""
    # Get the base directory (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..", "..")  # Go up to project root
    base_dir = os.path.abspath(base_dir)

    print(f"Base directory: {base_dir}")

    # Create analyzer
    analyzer = InteractionAnalyzer(base_dir)

    # Run analysis
    all_results = analyzer.analyze_all_combinations()

    if not all_results:
        print("No data found to analyze.")
        return

    # Print summary
    analyzer.print_summary(all_results)

    # Save detailed report
    output_file = os.path.join(base_dir, "interaction_analysis_report.txt")
    analyzer.save_detailed_report(all_results, output_file)
    print(f"\nDetailed report saved to: {output_file}")


if __name__ == "__main__":
    main()
