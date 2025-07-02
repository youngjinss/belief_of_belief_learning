#!/usr/bin/env python3
"""
Data generation script for Figure 5 experiments
This script generates training data for goal-directed agents in the ToMnet experiments
"""

import argparse
import os
import sys
import numpy as np
from data_generation import DataGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 5 data with goal-directed agents"
    )

    # Data generation parameters
    parser.add_argument("--n_agents", type=int, default=100, help="Number of agents")
    parser.add_argument(
        "--n_episodes_per_agent", type=int, default=100, help="Episodes per agent"
    )
    parser.add_argument(
        "--alpha_reward",
        type=float,
        default=0.01,
        help="Dirichlet concentration for rewards",
    )
    parser.add_argument(
        "--high_cost_ratio", type=float, default=0.2, help="Ratio of high-cost agents"
    )
    parser.add_argument("--min_past", type=int, default=0, help="Minimum past episodes")
    parser.add_argument(
        "--max_past", type=int, default=10, help="Maximum past episodes"
    )
    parser.add_argument(
        "--n_workers", type=int, default=None, help="Number of parallel workers"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="data/figure5", help="Output directory"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="goal_directed", help="Experiment name"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Figure 5 Data Generation")
    print("=" * 60)
    print(f"Agents: {args.n_agents}")
    print(f"Episodes per agent: {args.n_episodes_per_agent}")
    print(f"Alpha reward: {args.alpha_reward}")
    print(f"High cost ratio: {args.high_cost_ratio}")
    print(f"Past episodes range: {args.min_past}-{args.max_past}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize data generator
    generator = DataGenerator()

    # Generate goal-directed agent data
    save_path = os.path.join(
        args.output_dir, "goal_directed_training_data.pkl"
    )

    print("\nGenerating goal-directed agent data...")
    dataset = generator.generate_goal_directed_agent_data(
        n_agents=args.n_agents,
        n_episodes_per_agent=args.n_episodes_per_agent,
        alpha_reward=args.alpha_reward,
        high_cost_ratio=args.high_cost_ratio,
        min_past=args.min_past,
        max_past=args.max_past,
        save_path=save_path,
        n_workers=args.n_workers,
    )

    print(f"\nData generation completed!")
    print(f"Total samples: {len(dataset['data'])}")
    print(f"Saved to: {save_path}")

    # Save metadata
    meta_path = os.path.join(args.output_dir, f"{args.experiment_name}_meta.txt")
    with open(meta_path, "w") as f:
        f.write("Figure 5 Data Generation Metadata\n")
        f.write("=" * 40 + "\n")
        f.write(f"Agents: {args.n_agents}\n")
        f.write(f"Episodes per agent: {args.n_episodes_per_agent}\n")
        f.write(f"Alpha reward: {args.alpha_reward}\n")
        f.write(f"High cost ratio: {args.high_cost_ratio}\n")
        f.write(f"Past episodes range: {args.min_past}-{args.max_past}\n")
        f.write(f"Total samples: {len(dataset['data'])}\n")
        f.write(f"State dimension: {dataset['meta']['state_dim']}\n")
        f.write(f"Grid size: {dataset['meta']['grid_size']}\n")

    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
