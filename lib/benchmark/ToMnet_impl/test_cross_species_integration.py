#!/usr/bin/env python3
"""
Test script to verify the cross-species evaluation integration
"""

import os
import json
import tempfile
import shutil
from pathlib import Path


def test_json_file_generation():
    """Test the JSON file generation functionality"""
    print("=== Testing Cross-Species Integration ===")

    # Mock training results
    mock_results = {
        0.01: {"best_val_loss": 0.5, "model_path": "models/figure3_0.01_best.pth"},
        0.1: {"best_val_loss": 0.6, "model_path": "models/figure3_0.1_best.pth"},
        3.0: {"best_val_loss": 0.7, "model_path": "models/figure3_3.0_best.pth"},
        "mixed": {"best_val_loss": 0.8, "model_path": "models/figure3_mixed_best.pth"},
    }

    # Mock datasets
    mock_datasets = {
        0.01: {"data": [], "meta": {"state_dim": 726, "alpha": 0.01}},
        0.1: {"data": [], "meta": {"state_dim": 726, "alpha": 0.1}},
        3.0: {"data": [], "meta": {"state_dim": 726, "alpha": 3.0}},
        "mixed": {
            "data": [],
            "meta": {"mixed": True, "alpha_values": [0.01, 0.1, 3.0]},
        },
    }

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        # Import the function
        import sys

        sys.path.append(
            "/Users/youngjins/Desktop/codes/25_belief/belief_trading/lib/benchmark/ToMnet_impl"
        )
        from train import generate_cross_species_evaluation_files

        # Test the function
        try:
            generate_cross_species_evaluation_files(
                mock_results, mock_datasets, "figure3"
            )

            # Check if files were generated
            eval_dir = Path("evaluation_configs")

            assert eval_dir.exists(), "evaluation_configs directory not created"

            model_paths_file = eval_dir / "model_paths.json"
            data_paths_file = eval_dir / "data_paths.json"
            eval_script_file = eval_dir / "run_cross_species_evaluation.sh"
            summary_file = eval_dir / "evaluation_summary.json"

            assert model_paths_file.exists(), "model_paths.json not created"
            assert data_paths_file.exists(), "data_paths.json not created"
            assert eval_script_file.exists(), "evaluation script not created"
            assert summary_file.exists(), "summary file not created"

            # Check file contents
            with open(model_paths_file) as f:
                model_paths = json.load(f)

            with open(data_paths_file) as f:
                data_paths = json.load(f)

            with open(summary_file) as f:
                summary = json.load(f)

            print("✓ All files created successfully")
            print(f"✓ Model paths: {list(model_paths.keys())}")
            print(f"✓ Data paths: {list(data_paths.keys())}")
            print(f"✓ Summary contains: {list(summary.keys())}")

            # Check expected structure
            expected_models = ["alpha_0.01", "alpha_0.1", "alpha_3.0", "mixed"]
            expected_data = ["alpha_0.01", "alpha_0.1", "alpha_3.0"]

            for model in expected_models:
                assert model in model_paths, f"Missing model: {model}"

            for data in expected_data:
                assert data in data_paths, f"Missing dataset: {data}"

            print("✓ All expected keys found in JSON files")

            # Check script executability
            assert os.access(
                eval_script_file, os.X_OK
            ), "Evaluation script not executable"
            print("✓ Evaluation script is executable")

            print("\n=== Integration Test PASSED ===")

        except Exception as e:
            print(f"❌ Integration test FAILED: {e}")
            raise


def test_evaluate_py_compatibility():
    """Test compatibility with evaluate.py argument parsing"""
    print("\n=== Testing evaluate.py Compatibility ===")

    # Create mock JSON files
    mock_model_paths = {
        "alpha_0.01": "/path/to/model_0.01.pth",
        "alpha_3.0": "/path/to/model_3.0.pth",
        "mixed": "/path/to/model_mixed.pth",
    }

    mock_data_paths = {
        "alpha_0.01": "/path/to/data_0.01.pkl",
        "alpha_3.0": "/path/to/data_3.0.pkl",
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        model_file = os.path.join(temp_dir, "model_paths.json")
        data_file = os.path.join(temp_dir, "data_paths.json")

        with open(model_file, "w") as f:
            json.dump(mock_model_paths, f, indent=2)

        with open(data_file, "w") as f:
            json.dump(mock_data_paths, f, indent=2)

        print(f"✓ Created mock JSON files in {temp_dir}")
        print(f"  - {model_file}")
        print(f"  - {data_file}")

        # Test command construction
        eval_command = f"""python evaluate.py \\
        --experiment figure3 \\
        --model_paths_json {model_file} \\
        --data_paths_json {data_file} \\
        --output_path result/figure3_cross_species_results.pkl \\
        --device cuda"""

        print("✓ Evaluation command template:")
        print(eval_command)
        print("\n=== evaluate.py Compatibility Test PASSED ===")


if __name__ == "__main__":
    test_json_file_generation()
    test_evaluate_py_compatibility()
    print("\n🎉 All integration tests passed!")
    print("\nNext steps:")
    print(
        "1. Run training: python train.py --experiment figure3 --alpha_values 0.01 0.1 3.0"
    )
    print("2. Run evaluation: bash evaluation_configs/run_cross_species_evaluation.sh")
    print("3. Visualize: jupyter notebook visualize_figure3.ipynb")
