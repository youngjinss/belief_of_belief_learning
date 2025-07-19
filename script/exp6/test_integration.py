#!/usr/bin/env python3
"""
Test script to demonstrate single-agent and multi-agent integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from script.exp6.config import Config

def test_single_agent_config():
    """Test single-agent configuration"""
    print("=== Testing Single-Agent Configuration ===")
    
    # Create config with no blockers
    config = Config()
    config.achiever_types = {"lv0va": 100, "lv1va": 100}
    config.blocker_types = {}  # Empty dict for single-agent mode
    
    print(f"Is single-agent mode: {config.is_single_agent_mode()}")
    print(f"Environment name: {config.get_env_name()}")
    print(f"Test data proportion: {config.get_test_data_proportion()}")
    
    # Test data paths
    for achiever_type in config.achiever_types:
        data_path = config.get_data_path(achiever_type, None)
        print(f"Data path for {achiever_type}: {data_path}")
        
        test_path = config.get_data_path(achiever_type, None, is_test=True)
        print(f"Test data path for {achiever_type}: {test_path}")
    
    print()

def test_multi_agent_config():
    """Test multi-agent configuration"""
    print("=== Testing Multi-Agent Configuration ===")
    
    # Create config with blockers
    config = Config()
    config.achiever_types = {"lv0va": 100, "lv1va": 100}
    config.blocker_types = {"lv0vb": 100, "lv1vb": 100}
    
    print(f"Is single-agent mode: {config.is_single_agent_mode()}")
    print(f"Environment name: {config.get_env_name()}")
    print(f"Test data proportion: {config.get_test_data_proportion()}")
    
    # Test data paths
    for achiever_type in config.achiever_types:
        for blocker_type in config.blocker_types:
            data_path = config.get_data_path(achiever_type, blocker_type)
            print(f"Data path for {achiever_type} vs {blocker_type}: {data_path}")
            
            test_path = config.get_data_path(achiever_type, blocker_type, is_test=True)
            print(f"Test data path for {achiever_type} vs {blocker_type}: {test_path}")
    
    print()

def test_generate_command():
    """Show example commands for generation"""
    print("=== Example Generation Commands ===")
    
    print("Single-agent mode (KeyDoor environment):")
    print("  python script/exp6/generate.py --config_override --blocker_type none")
    print()
    
    print("Multi-agent mode (AchieverBlocker environment):")
    print("  python script/exp6/generate.py")
    print()
    
    print("Generate test data:")
    print("  python script/exp6/generate.py --test_data")
    print()

if __name__ == "__main__":
    test_single_agent_config()
    test_multi_agent_config()
    test_generate_command()