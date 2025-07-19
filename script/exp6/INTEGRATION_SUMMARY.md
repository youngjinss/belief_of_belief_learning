# Integration Summary: Single-Agent and Multi-Agent Systems in Exp6

## Overview
Successfully integrated single-agent (KeyDoor environment) and multi-agent (AchieverBlocker environment) systems in exp6, allowing flexible experimentation with both paradigms.

## Key Changes

### 1. Configuration (script/exp6/config.py)
- Added `is_single_agent_mode()` method to detect when no blockers are configured
- Updated `get_env_name()` to return appropriate environment based on mode:
  - Single-agent: `MiniGrid-KeyDoor-{size}-v1`
  - Multi-agent: `MiniGrid-AchieverBlocker-{size}-v1`
- Modified `get_agent_pair_name()` to handle single-agent data paths
- Updated `get_data_path()` and related methods to support blocker_type=None
- Added `get_test_data_proportion()` method to use existing training_proportion
- Enhanced `update_from_args()` to handle blocker_type="none" for single-agent mode

### 2. Environment Selection (script/exp6/generate.py)
- Added KeyDoorEnv import from `lib/env/gym_minigrid/envs/keydoor.py`
- Updated `run_single_game()` to detect single-agent mode and create appropriate environment
- Modified environment creation logic to handle both KeyDoor and AchieverBlocker environments
- Added `env_to_maze_format_single_agent()` function for single-agent maze representation

### 3. Agent Handling
- Conditional blocker agent creation - only created in multi-agent mode
- Updated game loop to handle single-agent step execution
- Modified reward handling for single-agent mode (scalar vs dict rewards)
- Updated key tracking to use `agent_keys` (KeyDoor) vs `achiever_keys` (AchieverBlocker)

### 4. Data Structure
- Single-agent data paths: `./data/{env_name}/{achiever_type}/`
- Multi-agent data paths: `./data/{env_name}/{achiever_type}_{blocker_type}/`
- Updated trajectory data structure to exclude blocker data in single-agent mode
- Modified save_game_with_labels() to handle None blocker values

### 5. Trajectory Format
- Single-agent format: `[x, y] : action : interaction`
- Multi-agent format: `[x1, y1][x2, y2] : action1,action2 : interaction1,interaction2`
- Conditional blocker section in saved files (only written in multi-agent mode)

### 6. Command Line Interface
- Added "none" as a valid blocker_type choice for single-agent mode
- Example usage:
  - Single-agent: `python script/exp6/generate.py --config_override --blocker_type none`
  - Multi-agent: `python script/exp6/generate.py` (default behavior)

## Benefits
1. **Unified Codebase**: Single codebase handles both single and multi-agent scenarios
2. **Flexible Experimentation**: Easy to switch between paradigms via configuration
3. **Backward Compatible**: Existing multi-agent functionality preserved
4. **Clean Data Organization**: Clear separation of single vs multi-agent data
5. **Configuration-Driven**: Uses existing config.py settings, avoiding hardcoded values

## Testing
Created `test_integration.py` to verify both modes work correctly and demonstrate usage patterns.