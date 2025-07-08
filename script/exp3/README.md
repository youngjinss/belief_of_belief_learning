# KeyDoor Environment Experiments

This directory contains experiments for the KeyDoor environment with A* and random agents.

## Structure

- `agents.py` - Agent implementations (AStarAgent, RandomAgent)  
- `config.py` - Configuration class for experiments
- `render_kd.py` - Main rendering script with agent_type argument
- `backup/` - Backup files and old implementations

## Usage

### A* Agent
```bash
python render_kd.py --agent_type astar --episodes 3 --pause 0.5 --debug
```

### Random Agent
```bash
python render_kd.py --agent_type random --episodes 3 --pause 0.3
```

### Available Arguments

- `--agent_type {astar,random}` - Agent type to use
- `--seed SEED` - Random seed (default: 42)
- `--episodes EPISODES` - Number of episodes (default: 3)
- `--pause PAUSE` - Pause between actions in seconds (default: 0.5)
- `--max_steps MAX_STEPS` - Maximum steps per episode (default: 500)
- `--env_size {3x3,5x5,9x9,11x11}` - Environment size (default: 9x9)
- `--observability {full,partial}` - Observability type (default: full)
- `--debug` - Enable debug output

## Features

- **Automatic key pickup**: Agent automatically picks up keys when stepping on them
- **Automatic door opening**: Agent automatically opens doors when stepping on them (if they have the key)
- **Turn-based navigation**: A* agent uses proper MiniGrid turn-based movement
- **Configurable environment**: Support for different grid sizes and observability modes
- **Structured code**: Following ToMnetF pattern with separate agents, config, and visualization files

## Environment Details

The KeyDoor environment requires the agent to:
1. Collect the target key (color matches target door)
2. Navigate to the target door
3. Open the door by stepping on it (automatic with key)

Success is achieved when the agent reaches the opened door position.

## Code Structure

This code follows the ToMnetF pattern with:
- **agents.py**: Contains all agent implementations (AStarAgent, RandomAgent)
- **config.py**: Configuration class with environment and agent parameters
- **render_kd.py**: Main script that handles rendering and episode running

## Agent Types

### AStarAgent
- Uses A* pathfinding algorithm for optimal navigation
- Two-phase strategy: collect key, then open door
- Handles MiniGrid's turn-based movement system (turn left/right, then forward)
- Automatic key collection and door opening when stepping on objects

### RandomAgent
- Performs random actions with bias towards movement
- 80% chance of movement actions (turn_left, turn_right, forward)
- 20% chance of toggle action (for door opening)
- No learning or strategic behavior

## Environment Registration

The custom KeyDoor environment is registered in:
```
lib/env/gym_minigrid/envs/keydoor.py
```

Import path setup:
```python
# Add gym_minigrid to Python path
gym_minigrid_path = os.path.join(os.path.dirname(__file__), '../../lib/env')
sys.path.insert(0, gym_minigrid_path)
import gym_minigrid
```