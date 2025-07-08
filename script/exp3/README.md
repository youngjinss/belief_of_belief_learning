# Exp3: Custom KeyDoor Environment

This directory contains the implementation and testing of a custom MiniGrid environment with 4 keys and 4 doors, designed for experiments with preference-based decision making and key collection strategies.

## Overview

The KeyDoor environment extends the MiniGrid framework to create a multi-key, multi-door scenario where agents must collect keys to open doors based on their preferences and costs.

## Environment Registration Process

### 1. Environment Implementation

The custom KeyDoor environment is implemented in:
```
lib/env/gym_minigrid/envs/keydoor.py
```

**Key Components:**
- `KeyDoorEnv`: Base environment class
- `KeyDoor3x3Env`, `KeyDoor5x5Env`, `KeyDoor9x9Env`, `KeyDoor11x11Env`: Size-specific variants

### 2. Registration System

The environment registration follows the standard gym/gymnasium pattern:

#### a) Environment Registration
```python
# In keydoor.py
from gym_minigrid.register import register

register(id="MiniGrid-KeyDoor-3x3-v0", entry_point="gym_minigrid.envs:KeyDoor3x3Env")
register(id="MiniGrid-KeyDoor-5x5-v0", entry_point="gym_minigrid.envs:KeyDoor5x5Env")
register(id="MiniGrid-KeyDoor-9x9-v0", entry_point="gym_minigrid.envs:KeyDoor9x9Env")
register(id="MiniGrid-KeyDoor-11x11-v0", entry_point="gym_minigrid.envs:KeyDoor11x11Env")
```

#### b) Module Import Chain
```python
# lib/env/gym_minigrid/__init__.py
from . import envs

# lib/env/gym_minigrid/envs/__init__.py
from .keydoor import *  # This triggers registration
```

#### c) Registration Function
```python
# lib/env/gym_minigrid/register.py
def register(id, entry_point, reward_threshold=0.95):
    gym_register(id=id, entry_point=entry_point, reward_threshold=reward_threshold)
    # Also registers with gymnasium if available
```

### 3. Import Path Setup

For scripts to access the custom environment, the correct import path must be configured:

```python
# Add gym_minigrid to Python path
gym_minigrid_path = os.path.join(os.path.dirname(__file__), '../../lib/env')
sys.path.insert(0, gym_minigrid_path)

# Import gym_minigrid (triggers registration)
import gym_minigrid
```

**Critical Note:** The path must point to `lib/env` (not just `lib`) to ensure the local gym_minigrid module is imported instead of the system-installed one.

## Environment Features

### Grid Layout
- **Sizes:** 3x3, 5x5, 9x9, 11x11
- **Walls:** Surrounding walls with doors placed on walls
- **Objects:** 4 keys and 4 doors with matching colors (red, green, blue, yellow)
- **Observability:** Full observability (agent can see entire grid)

### Agent Capabilities
- **Actions:** 6 actions (up=0, down=1, left=2, right=3, stay=4, pickup=5)
- **Inventory:** Can carry multiple keys (configurable `max_keys`)
- **Movement:** Standard MiniGrid movement with collision detection

### Reward Structure
- **Target Key Collection:** +0.5 reward for collecting target color key
- **Non-target Key Collection:** Negative reward based on cost parameter
- **Door Opening:** Final reward based on preference parameter when target door is opened
- **Episode Termination:** When agent opens target door

### Configuration Parameters
```python
preference = {'red': 1.0, 'green': 0.8, 'blue': 0.6, 'yellow': 0.4}
cost = {'red': 0.1, 'green': 0.2, 'blue': 0.3, 'yellow': 0.4}
max_keys = 4  # Maximum keys agent can carry
```

## Usage Examples

### Basic Environment Creation
```python
import gymnasium as gym
import gym_minigrid

# Create environment
env = gym.make('MiniGrid-KeyDoor-5x5-v0')

# Reset and run
obs, info = env.reset()
action = 5  # pickup action
obs, reward, terminated, truncated, info = env.step(action)
```

### Direct Class Instantiation
```python
from gym_minigrid.envs.keydoor import KeyDoor5x5Env

# Create with custom parameters
env = KeyDoor5x5Env(
    max_keys=2,
    preference={'red': 2.0, 'green': 1.0, 'blue': 0.5, 'yellow': 0.1},
    cost={'red': 0.05, 'green': 0.1, 'blue': 0.2, 'yellow': 0.3}
)
```

## Testing

### Simple Test Script
```bash
python simple_test.py
```

This script:
1. Imports the custom gym_minigrid module
2. Verifies environment registration
3. Creates KeyDoor environment instance
4. Tests reset and step functionality
5. Displays environment information

### Expected Output
```
 gym_minigrid imported successfully
Registered environments: [... 'MiniGrid-KeyDoor-3x3-v0', 'MiniGrid-KeyDoor-5x5-v0', ...]
 KeyDoor5x5Env created successfully
 Environment reset successful. Mission: collect red key and open red door
 Step successful. Reward: 0
 All tests passed!
```

## Key Implementation Details

### Environment Generation
1. **Grid Creation:** Creates empty grid with surrounding walls
2. **Key Placement:** Randomly places 4 keys in open spaces
3. **Door Placement:** Places 4 doors on wall positions
4. **Agent Placement:** Randomly places agent in open space
5. **Target Selection:** Selects target door based on highest preference

### Action Handling
- **Movement Actions (0-4):** Standard MiniGrid movement
- **Pickup Action (5):** Custom implementation for key collection
- **Key-Door Matching:** Keys must match door colors to unlock
- **Inventory Management:** Tracks collected keys in `agent_keys` list

### Reward Calculation
- **Immediate Rewards:** Given for key collection
- **Final Reward:** Given when target door is opened
- **Cost Penalties:** Applied for collecting non-target keys

## Troubleshooting

### Common Issues
1. **Environment Not Found:** Ensure correct import path setup
2. **Array Comparison Errors:** Fixed by using tuple conversion for position comparisons
3. **Reset Return Values:** Handled both old and new gymnasium return formats

### Import Path Issues
The most critical aspect is ensuring the local gym_minigrid module is imported:
```python
# Correct path - points to lib/env
gym_minigrid_path = os.path.join(os.path.dirname(__file__), '../../lib/env')
sys.path.insert(0, gym_minigrid_path)
```

This ensures the custom KeyDoor environment is registered and available for use.