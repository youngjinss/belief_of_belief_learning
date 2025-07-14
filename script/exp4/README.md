# AchieverBlocker Experiment 4 - Multi-Agent ToMnet Implementation

This directory contains a complete multi-agent ToMnet implementation for the AchieverBlocker environment with strategic interaction between two agent types.

## Overview

AchieverBlocker Experiment 4 implements a competitive multi-agent environment where:
1. **Achiever agents** navigate a 9x9 grid to collect preferred keys and open corresponding doors
2. **Blocker agents** observe achiever behavior, infer their goals, and strategically block access
3. **Strategic interaction** creates theory of mind requirements for successful blocking

The implementation extends exp3's ToMnet architecture for multi-agent scenarios with:
- **Dual-agent observations** with role differentiation
- **Agent type prediction** as an additional learning task
- **Strategic blocking behavior** with competitive dynamics
- **Enhanced data generation** for multi-agent trajectories

## Project Structure

```
script/exp4/
├── config.py              # Configuration for multi-agent experiments
├── generate.py            # Multi-agent trajectory data generation
├── train.py               # Enhanced ToMnet training with agent prediction
├── evaluate.py            # Multi-agent evaluation and metrics
├── visualize.py           # Results visualization and plotting
├── visualize_sr.py        # Successor representation visualization
├── tomnet.py              # Extended ToMnet with agent classification head
├── achievers.py           # Achiever agent implementations (A*, Value, Random)
├── blockers.py            # Blocker agent implementations (Random, GoalDirect)
├── data_generation.py     # Multi-agent data processing utilities
├── simulate_game.py       # Multi-agent game simulation with GUI
└── simulate_trajectory.py # Multi-agent trajectory visualization
```

## Key Features

### 🎯 **Multi-Agent Competitive Environment**
- **Achiever agents**: Navigate to collect preferred keys and open doors (7 actions: up, right, down, left, stay, pickup, toggle)
- **Blocker agents**: Infer achiever goals and strategically block access (6 actions: up, right, down, left, stay, broken)
- **Strategic dynamics**: Theory of mind requirements for successful blocking
- **Game termination**: Blockers can end game early when positioned at predicted target door

### 🧠 **Enhanced ToMnet Architecture**
- **Agent classification head**: Predicts whether observing achiever (0) or blocker (1)
- **Multi-agent observations**: Each agent receives full state information about both agents
- **Extended loss function**: Action, goal, agent type, consumption, and successor representation losses
- **Dual-agent training**: Learns behaviors and intentions for both agent types

### 📊 **Advanced Data Processing**
- **Multi-agent trajectory generation**: Two samples per trajectory (one per agent)
- **Agent type labeling**: Each sample tagged with agent role
- **Vectorized SR calculation**: Optimized successor representation computation
- **Enhanced data paths**: Combined naming scheme for agent type pairs

### 🚀 **Strategic Agent Types**

#### Achiever Agents
- **Value Agent**: Strategic navigation with preference optimization
- **A* Agent**: Optimal pathfinding with goal-directed behavior  
- **Random Agent**: Baseline exploration behavior

#### Blocker Agents
- **GoalDirect Agent**: Infers achiever goals from observed key collection patterns
- **Random Agent**: Baseline blocking behavior

## Quick Start

### 1. Multi-Agent Data Generation
```bash
# Generate training data with value achiever and goal-directed blocker
python script/exp4/generate.py --achiever_type value --blocker_type goal_direct

# Generate test data with different agent combinations
python script/exp4/generate.py --n_games 2000 --achiever_type astar --blocker_type random --test_data

# Generate data for all agent combinations
python script/exp4/generate.py --achiever_type value --blocker_type goal_direct
python script/exp4/generate.py --achiever_type astar --blocker_type random
```

### 2. Multi-Agent Game Simulation
```bash
# Run competitive simulation with visualization
python script/exp4/simulate_game.py --episodes 1 --render --achiever_type value --blocker_type goal_direct

# Save gameplay as animated GIF
python script/exp4/simulate_trajectory.py --episodes 5 --gif_output achiever_blocker_demo

# Visualize multi-agent successor representation
python script/exp4/visualize_sr.py --data_file data/MiniGrid-AchieverBlocker-9x9-v1/value_goal_direct/test0.txt
```

### 3. Enhanced Model Training
```bash
# Train with multi-agent data (default: value achiever + goal_direct blocker)
python script/exp4/train.py --save_dir ./results/exp4/

# Custom training with agent type weighting
python script/exp4/train.py \
    --epochs 100 --batch_size 1024 --lr 0.0001 \
    --agent_weight 1.5 --device cuda:0 --save_dir ./results/exp4/
```

### 4. Multi-Agent Evaluation
```bash
# Evaluate multi-agent model performance
python script/exp4/evaluate.py \
    --model_path ./results/exp4/best_model.pth \
    --result_dir ./results/exp4/

# Generate multi-agent visualizations
python script/exp4/visualize.py \
    --result_dir ./results/exp4/ \
    --plot_type all
```

## Configuration

All parameters are centralized in `config.py` for multi-agent experiments:

### Environment Settings
```python
self.env_name = "MiniGrid-AchieverBlocker-{size}-v1"
self.achiever_type = "value"      # "astar", "random", "value"
self.blocker_type = "goal_direct" # "random", "goal_direct"
self.width = 9
self.height = 9
self.max_steps = 500
```

### Model Architecture
```python
self.model_config = {
    "use_mentalnet": True,           # Enable Mental Network
    "residual_blocks": 5,
    "n_echar": 128,                  # Character embedding size
    "n_ement": 128,                  # Mental state embedding size
    "achiever_action_space": 7,      # Achiever actions
    "blocker_action_space": 6,       # Blocker actions
    "goal_space": 4,                 # 4 colored goals
    "agent_space": 2,                # Achiever (0) or Blocker (1)
}
```

### Training Configuration
```python
self.training_config = {
    "batch_size": 1024,
    "epochs": 200,
    "lr": 0.0001,
    "device": "cuda:3",
    "agent_weight": 1.0,             # Weight for agent classification loss
    "early_stopping_patience": 30,
}
```

## Environment Details

The AchieverBlocker environment creates strategic interaction between two agents:

### Achiever Objectives:
1. Navigate the 9x9 grid with 4 colored keys and doors
2. Collect the preferred key (based on goal preferences)
3. Open the corresponding door to win the game

### Blocker Objectives:
1. Observe achiever's movement and key collection patterns
2. Infer achiever's goal preferences through theory of mind
3. Position at the predicted target door and use "broken" action to block access

### Game Dynamics:
- **Turn-based**: Agents act sequentially each timestep
- **Full observability**: Each agent observes both agent positions and states
- **Collision avoidance**: Agents cannot occupy the same grid cell
- **Multiple termination**: Game ends when achiever succeeds OR blocker successfully blocks

## Strategic Agent Implementations

### Achiever Agents (`achievers.py`)

#### Value Achiever
- **Strategic planning**: Uses value iteration with goal preferences
- **Preference optimization**: Maximizes reward based on colored door preferences
- **Anti-blocking**: May adapt strategy if blocked repeatedly

#### A* Achiever  
- **Optimal pathfinding**: Finds shortest path to preferred goals
- **Deterministic behavior**: Predictable for blocker agents to analyze
- **Two-phase strategy**: Collect key → navigate to door

#### Random Achiever
- **Exploration baseline**: Random movement with goal bias
- **Unpredictable behavior**: Difficult for blockers to predict

### Blocker Agents (`blockers.py`)

#### GoalDirect Blocker
- **Theory of mind**: Infers achiever preferences from key collection behavior
- **Strategic positioning**: Moves to predicted target doors
- **Game termination**: Uses "broken" action when positioned optimally

#### Random Blocker
- **Baseline behavior**: Random actions across the environment
- **No strategic reasoning**: Provides comparison baseline

## Data Structure

### Generated Data Paths
```
data/
└── MiniGrid-AchieverBlocker-9x9-v1/
    └── value_goal_direct/              # Training data for agent pair
        ├── test0.txt                   # Individual trajectory files
        ├── test1.txt
        ├── ...
        ├── processed_data_exp4.pkl     # Cached processed data
        └── test/                       # Test data
            ├── test0.txt
            ├── test1.txt
            ├── ...
            └── processed_test_data_exp4.pkl
```

### Multi-Agent Trajectory Format
Each trajectory file contains dual-agent information:
- **Maze representation**: 9x9 grid with both agents and objects
- **Achiever actions**: 7-dimensional action space [up, right, down, left, stay, pickup, toggle]
- **Blocker actions**: 6-dimensional action space [up, right, down, left, stay, broken]
- **Agent positions**: (x, y) coordinates for both achiever and blocker
- **Agent states**: Inventory and goal information
- **Agent labels**: Achiever (0) or Blocker (1) for each observation
- **Strategic outcomes**: Success/failure and blocking effectiveness

## Model Training

### Enhanced Architecture Options
```bash
# Benchmark model (CharNet + PredNet only)
python script/exp4/train.py --use_mentalnet False

# Proposed model (CharNet + MentalNet + PredNet)  
python script/exp4/train.py --use_mentalnet True
```

### Extended Loss Components
- **Action loss**: Cross-entropy for both achiever and blocker actions
- **Goal loss**: Cross-entropy for goal preference prediction
- **Agent loss**: Cross-entropy for agent type classification (NEW)
- **Consumption loss**: Binary cross-entropy for key collection
- **SR loss**: KL divergence for successor representation

### Multi-Agent Training Features
- **Dual-agent samples**: Each trajectory generates samples for both agent types
- **Agent classification**: Neural network learns to distinguish achiever vs blocker behavior
- **Strategic loss weighting**: Configurable emphasis on agent prediction accuracy
- **Competitive dynamics**: Model learns both cooperative and adversarial behaviors

## Evaluation Metrics

### Standard Multi-Agent Metrics
- **Achiever action accuracy**: Action prediction for achiever agents
- **Blocker action accuracy**: Action prediction for blocker agents  
- **Goal inference accuracy**: Goal prediction from observed behavior
- **Agent classification accuracy**: Distinguishing achiever vs blocker (NEW)

### Strategic Analysis
- **Blocking effectiveness**: Success rate of blocker interference
- **Theory of mind accuracy**: How well blockers predict achiever goals
- **Multi-agent confusion matrices**: Detailed error analysis for both agent types
- **Strategic adaptation**: How agents respond to opponent behavior

## Performance Optimizations

### Multi-Agent Data Processing
- **Vectorized SR calculation**: Optimized for dual-agent scenarios
- **Efficient trajectory parsing**: Handles complex multi-agent state representations
- **Enhanced caching**: Separate processed data for different agent combinations
- **Memory optimization**: Efficient storage for expanded state spaces

### Training Optimizations
- **Balanced sampling**: Equal representation of achiever and blocker samples
- **Strategic batch composition**: Ensures diverse agent type combinations
- **Loss balancing**: Automated weighting for stable multi-task learning
- **Early stopping**: Monitors all loss components for optimal convergence

## Troubleshooting

### Common Multi-Agent Issues

1. **Agent imbalance in training**:
   ```bash
   python script/exp4/train.py --agent_weight 2.0 --batch_size 512
   ```

2. **No multi-agent data found**:
   ```bash
   # Generate small dataset for testing
   python script/exp4/generate.py --n_games 100 --achiever_type value --blocker_type random
   ```

3. **Strategic behavior not emerging**:
   ```bash
   # Try different agent combinations
   python script/exp4/generate.py --achiever_type astar --blocker_type goal_direct
   ```

4. **Multi-agent environment errors**:
   ```bash
   # Test environment with simple simulation
   python script/exp4/simulate_game.py --episodes 1
   ```

### Experiment History:
- 20250714_021705 -> multi-agent value vs goaldirected (small parameter)
- 20250714_182233 -> multi-agent value vs goaldirected (large parameter) with 2 GPU
->> 버그 있었음 (generate.py에 blocker의 interaction 조건이 의도와 다름, blocker-door 좌표가 같아야하고, broken(5)를 선택해야 "0", "1"로 분류가 가능한데, 지금은 그렇게 안되어 있었음 (추론 잘 하면 끝)))