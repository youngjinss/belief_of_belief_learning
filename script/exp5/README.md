# AchieverBlocker Experiment 5 - Multi-Agent ToMnet Implementation

This directory contains a complete multi-agent ToMnet implementation for the AchieverBlocker environment with strategic interaction between two agent types.

## Overview

AchieverBlocker Experiment 5 implements a competitive multi-agent environment where:
1. **Achiever agents** navigate a 9x9 grid to collect keys and open corresponding doors
2. **Blocker agents** observe achiever behavior, infer their goals, and strategically block access
3. **Strategic interaction** creates theory of mind requirements for successful blocking

The implementation uses a ToMnet architecture for multi-agent scenarios with:
- **Dual-agent observations** with role differentiation
- **Agent type prediction** as an additional learning task
- **Strategic blocking behavior** with competitive dynamics
- **Enhanced data generation** for multi-agent trajectories

## Project Structure

```
script/exp5/
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
- **Level0ValueAchiever (lv0va)**: Level-0 value-based agent using value iteration for direct pathfinding
- **Level1ValueAchiever (lv1va)**: Level-1 value-based agent with deception strategies (collects decoy key first)
- **AStarAgent (astar)**: Optimal pathfinding with A* algorithm
- **ValueAgent**: Strategic navigation with preference optimization
- **RandomAgent**: Baseline exploration behavior

#### Blocker Agents
- **Level0ValueBlocker (lv0vb)**: Level-0 value-based blocker with random door selection and value iteration
- **Level1ValueBlocker (lv1vb)**: Level-1 value-based blocker with distance tracking and early target prediction
- **RandomlySelectedAgent**: Level-0 reasoning - randomly selects target door and blocks it
- **RuleBasedAgent**: Level-1 reasoning - first blocks random door, then infers target from achiever's key
- **GoalDirectAgent**: Infers achiever goals from observed key collection patterns
- **RandomAgent**: Baseline blocking behavior

## Quick Start

### 1. Multi-Agent Data Generation
```bash
# Generate training data with level-0 value achiever and level-0 value blocker
python script/exp5/generate.py --achiever_type lv0va --blocker_type lv0vb

# Generate test data with different agent combinations
python script/exp5/generate.py --n_games 2000 --achiever_type lv1va --blocker_type lv1vb --test_data

# Generate data for all agent combinations
python script/exp5/generate.py --achiever_type lv0va --blocker_type lv0vb
python script/exp5/generate.py --achiever_type lv1va --blocker_type lv1vb
python script/exp5/generate.py --achiever_type astar --blocker_type randomly_selected
python script/exp5/generate.py --achiever_type value --blocker_type rule_based
```

### 2. Multi-Agent Game Simulation
```bash
# Run competitive simulation with visualization
python script/exp5/simulate_game.py --episodes 1 --render --achiever_type lv0va --blocker_type lv0vb

# Save gameplay as animated GIF
python script/exp5/simulate_trajectory.py --episodes 5 --gif_output achiever_blocker_demo

# Visualize multi-agent successor representation
python script/exp5/visualize_sr.py --data_file data/MiniGrid-AchieverBlocker-9x9-v1/lv0va_lv0vb/test0.txt
```

### 3. Enhanced Model Training
```bash
# Train with multi-agent data (default: lv0va achiever + lv0vb blocker)
python script/exp5/train.py --save_dir ./results/exp5/

# Custom training with agent type weighting
python script/exp5/train.py \
    --epochs 300 --batch_size 750 --lr 0.0001 \
    --agent_weight 0.1 --device cuda:3 --save_dir ./results/exp5/
```

### 4. Multi-Agent Evaluation
```bash
# Evaluate multi-agent model performance
python script/exp5/evaluate.py \
    --model_path ./results/exp5/best_model.pth \
    --result_dir ./results/exp5/

# Generate multi-agent visualizations
python script/exp5/visualize.py \
    --result_dir ./results/exp5/ \
    --plot_type all
```

## Configuration

All parameters are centralized in `config.py` for multi-agent experiments:

### Environment Settings
```python
self.env_name = "MiniGrid-AchieverBlocker-{size}-v1"
self.achiever_types = {
    "lv0va": self.n_games_per_type,
    "lv1va": self.n_games_per_type,
}  # Options: "lv0va", "lv1va", "astar", "random", "value"
self.blocker_types = {
    "lv0vb": self.n_games_per_type,
    "lv1vb": self.n_games_per_type,
}  # Options: "lv0vb", "lv1vb", "random", "goal_direct", "randomly_selected", "rule_based"
self.width = 9
self.height = 9
self.max_steps = 50
```

### Model Architecture
```python
self.model_config = {
    "use_mentalnet": True,           # Enable Mental Network
    "residual_blocks": 5,
    "n_echar": 128,                  # Character embedding size
    "n_ement": 128,                  # Mental state embedding size
    "out_channels": 64,
    "channels_in": 9,                # 8 original channels + 1 heading direction channel
    "current_state_channels": 8,     # For MentalNet: 8 original channels (no heading direction)
    "achiever_action_space": 7,      # Achiever actions
    "blocker_action_space": 6,       # Blocker actions
    "goal_space": 4,                 # 4 colored goals
    "env_width": self.width,
    "env_height": self.height,
    "hidden_size_lstm": 64,
}
```

### Training Configuration
```python
self.training_config = {
    "batch_size": 750,
    "epochs": 300,
    "lr": 0.0001,
    "weight_decay": 0.001,
    "training_proportion": 0.9,
    "device": "cuda:3",
    "device_ids": [3, 2],            # GPU IDs for parallel training
    "use_parallel": True,            # Enable parallel GPU training
    "use_amp": True,                 # Automatic Mixed Precision
    "gradient_accumulation_steps": 2,
    "pin_memory": True,
    "num_workers": 4,
    "optimizer": "adam",
}

# Training process configuration
self.training_process_config = {
    "early_stopping_patience": 30,
    "early_stopping_min_delta": 0.001,
    "max_grad_norm": 1.0,
    "action_weight": 0.25,
    "goal_weight": 0.25,
    "agent_weight": 0.1,             # Weight for agent classification loss
    "type_weight": 0.1,
    "consumption_weight": 0.15,
    "sr_weight": 0.15,
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

#### Level0ValueAchiever (lv0va)
- **Direct approach**: Uses value iteration for optimal pathfinding to target door
- **Stochastic policy**: Temperature-based action selection
- **Two-phase strategy**: Collect key → navigate to door
- **Automatic key pickup**: Keys are picked up when stepping on them

#### Level1ValueAchiever (lv1va)  
- **Deceptive strategy**: Collects decoy key first, then target key
- **Blocker observation**: Switches to target key if blocker moves to any door
- **Value iteration**: Uses optimal pathfinding for both phases
- **Strategic reasoning**: Level-1 reasoning to deceive blockers

#### AStarAgent (astar)
- **Optimal pathfinding**: Finds shortest path using A* algorithm
- **Deterministic behavior**: Predictable for blocker agents to analyze
- **Two-phase strategy**: Collect key → navigate to door

#### ValueAgent
- **Strategic planning**: Uses value iteration with goal preferences
- **Preference optimization**: Maximizes reward based on colored door preferences

#### RandomAgent
- **Exploration baseline**: Random movement with goal bias
- **Unpredictable behavior**: Difficult for blockers to predict

### Blocker Agents (`blockers.py`)

#### Level0ValueBlocker (lv0vb)
- **Random door selection**: Chooses target door randomly
- **Multi-attempt strategy**: Tries different doors if wrong
- **Value iteration**: Uses optimal pathfinding to selected door
- **Strategic positioning**: Moves to predicted target doors

#### Level1ValueBlocker (lv1vb)
- **Distance tracking**: Monitors achiever's distance to keys over time
- **Early prediction**: Predicts target based on movement patterns
- **Multi-phase strategy**: Random door → wait → infer target → block
- **Theory of mind**: Infers achiever preferences from behavior

#### RandomlySelectedAgent
- **Level-0 reasoning**: Randomly selects target door and blocks it
- **Multi-attempt tracking**: Tracks tried doors and attempts others if wrong
- **BFS pathfinding**: Uses breadth-first search for navigation

#### RuleBasedAgent
- **Level-1 reasoning**: First blocks random door, then infers target from achiever's key
- **Key observation**: Stores observed keys from achiever
- **Multi-attempt strategy**: Cycles through observed keys if wrong

#### GoalDirectAgent
- **Theory of mind**: Infers achiever preferences from key collection behavior
- **Wait strategy**: Waits until achiever picks up first key
- **Strategic positioning**: Moves to predicted target doors

#### RandomAgent
- **Baseline behavior**: Random actions across the environment
- **No strategic reasoning**: Provides comparison baseline

## Data Structure

### Generated Data Paths
```
data/
└── MiniGrid-AchieverBlocker-9x9-v1/
    └── lv0va_lv0vb/                    # Training data for agent pair
        ├── test0.txt                   # Individual trajectory files
        ├── test1.txt
        ├── ...
        ├── processed_data_exp5_lv0va_lv0vb.pkl     # Cached processed data
        └── test/                       # Test data
            ├── test0.txt
            ├── test1.txt
            ├── ...
            └── processed_test_data_exp5_lv0va_lv0vb.pkl
```

### Multi-Agent Trajectory Format
Each trajectory file contains dual-agent information:
- **Maze representation**: 9x9 grid with both agents and objects
- **Achiever actions**: 7-dimensional action space [up, right, down, left, stay, pickup, toggle]
- **Blocker actions**: 6-dimensional action space [up, right, down, left, stay, broken]
- **Agent positions**: (x, y) coordinates for both achiever and blocker
- **Agent states**: Inventory and goal information
- **Agent labels**: Achiever (0) or Blocker (1) for each observation
- **Type labels**: Agent type (0 for randomly select/achiever, 1 for rule-based blocker)
- **Consumption labels**: 8-dimensional binary vector for resource consumption
- **Successor representation**: SR data for multiple discount factors (0.5, 0.9, 0.99)
- **Strategic outcomes**: Success/failure and blocking effectiveness

## Model Training

### Enhanced Architecture Options
```bash
# Benchmark model (CharNet + PredNet only)
python script/exp5/train.py --use_mentalnet False

# Proposed model (CharNet + MentalNet + PredNet)  
python script/exp5/train.py --use_mentalnet True
```

### Extended Loss Components
- **Action loss**: Cross-entropy for both achiever and blocker actions
- **Goal loss**: Cross-entropy for goal preference prediction
- **Agent loss**: Cross-entropy for agent type classification (achiever vs blocker)
- **Type loss**: Cross-entropy for agent behavior type classification (randomly select vs rule-based)
- **Consumption loss**: Binary cross-entropy for key collection
- **SR loss**: KL divergence for successor representation

### Multi-Agent Training Features
- **Trajectory slicing**: Multiple samples generated from different time steps of same trajectory
- **Agent classification**: Neural network learns to distinguish achiever vs blocker behavior
- **Type classification**: Neural network learns to distinguish between agent behavior types
- **Strategic loss weighting**: Configurable emphasis on different prediction tasks
- **Competitive dynamics**: Model learns both cooperative and adversarial behaviors
- **Parallel training**: Multi-GPU support with data parallelization
- **Mixed precision**: Automatic mixed precision for memory efficiency

## Evaluation Metrics

### Standard Multi-Agent Metrics
- **Action accuracy**: Action prediction for both agent types
- **Goal inference accuracy**: Goal prediction from observed behavior
- **Agent classification accuracy**: Distinguishing achiever vs blocker
- **Type classification accuracy**: Distinguishing between agent behavior types
- **Consumption prediction accuracy**: Predicting resource consumption patterns
- **SR prediction accuracy**: Successor representation prediction quality

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
   python script/exp5/train.py --agent_weight 2.0 --batch_size 512
   ```

2. **No multi-agent data found**:
   ```bash
   # Generate small dataset for testing
   python script/exp5/generate.py --n_games 100 --achiever_type lv0va --blocker_type lv0vb
   ```

3. **Strategic behavior not emerging**:
   ```bash
   # Try different agent combinations
   python script/exp5/generate.py --achiever_type lv1va --blocker_type lv1vb
   ```

4. **Multi-agent environment errors**:
   ```bash
   # Test environment with simple simulation
   python script/exp5/simulate_game.py --episodes 1
   ```

## exp5 버전 기록
- Multi-blocker type AchieverBlocker environment with multi-attempt game mechanics

### Agent Rule Fixes (exp5) - fixed
**RandomlySelectedAgent (Level-0 Reasoning)**:
- Fixed multi-attempt tracking system to allow breaking multiple wrong doors before finding target
- Added `tried_doors` set to track attempted doors across multiple break attempts
- Implemented proper reset mechanism after each wrong door break attempt
- Game continues after breaking wrong doors (with -1 penalty) until correct door is found

**RuleBasedAgent (Level-1 Reasoning)**:
- **Critical Bug Fix**: Fixed key observation logic in `_check_and_store_achiever_keys`
- Previous bug: stored array indices instead of color names in `observed_keys`
- Fixed to store actual color names ("red", "green", "blue", "yellow")
- Implemented multi-attempt strategy using observed key sequence
- Added proper cycling through `observed_keys` when wrong doors are broken

**Environment Changes**:
- Modified game termination rules: only ends when achiever reaches target OR blocker breaks actual target door
- Breaking wrong doors gives -1 reward but game continues
- Supports multiple break attempts per trajectory

**Data Generation**:
- Multi-blocker type support (Type 0: randomly_selected, Type 1: rule_based)
- Even distribution across blocker types (50/50 split)
- Trajectory-based interaction analysis processing all break attempts
- Updated interaction logic: "1" (success), "0" (attempted wrong door), "X" (no attempt)
