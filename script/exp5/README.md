# AchieverBlocker Experiment 5 - Multi-Agent ToMnet Implementation

This directory contains a comprehensive multi-agent Theory of Mind (ToMnet) implementation for the AchieverBlocker environment with strategic interactions between competitive agents.

## Overview

AchieverBlocker Experiment 5 implements a competitive multi-agent environment where:
1. **Achiever agents** navigate a 9x9 grid to collect keys and open corresponding doors
2. **Blocker agents** observe achiever behavior, infer their goals, and strategically block access
3. **Strategic interaction** creates theory of mind requirements for successful blocking

The implementation features an advanced ToMnet architecture with:
- **Flexible architecture support** (2-stage and 3-stage variants)
- **Multi-task learning** for actions, goals, agent types, and consumption prediction
- **Vectorized successor representation** computation for efficient training
- **Dual-agent observations** with comprehensive role differentiation
- **Enhanced data generation** pipeline with parallel processing

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

### 🧠 **Advanced ToMnet Architecture**
- **Flexible Architecture**: Supports both 2-stage (CharNet→PredNet) and 3-stage (CharNet→MentalNet→PredNet) variants
- **Multi-task Learning**: Simultaneous prediction of actions, goals, agent types, consumption patterns, and successor representations
- **Agent Classification Head**: Distinguishes between achiever (0) and blocker (1) agents
- **Type Classification Head**: Differentiates between agent behavior types (Level-0/1 reasoning)
- **Enhanced Loss Functions**: Weighted combination of action, goal, agent, type, consumption, and SR losses
- **Residual Blocks**: Deep residual connections for improved gradient flow
- **ConvLSTM Integration**: Temporal sequence processing with convolutional memory

### 📊 **Advanced Data Processing**
- **Multi-agent trajectory generation**: Comprehensive dual-agent data with parallel processing
- **Agent type labeling**: Each sample tagged with agent role and reasoning level
- **Vectorized SR calculation**: Optimized successor representation computation across multiple discount factors
- **Enhanced data paths**: Combined naming scheme for agent type pairs with efficient caching
- **Trajectory slicing**: Dynamic sequence length handling for efficient training
- **Parallel data generation**: Multiprocessing support for faster dataset creation

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
    "use_mentalnet": True,           # Enable Mental Network (3-stage) or False (2-stage)
    "residual_blocks": 5,            # Number of residual blocks in CharNet
    "n_echar": 128,                  # Character embedding size
    "n_ement": 128,                  # Mental state embedding size
    "out_channels": 64,              # Output channels for convolutional layers
    "channels_in": 9,                # 8 original channels + 1 heading direction channel
    "current_state_channels": 8,     # For MentalNet: 8 original channels (no heading direction)
    "achiever_action_space": 7,      # Achiever actions: up, right, down, left, stay, pickup, toggle
    "blocker_action_space": 6,       # Blocker actions: up, right, down, left, stay, broken
    "goal_space": 4,                 # 4 colored goals: red, green, blue, yellow
    "env_width": self.width,         # Environment width (9)
    "env_height": self.height,       # Environment height (9)
    "hidden_size_lstm": 64,          # LSTM hidden size for ConvLSTM
    "n_past": 5,                     # Number of past episodes for character network
    "n_recent": 20,                  # Number of recent timesteps for mental network
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
    "use_parallel": True,            # Enable DataParallel training
    "use_amp": True,                 # Automatic Mixed Precision for memory efficiency
    "gradient_accumulation_steps": 2, # Gradient accumulation for larger effective batch size
    "pin_memory": True,              # Pin memory for faster data transfer
    "num_workers": 4,                # DataLoader worker processes
    "optimizer": "adam",             # Optimizer type
    "scheduler": "cosine",           # Learning rate scheduler
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
- **Vectorized Value Iteration**: Efficient planning using vectorized operations for optimal pathfinding
- **Stochastic Policy**: Temperature-based action selection with configurable exploration
- **Two-phase Strategy**: Collect preferred key → navigate to corresponding door
- **Automatic Key Pickup**: Keys are automatically collected when stepping on them
- **Cost Functions**: Configurable movement, wall, conflict, and consumption penalties
- **Q-value Clipping**: Prevents numerical instability with bounded Q-values

#### Level1ValueAchiever (lv1va)
- **Sophisticated Deception**: Collects decoy key first to mislead blockers about true preferences
- **Adaptive Strategy**: Monitors blocker movements and switches to target key if blocker moves to any door
- **Dual-phase Planning**: Uses value iteration for both deception and goal achievement phases
- **Strategic Reasoning**: Level-1 reasoning to model and deceive Level-0 blocker opponents
- **Dynamic Target Selection**: Adjusts strategy based on blocker behavior and positioning

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
- **Random Door Selection**: Chooses target door randomly from available options
- **Multi-attempt Strategy**: Tracks tried doors and attempts different ones if wrong
- **Vectorized Value Iteration**: Uses efficient pathfinding to reach selected doors
- **Strategic Positioning**: Moves to predicted target doors and uses "broken" action to block
- **Game Continuation**: Continues playing after wrong door attempts with penalty

#### Level1ValueBlocker (lv1vb)
- **Distance Tracking**: Monitors achiever's distance to different keys over time
- **Early Prediction**: Predicts target door based on movement patterns before key collection
- **Multi-phase Strategy**: Random door → observation → inference → strategic blocking
- **Theory of Mind**: Infers achiever preferences from observed behavior and key collection patterns
- **Adaptive Blocking**: Adjusts strategy based on achiever's deceptive behaviors

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
Each trajectory file contains comprehensive dual-agent information:
- **Maze Representation**: 9x9 grid with both agents, keys, doors, and walls
- **Achiever Actions**: 7-dimensional action space [up=0, right=1, down=2, left=3, stay=4, pickup=5, toggle=6]
- **Blocker Actions**: 6-dimensional action space [up=0, right=1, down=2, left=3, stay=4, broken=5]
- **Agent Positions**: (x, y) coordinates for both achiever and blocker at each timestep
- **Agent States**: Inventory information, collected keys, and goal preferences
- **Agent Labels**: Achiever (0) or Blocker (1) for each observation sample
- **Type Labels**: Behavior type classification (Level-0 vs Level-1 reasoning)
- **Consumption Labels**: 8-dimensional binary vector for key collection patterns
- **Successor Representation**: Vectorized SR data for multiple discount factors (0.5, 0.9, 0.99)
- **Goal Rankings**: Preference rankings for different colored doors
- **Strategic Outcomes**: Success/failure rates and blocking effectiveness metrics
- **Interaction Results**: Detailed analysis of blocker attempts and success rates

## Model Training

### Enhanced Architecture Options
```bash
# Benchmark model (CharNet + PredNet only)
python script/exp5/train.py --use_mentalnet False

# Proposed model (CharNet + MentalNet + PredNet)  
python script/exp5/train.py --use_mentalnet True
```

### Extended Loss Components
- **Action Loss**: Cross-entropy for both achiever (7-class) and blocker (6-class) action prediction
- **Goal Loss**: Cross-entropy for goal preference prediction (4 colored doors)
- **Agent Loss**: Cross-entropy for agent type classification (achiever vs blocker)
- **Type Loss**: Cross-entropy for agent behavior type classification (Level-0 vs Level-1 reasoning)
- **Consumption Loss**: Binary cross-entropy for key collection pattern prediction
- **SR Loss**: KL divergence for successor representation prediction across multiple discount factors
- **Weighted Combination**: Configurable loss weights for balanced multi-task learning
- **Trajectory Slicing**: Dynamic sequence length handling for efficient training

### Multi-Agent Training Features
- **Trajectory Slicing**: Dynamic sequence length handling with multiple samples from each trajectory
- **Agent Classification**: Neural network learns to distinguish achiever vs blocker behavior patterns
- **Type Classification**: Distinguishes between different reasoning levels (Level-0 vs Level-1)
- **Strategic Loss Weighting**: Configurable emphasis on different prediction tasks with automatic balancing
- **Competitive Dynamics**: Model learns both cooperative and adversarial multi-agent behaviors
- **Parallel Training**: Multi-GPU support with DataParallel for faster training
- **Mixed Precision**: Automatic mixed precision (AMP) for memory efficiency and speed
- **Gradient Accumulation**: Larger effective batch sizes through gradient accumulation
- **Early Stopping**: Monitors validation loss with configurable patience and delta thresholds
- **Vectorized Operations**: Efficient batch processing of SR computation and loss calculations

## Evaluation Metrics

### Standard Multi-Agent Metrics
- **Action Accuracy**: Action prediction accuracy for both achiever and blocker agents
- **Goal Inference Accuracy**: Goal prediction accuracy from observed behavior patterns
- **Agent Classification Accuracy**: Distinguishing between achiever and blocker agents
- **Type Classification Accuracy**: Distinguishing between Level-0 and Level-1 reasoning patterns
- **Consumption Prediction Accuracy**: Predicting key collection patterns and resource consumption
- **SR Prediction Accuracy**: Successor representation prediction quality across discount factors
- **Per-Agent Metrics**: Separate evaluation for each agent type with detailed breakdowns

### Strategic Analysis
- **Blocking Effectiveness**: Success rate of blocker interference and strategic positioning
- **Theory of Mind Accuracy**: How well model predicts achiever goals from blocker perspective
- **Multi-agent Confusion Matrices**: Detailed error analysis for both agent types
- **Strategic Adaptation**: How agents respond to opponent behavior and strategy changes
- **N_past Analysis**: Performance evaluation across different numbers of past episodes
- **Character Embedding Analysis**: PCA and t-SNE visualization of learned agent representations
- **Training Dynamics**: Analysis of loss component convergence and learning curves

## Performance Optimizations

### Multi-Agent Data Processing
- **Vectorized SR Calculation**: Optimized numpy operations for dual-agent scenarios with parallel processing
- **Efficient Trajectory Parsing**: Handles complex multi-agent state representations with minimal memory overhead
- **Enhanced Caching**: Separate processed data files for different agent combinations with pickle serialization
- **Memory Optimization**: Efficient storage for expanded state spaces using sparse representations
- **Parallel Data Generation**: Multiprocessing support for faster dataset creation
- **Batch Processing**: Vectorized operations for processing multiple trajectories simultaneously

### Training Optimizations
- **Balanced Sampling**: Equal representation of achiever and blocker samples in training batches
- **Strategic Batch Composition**: Ensures diverse agent type combinations and reasoning levels
- **Loss Balancing**: Automated weighting for stable multi-task learning with configurable coefficients
- **Early Stopping**: Monitors all loss components for optimal convergence with patience control
- **Gradient Clipping**: Prevents gradient explosion with configurable maximum norm
- **Learning Rate Scheduling**: Cosine annealing and step decay for optimal convergence
- **Memory Efficient Loading**: DataLoader optimizations with pin_memory and num_workers

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

## Recent Updates and Improvements

### Architecture Enhancements
- **Dual Architecture Support**: Added support for both 2-stage and 3-stage ToMnet variants
- **Enhanced Model Components**: Improved CharNet with residual blocks and ConvLSTM integration
- **Multi-task Learning**: Extended to predict actions, goals, agent types, consumption, and SR simultaneously
- **Trajectory Slicing**: Dynamic sequence length handling for more efficient training
- **Vectorized Operations**: Optimized SR computation and loss calculations for better performance

### Agent Implementation Updates
- **Value-based Agents**: Implemented Level-0 and Level-1 value agents with sophisticated strategies
- **Deception Mechanisms**: Level-1 achiever agents use decoy key collection to mislead blockers
- **Theory of Mind**: Level-1 blocker agents track achiever behavior patterns for better prediction
- **Multi-attempt Strategy**: Blockers can attempt multiple doors with proper tracking and penalties
- **Vectorized Planning**: Efficient value iteration implementation for all value-based agents

### Data Processing Improvements
- **Parallel Data Generation**: Multiprocessing support for faster dataset creation
- **Enhanced Caching**: Separate processed data files for different agent combinations
- **Comprehensive Labeling**: Rich trajectory data with SR, consumption, and interaction labels
- **Memory Optimization**: Efficient storage using sparse representations and vectorized operations
- **Batch Processing**: Vectorized operations for processing multiple trajectories simultaneously

### Training and Evaluation Features
- **Mixed Precision Training**: Automatic mixed precision for memory efficiency and speed
- **Multi-GPU Support**: DataParallel training with gradient accumulation
- **Advanced Metrics**: Comprehensive evaluation including N_past analysis and embedding visualization
- **Early Stopping**: Monitors all loss components with configurable patience
- **Learning Rate Scheduling**: Cosine annealing and step decay for optimal convergence

## exp5 Version History
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
