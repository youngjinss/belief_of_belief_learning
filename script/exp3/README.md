# KeyDoor Experiment 3 - ToMnet Implementation

This directory contains a complete ToMnet implementation for the KeyDoor environment with hierarchical belief modeling and trajectory data generation.

## Overview

KeyDoor Experiment 3 implements a multi-colored key-door environment where agents must:
1. Navigate a 9x9 grid with 4 colored keys and 4 colored doors
2. Collect keys and open corresponding doors
3. Demonstrate strategic reasoning about goal preferences and costs

The implementation follows the ToMnet architecture for theory of mind modeling with:
- **Character embeddings** from past episodes
- **Mental state inference** for goal prediction
- **Action prediction** based on inferred mental states
- **Successor representation** for spatial reasoning

## Project Structure

```
script/exp3/
├── config.py              # Configuration class with all parameters
├── generate.py            # Trajectory data generation
├── train.py               # ToMnet model training
├── evaluate.py            # Model evaluation and metrics
├── visualize.py           # Result visualization and plotting
├── visualize_sr.py        # Successor representation visualization
├── tomnet.py              # ToMnet model architecture
├── agents.py              # Agent implementations (A*, Value, Random)
├── data_generation.py     # Data processing utilities
├── simulate_game.py       # Single game simulation with GUI
├── simulate_trajectory.py # Trajectory simulation and visualization
└── backup/                # Backup files for testing
    ├── simple_test.py
    ├── test_keydoor.py
    └── visualize_gotodoor.py
```

## Key Features

### 🎯 **Multi-Agent Environment**
- **4 colored keys** (red, green, blue, yellow)
- **4 colored doors** with matching key requirements
- **Variable goal preferences** with ToMnetF-style reward generation
- **Variable door costs** for strategic decision making

### 🧠 **ToMnet Architecture**
- **Character Network**: Encodes past episodes for character understanding
- **Mental Network**: Infers goals from character and current state (optional)
- **Prediction Network**: Predicts actions based on mental states
- **Multi-task Learning**: Actions, goals, consumption labels, and successor representation

### 📊 **Data Processing**
- **Trajectory slicing** for varied episode lengths
- **Processed data caching** for performance optimization
- **Automatic data path management** based on environment and agent type
- **Goal ranking** for character matching

### 🚀 **Training & Evaluation**
- **Early stopping** with validation monitoring
- **Model checkpointing** with best model saving
- **Comprehensive metrics** (accuracy, precision, recall, F1)
- **N_past evaluation** for character embedding analysis
- **Character embedding visualization**

## Quick Start

### 1. Data Generation
```bash
# Generate training data (100k games by default)
python script/exp3/generate.py

# Generate test data with custom parameters
python script/exp3/generate.py --n_games 2000 --random_seed 123 --test_data

# Different agent types
python script/exp3/generate.py --agent_type astar
python script/exp3/generate.py --agent_type random
python script/exp3/generate.py --agent_type value
```

### 2. Game Simulation & Visualization
```bash
# Run single game with visualization
python script/exp3/simulate_game.py --episodes 1 --render --agent_type value

# Simulate trajectory and save visualization
python script/exp3/simulate_trajectory.py --episodes 5 --gif_output keydoor_demo

# Visualize successor representation
python script/exp3/visualize_sr.py --data_file data/MiniGrid-KeyDoor-9x9-v0/value/test0.txt
```

### 3. Model Training
```bash
# Train with default config (value agent, 9x9 environment)
python script/exp3/train.py --save_dir ./results/exp3/

# Custom training with specific parameters
python script/exp3/train.py \
    --epochs 100 --batch_size 1024 --lr 0.0001 \
    --device cuda:0 --save_dir ./results/exp3/
```

### 4. Model Evaluation  
```bash
# Evaluate trained model
python script/exp3/evaluate.py \
    --model_path ./results/exp3/best_model.pth \
    --result_dir ./results/exp3/

# Generate visualizations
python script/exp3/visualize.py \
    --result_dir ./results/exp3/ \
    --plot_type all
```

## Configuration

All parameters are centralized in `config.py`:

### Environment Settings
```python
self.env_name = "MiniGrid-KeyDoor-{size}-v0"
self.agent_type = "value"  # "astar", "random", "value"
self.width = 9
self.height = 9
self.max_steps = 500
```

### Model Architecture
```python
self.model_config = {
    "use_mentalnet": True,          # Enable Mental Network
    "residual_blocks": 5,
    "n_echar": 128,                 # Character embedding size
    "n_ement": 128,                 # Mental state embedding size
    "action_space": 7,              # KeyDoor actions
    "goal_space": 4,                # 4 colored goals
}
```

### Training Configuration
```python
self.training_config = {
    "batch_size": 1024,
    "epochs": 200,
    "lr": 0.0001,
    "device": "cuda:3",
    "early_stopping_patience": 30,
}
```

## Data Structure

### Generated Data Paths
```
data/
└── MiniGrid-KeyDoor-9x9-v0/
    └── value/                      # Training data
        ├── test0.txt               # Individual trajectory files
        ├── test1.txt
        ├── ...
        ├── processed_data_exp3.pkl # Cached processed data
        └── test/                   # Test data
            ├── test0.txt
            ├── test1.txt
            ├── ...
            └── processed_test_data_exp3.pkl
```

### Trajectory File Format
Each trajectory file contains:
- **Maze representation**: 9x9 grid with object encoding (walls, keys, doors, agent)
- **Actions**: 7-dimensional action space [up, right, down, left, stay, pickup, toggle]
- **Agent positions**: (x, y) coordinates and heading direction
- **Inventory state**: Key collection status
- **Goal information**: Goal preferences and rankings for each color
- **Rewards**: Dynamic goal-based rewards
- **Successor representation**: Multi-gamma spatial value maps

## Model Training

### Architecture Options
```bash
# Benchmark model (CharNet + PredNet only)
python script/exp3/train.py --use_mentalnet False

# Proposed model (CharNet + MentalNet + PredNet)  
python script/exp3/train.py --use_mentalnet True
```

### Loss Components
- **Action loss**: Cross-entropy for action prediction
- **Goal loss**: Cross-entropy for goal prediction  
- **Consumption loss**: Binary cross-entropy for key collection
- **SR loss**: KL divergence for successor representation

### Training Features
- **Processed data caching**: Avoids reprocessing on repeated runs
- **Dynamic batching**: Handles variable trajectory lengths
- **Early stopping**: Prevents overfitting
- **Model checkpointing**: Saves best model automatically
- **Comprehensive logging**: Training history and metrics

## Evaluation Metrics

### Standard Metrics
- **Action accuracy**: Overall action prediction accuracy
- **Goal accuracy**: Goal inference accuracy
- **Action-wise accuracy**: Per-action performance breakdown
- **Confusion matrices**: Detailed error analysis

### Advanced Analysis
- **N_past evaluation**: Character embedding effectiveness
- **Character embeddings**: t-SNE visualization of character spaces
- **Action likelihood analysis**: Probability distribution analysis

## Agents

### Value Agent (Default)
- **Value iteration** with configurable parameters
- **Stochastic policy** with temperature control
- **Two-phase strategy**: key collection → door opening
- **Optimal for strategic reasoning experiments**

### A* Agent
- **Optimal pathfinding** with A* algorithm
- **Deterministic behavior** 
- **Turn-based navigation** for MiniGrid compatibility
- **Baseline for performance comparison**

### Random Agent
- **Exploration baseline** with movement bias
- **80% movement actions**, 20% interaction actions
- **No strategic reasoning**

## Performance Features

### Data Processing Optimization
- **Processed data caching**: Pickle files for repeated runs
- **Automatic existence checking**: Skips regeneration if data exists
- **Memory efficient**: Compressed data storage
- **Fast loading**: Direct tensor loading from cache

### Training Optimization
- **GPU acceleration**: CUDA support with device selection
- **Batch processing**: Efficient parallel data loading
- **Early stopping**: Automatic training termination
- **Model checkpointing**: Best model preservation

### Evaluation Optimization
- **Cached test data**: Avoids reprocessing test data
- **Vectorized metrics**: Fast batch evaluation
- **Memory management**: Efficient large dataset handling

## Simulation and Visualization Features

### Interactive Game Simulation
The `simulate_game.py` script provides real-time game visualization:
- **GUI rendering**: Watch agent behavior in real-time
- **Multiple agent types**: Test different strategies (A*, Value, Random)
- **GIF generation**: Save gameplay as animated GIFs
- **Episode configuration**: Control number of episodes and visualization settings

### Trajectory Analysis
The `simulate_trajectory.py` script offers detailed trajectory analysis:
- **Maze visualization**: Convert environment to text-based maze format
- **Action logging**: Track all agent decisions and movements
- **State transitions**: Analyze environment changes over time

### Successor Representation Visualization
The `visualize_sr.py` script displays spatial reasoning patterns:
- **Multi-gamma SR maps**: Visualize different time horizons
- **Overlay on maze**: Show SR values in context of environment layout
- **Heatmap visualization**: Color-coded representation of spatial values

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   ```bash
   python script/exp3/train.py --batch_size 512 --device cuda:0
   ```

2. **No data found**:
   ```bash
   # Generate a small dataset for testing
   python script/exp3/generate.py --n_games 100
   ```

3. **Environment import errors**:
   ```bash
   # Test environment installation
   python script/exp3/simulate_game.py --episodes 1
   ```

4. **Processed data corruption**:
   ```bash
   # Remove cached files to regenerate
   rm data/MiniGrid-KeyDoor-9x9-v0/value/processed_data_exp3.pkl
   rm data/MiniGrid-KeyDoor-9x9-v0/value/test/processed_test_data_exp3.pkl
   ```

### Performance Tips

1. **Use processed data caching**: Let the system cache processed data for faster repeated runs
2. **Monitor GPU memory**: Adjust batch size based on available VRAM
3. **Use early stopping**: Set appropriate patience for your dataset size
4. **Parallel data generation**: Uses all CPU cores by default

## Results

Training produces:
- **Model checkpoints**: `best_model.pth` in results directory
- **Training history**: JSON files with loss curves  
- **Evaluation metrics**: Comprehensive performance analysis
- **Visualizations**: Character embeddings and training curves

Check `results/exp3/[timestamp]/` for all outputs.

## Dependencies

Ensure you have:
- PyTorch with CUDA support
- NumPy, matplotlib
- scikit-learn for metrics
- Custom MiniGrid environment (`lib/env/gym_minigrid/`)

## Citation

This implementation extends ToMnet architecture for multi-agent belief modeling in discrete navigation environments.