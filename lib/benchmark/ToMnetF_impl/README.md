# ToMnetF Implementation

This directory contains the refactored ToMnetF (Theory of Mind Network with CNN Features) implementation, organized following the ToMnet_impl structure for better maintainability and extensibility.

## Overview

ToMnetF is a CNN-based implementation of Theory of Mind Networks that learns to predict agent behavior in gridworld environments. This implementation uses ResNet blocks and LSTM networks to process trajectory data and predict actions.

## Directory Structure

```
ToMnetF_impl/
├── scripts/
│   ├── environment.py              # LabMaze GridWorld environment
│   ├── data_generation.py          # Agnostic data processing and loading utilities
│   └── experiment1/                # Experiment-specific implementations
│       ├── config.py               # Environment configuration
│       ├── agents.py               # A* and Random agents
│       ├── tomnet.py               # ToMnetF CNN architecture
│       ├── generate.py             # Experiment-specific trajectory generation
│       ├── train.py                # Advanced training system
│       ├── evaluate.py             # Cross-species evaluation and metrics
│       └── visualize.py            # Publication-quality visualization
├── shell/
│   └── run_exp1.sh                 # Complete workflow automation
├── data/experiment1/               # Training data
├── models/experiment1/             # Trained models
├── result/experiment1/             # Results and evaluations
├── plots/experiment1/              # Generated plots
└── log/                           # Execution logs
```

## Architecture

### ToMnetF Model Components

1. **CharNet**: Character network that processes trajectory sequences
   - Time-distributed CNN layers
   - Residual blocks for deep feature extraction
   - LSTM for temporal modeling
   - Outputs character embeddings

2. **PredNet**: Prediction network that combines character embeddings with current state
   - CNN layers for spatial processing
   - Residual blocks
   - Fully connected layers for action prediction

### Key Features

- **CNN-based Architecture**: Uses convolutional layers for spatial processing
- **Residual Connections**: Deep residual blocks for improved training
- **Time-distributed Processing**: Handles variable-length trajectory sequences
- **Character Embeddings**: Learns representations of agent behavior patterns

## Usage

### Quick Start

```bash
# Run complete pipeline
bash shell/run_exp1.sh all

# Or run individual components
bash shell/run_exp1.sh data_generation
bash shell/run_exp1.sh train
bash shell/run_exp1.sh evaluate
bash shell/run_exp1.sh visualize
```

### Manual Usage

#### 1. Data Generation
```bash
cd scripts/experiment1
python generate.py --n_games 10000 --output_dir experiment1 --observability full --max_moves 50
```

#### 2. Data Preprocessing
```bash
cd scripts
python data_generation.py --data_dir ../data/experiment1 --use_percentage 0.9 --experiment_no 1
```

#### 3. Training
```bash
cd scripts/experiment1
python train.py --experiment_no 1 --epochs 50 --batch_size 512 --device cuda:0 --lr 1e-4
```

#### 4. Evaluation
```bash
cd scripts/experiment1
python evaluate.py --model_paths ../../models/experiment1/exp1_best.pth \
                   --test_data_paths ../../data/experiment1/processed_data_exp1.pkl \
                   --experiment_no 1 --device cuda:0
```

#### 5. Visualization
```bash
cd scripts/experiment1
python visualize.py --experiment_no 1 --plot_type all
```

### Python API Usage

#### 1. Data Generation
```python
from scripts.experiment1.generate import generate_trajectories

generate_trajectories(
    n_games=10000,
    output_dir="experiment1",
    observability="full"
)
```

#### 2. Data Preprocessing
```python
from scripts.data_generation import generate_input_data

processed_data = generate_input_data(
    data_dir="../data/experiment1",
    use_percentage=0.9
)
```

#### 3. Training
```python
from scripts.experiment1.train import train_tomnet

model, history, results = train_tomnet(
    experiment_no=1,
    epochs=50,
    batch_size=512,
    device="cuda:0"
)
```

#### 4. Evaluation
```python
from scripts.experiment1.evaluate import cross_species_evaluation

results = cross_species_evaluation(
    model_paths=["../../models/experiment1/exp1_best.pth"],
    test_data_paths=["../../data/experiment1/processed_data_exp1.pkl"],
    experiment_no=1
)
```

#### 5. Visualization
```python
from scripts.experiment1.visualize import create_summary_report

create_summary_report(experiment_no=1)
```

## Configuration

### Model Parameters

- **Batch Size**: 512 (default)
- **Learning Rate**: 1e-4
- **Epochs**: 50
- **Trajectory Size**: 10 time steps
- **Grid Size**: 13x13
- **Input Channels**: 10 (1 wall + 1 player + 4 goals + 4 actions)
- **Residual Blocks**: 5
- **Character Embedding Size**: 8
- **Output Channels**: 32

### Environment Parameters

- **Grid Size**: 13x13
- **Max Moves**: 50 per episode
- **Sight Radius**: 3 (for partial observability)
- **Goals**: A, B, C, D with rewards [2, 4, 8, 16]
- **Actions**: UP (0), RIGHT (1), DOWN (2), LEFT (3)

## Experiment Types

### Experiment 1 (Current Implementation)

- **Agent Type**: A* optimal pathfinding agent
- **Observability**: Full observability
- **Goal Strategy**: Highest value goal selection
- **Architecture**: CNN-based ToMnetF with ResNet blocks

### Future Experiments

The structure supports easy addition of new experiments by:
1. Creating new experiment directories under `scripts/`
2. Implementing experiment-specific agents and configurations
3. Adding corresponding data and model directories

## Output Files

### Models
- `exp1_best.pth`: Best model based on validation accuracy
- `exp1_final.pth`: Final model after all epochs

### Results
- `exp1_training_history.json`: Training curves and metrics
- `exp1_results.json`: Final training results and configuration
- `cross_species_evaluation_exp1.json`: Cross-species evaluation results
- `predictions.pkl`: Model predictions and probabilities

### Plots
- `training_curves_exp1.png`: Training and validation curves
- `confusion_matrix_exp1.png`: Action prediction confusion matrix
- `action_likelihood_exp1.png`: Likelihood distributions by action
- `character_embeddings_exp1.png`: Character embedding visualizations

## Dependencies

```
torch>=1.9.0
torchvision
numpy
matplotlib
seaborn
scikit-learn
pandas
labmaze
```

## Installation

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas

# Install labmaze (for environment)
pip install dm-labmaze
```

## Differences from Original ToMnet

### Architecture Changes

1. **CNN-based Processing**: Uses convolutional layers instead of MLPs
2. **Residual Connections**: Deep residual blocks for improved gradient flow
3. **Time-distributed Layers**: Explicit handling of temporal sequences
4. **Character Embeddings**: Learned representations of agent behavior

### Implementation Improvements

1. **Modular Design**: Experiment-specific organization
2. **Comprehensive Logging**: Detailed training and evaluation logs
3. **Advanced Visualization**: Publication-quality plots and analysis
4. **Cross-species Evaluation**: Systematic evaluation framework
5. **Automated Pipeline**: Shell scripts for complete workflow

## Research Applications

This implementation is suitable for:

- Theory of Mind research in artificial agents
- Multi-agent behavior prediction
- Transfer learning across different agent types
- Representation learning for sequential decision making
- Gridworld navigation and planning studies

## Citation

If you use this implementation, please cite the original ToMnet paper and acknowledge this implementation:

```bibtex
@article{rabinowitz2018machine,
  title={Machine theory of mind},
  author={Rabinowitz, Neil and Perbet, Frank and Song, Francis and Zhang, Chiyuan and Eslami, SM Ali and Botvinick, Matthew},
  journal={International conference on machine learning},
  year={2018}
}
```

## Experiemnt history

1. ** Exp1 **: ToMnet + A* star agent (only action prediction)
2. ** Exp2 **: ToMnet + A* star agent (action prediction, SR, consumption) -> 결과는 더 안정적으로 나옴
3. ** Exp3 **: ToMnet + A* star agent (action prediction, SR, consumption)
    - SR label is computed by each step.
    - N_past is not implemented in Exp2 -> have to fix it
    - Consumption label is computed by each step.
