# ToMnet Implementation for Reproduction

## Overview
This implementation reproduces the qualitative results of the "Machine Theory of Mind" paper (Rabinowitz et al., 2018). The focus is on demonstrating ToMnet's ability to infer character traits of random agents through observation of their behavioral trajectories.

The implementation provides a complete, production-ready system with enterprise-level automation, comprehensive evaluation capabilities, and publication-quality visualization tools.

## ToMnet Architecture for Figure 3 (from Appendix A.2)

**Data Configuration**:
- Variable number of past episodes: $ N_{past} \sim U{0, 10}$
- Each trajectory consists of a **single state-action pair** (length 1)
- When $ N_{past} = 0$, character embedding is set to $ e_{char} = 0 $

### Core Mathematical Framework
The ToMnet implementation for Figure 3 uses a simplified architecture focused on character inference:

1. **Character Net ($f_\theta$)**: 
   - Input: Single state/action pair per trajectory
   - Output: $ e_{char, i} \in R^2 $
      - Processes past episode trajectories $\{\tau_{ij}\}$ into character embeddings
   - Function: $e_{char,ij} = f_\theta(\tau_{ij}^{obs})$
   - Aggregation: $e_{char,i} = \sum_j e_{char,ij}$
   - Implementation: 3-layer MLP with ReLU activations
   - Output: 2D embeddings for visualization
   - Processing pipeline:
        1. Spatialize action and concatenate with state
        2. 1-layer convnet with 8 feature planes and ReLU
        3. Convolutional LSTM processing sequence indexed by j
        4. Average pooling
        5. Fully-connected layer to 2D embedding space

2. **Mental State Net**:
   - **Omitted in Figure 3 experiments** as specified in the paper
   - This simplification focuses on character-level inference only

3. **Prediction Net**:
   - Input: Spatialized $ e_{char,i} $ concatenated with query state
   - Outputs action probabilities: $\hat{\pi}(a_t \mid x_t^{obs}, e_{char})$
   - Function: concatenates current state with character embedding
   - Implementation: 2-layer MLP with softmax output for 5 actions
       - 2-layer convnet with 32 feature planes and ReLUs
       - Average pooling
       - Fully-connected layer to 5-dim logits
       - Softmax for action probabilities

4. **Training Details**:
    - Optimizer: Adam with learning rate 10⁻⁴
    - Batch size: 16
    - Training iterations: 40k minibatches


## Loss Function
The ToMnet is trained with the following loss components:

### Action Prediction Loss
$$ L_{action} = -\log \hat{\pi}(a_t^{obs} \mid x_t^{obs}, e_{char}) $$

### Consumption Prediction Loss
$$  L_{consumption,i} = \sum_k -\log p_{c_k}(c_k \mid x_t^{obs}, e_{char,i}, e_{mental,i}) $$

### Successor Representation Loss
$$ L_{SR,i} = \sum \tau \sum_s -{SR} \tau(s) \log \hat{SR} \tau(s) $$ 
where $ {SR} \tau(s) = (1/Z) \sum_(\Delta t=0)^{T-t} \tau^{\Delta t} I(s_{t + \Delta t} = s) $


## Implementation Components

### Environment (environment.py, Gridworld Environment)
- Grid size: 11×11
- Walls: Randomly sampled between endpoints (can be diagonal)
  - Sections 3.1-3.2: 0-4 walls
  - Sections 3.3-3.5: 0-6 walls
- Objects: 4 consumable terminal objects
- Episode termination:
  - Object consumption
  - Timeout: 31 steps (Sections 3.1-3.2) or 51 steps (Sections 3.3-3.5)


### Agents (agents.py)
**RandomAgent Class**:
- **Policy Generation**: Samples fixed policy $\pi_i ~ Dirichlet(\alpha, \alpha, \alpha, \alpha, \alpha)$ at initialization
- **Behavior**: Uses fixed policy throughout all episodes (no learning)
- **Species Parameter**: $\alpha$ controls stochasticity ($\alpha=0.01$: near-deterministic, $\alpha=3.0$: highly stochastic)
- **Dominant Action**: Tracks $\argmax(\pi_i)$ for embedding visualization coloring
- No reward function
- Policy sampled from Dirichlet(α)

**GoalDirectedAgent Class** (for completeness, not used in Figure 3):
- **Planning**: Uses value iteration to compute optimal policy for each environment
- **Rewards**: Sampled from Dirichlet distribution over 4 object types
- **Policy**: Softmax over Q-values with temperature parameter
- Rewards: ri ~ Dirichlet(α=0.01) for object preferences
- Move penalty: -0.01
- Wall collision penalty: 0.05
- Greedy variant: move penalty 0.5
- Planning: Value iteration with γ = 1

**Deep RL Agents (Sections 3.3-3.5)**:
- Additional subgoal object (non-terminal)
- Move penalty: -0.005
- Wall collision penalty: 0.05
- Episode end without object penalty: -1
- Subgoal reward: +1
- Preferred object reward: +1
- Other objects: 0 reward (but terminal)

### ToMnet Implementation (tomnet.py)
**CharacterNet Class**:
- **Input Processing**: Handles variable-length past trajectories
- **Architecture**: 3-layer MLP (state_dim+action_dim → 128 → 128 → embedding_dim)
- **Trajectory Aggregation**: 
  ```python
  def aggregate_character_embeddings(self, past_trajectories):
      # Process each trajectory through MLP
      trajectory_embeddings = []
      for trajectory in past_trajectories:
          # trajectory shape: (seq_len, state_dim + action_dim)
          embedding = self.character_mlp(trajectory)
          # Average across time dimension first
          time_averaged = embedding.mean(dim=0)
          trajectory_embeddings.append(time_averaged)
      
      # Sum across past episodes
      if len(trajectory_embeddings) > 0:
          e_char = torch.stack(trajectory_embeddings).sum(dim=0)
          # Optional: L2 normalization
          e_char = F.normalize(e_char, p=2, dim=-1)
      else:
          e_char = torch.zeros(self.embedding_dim)
      
      return e_char
  ```
- **Output**: 2D character embeddings for Figure 3 visualization

**PredictionNet Class**:
- **Input**: Concatenation of current state + character embedding (+ mental embedding if used)
- **Multi-head Architecture**: Separate heads for actions, object consumption, and successor representation
- **Action Head**: Linear layers with softmax output for 5-action distribution

**ToMnet Class**:
- **Configuration**: `use_mental_state=False` for Figure 3 experiments
- **Forward Pass**: past_trajectories → CharacterNet → (+ current_state) → PredictionNet → action_probs
- **Loss Computation**: Cross-entropy loss on action predictions

## Directory Structure

The implementation uses experiment-specific directories for better organization and extensibility:

```
ToMnet_impl/
├── scripts/                       # Core implementation
│   ├── tomnet.py                 # ToMnet architecture
│   ├── environment.py            # GridWorld environment
│   ├── agents.py                 # RandomAgent and GoalDirectedAgent
│   ├── data_generation.py        # Trajectory collection and batch formation
│   ├── train.py                  # Advanced training system
│   ├── evaluate.py               # Cross-species evaluation and metrics
│   └── visualize_figure3.py      # Publication-quality visualization
├── shell/                        # Automation scripts
│   ├── run_exp3.sh              # Complete workflow automation
│   └── visualize_figure3.sh     # Visualization pipeline
├── notebook/                     # Interactive analysis
│   └── visualize_figure3.ipynb  # Detailed figure reproduction
├── data/{experiment_type}/       # Training data organized by experiment
│   ├── alpha_0.01.pkl
│   ├── alpha_0.03.pkl
│   └── ...
├── models/{experiment_type}/     # Trained models organized by experiment
│   ├── 0.01_best.pth
│   ├── 0.03_best.pth
│   └── mixed_best.pth
├── result/{experiment_type}/     # Results organized by experiment
│   ├── training_results.json
│   ├── evaluation_results.pkl
│   ├── model_paths.json
│   ├── data_paths.json
│   └── run_cross_species_evaluation.sh
├── plots/{experiment_type}/      # Generated plots organized by experiment
│   ├── figure3a_action_likelihood.png
│   ├── figure3b_character_embeddings.png
│   ├── figure3c_cross_species_kl.png
│   └── figure3d_mixed_species.png
└── log/                         # Execution logs with timestamps
    ├── training/{timestamp}/
    ├── evaluation/{timestamp}/
    └── visualization/{timestamp}/
```

## Advanced Training System (scripts/train.py)

### Experiment Configuration System
```python
class ExperimentConfig:
    """Centralized configuration for different experiment types"""
    def __init__(self, experiment_type: str):
        if experiment_type == "figure3":
            self.char_embedding_dim = 10
            self.use_mental_state_net = False
            self.alpha_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
            self.dropout_rate = 0.3
            self.patience = 30  # Early stopping
            self.loss_weights = {"action_loss": 1.0}
            self.predictions = ["action"]
```

### Key Features
**1. Mixed-Species Training**
- Combines agents from multiple alpha values in a single training dataset
- Enables better generalization across different agent types
- Automatically creates mixed datasets when `--mixed_training` flag is used

**2. Advanced Device Management**
```python
# Automatic device detection
if platform.system() == "Darwin":  # macOS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
elif torch.cuda.is_available():
    device = f"cuda:{args.device_id}" if args.device_id else "cuda:0"
else:
    device = "cpu"
```

**3. Background Process Management**
- Runs training in background with PID tracking
- Creates timestamped log directories
- Enables monitoring of long-running training processes

**4. Comprehensive Checkpoint System**
```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": self.model.state_dict(),
    "optimizer_state_dict": self.optimizer.state_dict(),
    "scheduler_state_dict": self.scheduler.state_dict(),
    "val_loss": val_loss,
    "best_val_loss": self.best_val_loss,
    "train_losses": self.train_losses,
    "val_losses": self.val_losses,
    "train_accuracies": self.train_accuracies,
    "val_accuracies": self.val_accuracies,
}
```

### Training Configuration
- **Batch size**: 32-64 samples (configurable up to 512)
- **Optimizer**: Adam with learning rate 1e-3 to 1e-4
- **Training episodes**: 50,000-100,000 per species
- **Dataset size**: 1000 agents × 100 episodes × variable $N_{past}$
- **Validation**: 80/20 split, early stopping on validation accuracy
- **Parallel processing**: Configurable workers for data generation
- **Gradient clipping**: Max norm 1.0 for training stability

### Command Line Interface
```bash
python scripts/train.py --experiment figure3 \
    --n_agents 1000 \
    --n_epochs 100 \
    --n_episodes_per_agent 100 \
    --batch_size 512 \
    --learning_rate 1e-3 \
    --device cuda:0 \
    --n_workers 8 \
    --mixed_training \
    --alpha_values 0.01 0.03 0.1 0.3 1.0 3.0
```

## Advanced Evaluation System (scripts/evaluate.py)

### Figure 3-Specific Evaluation Pipeline
```python
def evaluate_figure3_cross_species(model_paths, dataset_paths, device="cpu"):
    """
    Comprehensive evaluation for Figure 3 reproduction
    - Figure 3a: Action likelihood vs N_past (0, 1, 5)
    - Figure 3b: Character embeddings (N_past = 10)
    - Figure 3c: Cross-species KL divergence (N_past = 1)
    - Figure 3d: Mixed species performance (N_past = 5)
    """
```

### Bayes-Optimal Baseline Implementation
```python
class BayesOptimalBaseline:
    def __init__(self, alpha_values):
        self.alpha_values = alpha_values
        
    def compute_posterior(self, prior_alpha, observed_actions):
        """Compute Dirichlet posterior after observing actions"""
        posterior = np.array([prior_alpha] * 5)
        for action in observed_actions:
            posterior[action] += 1
        return posterior
    
    def predict_policy(self, posterior):
        """Expected policy from Dirichlet posterior"""
        return posterior / posterior.sum()
```

### Advanced Metrics System
- **Action Likelihood**: Computes probability of observed actions under predicted policy
- **KL Divergence**: Measures policy prediction accuracy using proper statistical divergence
- **Jensen-Shannon Divergence**: Alternative distance metric for policy comparison
- **Character Embedding Analysis**: Extracts and analyzes 2D embeddings

### Cross-Species Testing
- Tests models trained on one $\alpha$ value against agents from different $\alpha$ values
- Computes action prediction likelihood and KL divergence matrices
- Generates data for Figure 3a (likelihood vs $N_{past}$) and 3c (cross-species generalization)

## Publication-Quality Visualization (scripts/visualize_figure3.py)

### Figure 3 Target Results

#### Experimental Conditions
1. **Near-deterministic agents**: $\alpha = 0.01$ (concentrated policies)
2. **Stochastic agents**: $\alpha = 3.0$ (uniform-like policies) 
3. **Mixed species training**: Agents from both $\alpha$ values in same dataset
4. **Full alpha range**: $\alpha \in \{0.01, 0.03, 0.1, 0.3, 1.0, 3.0\}$ for comprehensive evaluation

#### Generated Figures
1. **Figure 3a**: Action likelihood vs training alpha
   - X-axis: Training $\alpha$ values $\{0.01, 0.03, 0.1, 0.3, 1.0, 3.0\}$
   - Y-axis: Action likelihood
   - Multiple lines for $N_{past} = 0, 1, 5$ showing learning progression
   - Includes Bayes-optimal baseline comparison
   - Professional styling with error bars and annotations

2. **Figure 3b**: 2D character embedding scatter plot
   - X-axis: Normalized $e_1$, Y-axis: Normalized $e_2$
   - Colored by dominant action with intensity based on frequency
   - Automatic PCA reduction if embeddings > 2D
   - Shows clustering patterns for $N_{past} = 10$

3. **Figure 3c**: Cross-species KL divergence matrix
   - Training species vs test species performance comparison
   - Multiple lines showing how models trained on different alphas generalize
   - Within-species vs between-species performance differences
   - Statistical significance indicators

4. **Figure 3d**: Mixed species training performance
   - Comparison of single-species vs mixed-species training
   - Shows improved generalization of mixed training
   - Performance across different test alpha values

### Advanced Visualization Features
```python
def plot_figure3a_trained_alpha_vs_likelihood(results):
    """
    Action likelihood vs training alpha with multiple N_past lines
    - Separates lines for N_past = 0, 1, 5
    - Includes Bayes-optimal baseline comparison
    - Professional styling with error bars and annotations
    """

def plot_figure3b_character_embeddings(results):
    """
    2D character embedding visualization
    - Automatic PCA reduction if embeddings > 2D
    - Action-based coloring with intensity mapping
    - Clustering analysis and visualization
    """

def plot_figure3c_test_alpha_vs_kl(results):
    """
    Complete cross-species generalization matrix
    - Multiple training alpha lines
    - Within-species vs between-species comparison
    - Statistical significance indicators
    """
```

## Enterprise-Level Workflow Automation

### Complete Workflow Script (shell/run_exp3.sh)

**Modular Execution System**
```bash
# Run complete workflow
bash shell/run_exp3.sh all

# Run individual components
bash shell/run_exp3.sh train --n_agents 1000 --n_epochs 50
bash shell/run_exp3.sh evaluate
bash shell/run_exp3.sh visualize --save
bash shell/run_exp3.sh clean  # Clean up generated files
```

**Professional Process Management**
- Background execution with PID tracking
- Timestamped logging with organized directory structure
- Colored terminal output with info/success/warning/error levels
- Comprehensive error handling and recovery

**Advanced Features**
```bash
# Process monitoring
LOG_DIR="log/training/$(date +%Y%m%d_%H%M%S)"
python scripts/train.py [...] >> "$LOG_DIR/execution.log" 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > "$LOG_DIR/process.pid"
```

### Dedicated Visualization Pipeline (shell/visualize_figure3.sh)
```bash
# Basic usage
bash shell/visualize_figure3.sh --experiment figure3

# Advanced usage with custom settings
bash shell/visualize_figure3.sh \
    --experiment figure3 \
    --results_path result/figure3/evaluation_results.pkl \
    --save \
    --output_dir plots/figure3 \
    --device cuda:0
```

## Data Flow and Evaluation Metrics

### Data Generation Pipeline (data_generation.py)
**Trajectory Collection**:
1. **Agent Sampling**: Creates agents with specified $\alpha$ parameters
2. **Environment Sampling**: Generates random 11×11 gridworlds for each episode
3. **Trajectory Recording**: Records (state, action, reward) tuples until episode termination
4. **Data Splitting**: Separates past episodes (for character inference) from query episodes (for prediction)
5. **Batch Formation**: Creates batches with variable $N_{past}$ for meta-learning

**Data Format**:
```python
batch = {
    'past_trajectories': torch.Tensor,  # (batch_size, n_past, seq_len, state_dim + action_dim)
    'current_state': torch.Tensor,      # (batch_size, state_dim)
    'true_actions': torch.Tensor,       # (batch_size,) target actions
    'agent_ids': List[int],             # For tracking agent species
    'true_policies': torch.Tensor,      # (batch_size, action_dim) for KL computation
}
```

### Evaluation Metrics
1. **Action Likelihood**: $Likelihood = \hat{\pi}(a_t^{obs} \mid x_t^{obs}, e_{char})$
2. **KL Divergence**: $D_{KL}(\pi || \hat{\pi}) = \sum_a \pi(a) \log(\pi(a) / \hat{\pi}(a))$
   - where $\pi$ is the true policy and $\hat{\pi}$ is the predicted policy
3. **Jensen-Shannon Divergence**: Alternative symmetric distance metric
4. **Character Embedding Analysis**: 2D visualization and clustering metrics

## Usage Instructions

### Quick Start - Complete Workflow
```bash
# Complete Figure 3 reproduction workflow
bash shell/run_exp3.sh all

# Monitor progress
tail -f log/training/*/execution.log
```

### Step-by-Step Execution
```bash
# 1. Train models for different alpha values
python scripts/train.py --experiment figure3 --n_agents 1000 --n_epochs 100

# 2. Evaluate cross-species performance
python scripts/evaluate.py --experiment figure3

# 3. Generate Figure 3 visualizations
python scripts/visualize_figure3.py --experiment figure3 --save_plots

# 4. View detailed analysis in Jupyter
jupyter notebook notebook/visualize_figure3.ipynb
```

### Advanced Training Options
```bash
# Train with mixed species for better generalization
python scripts/train.py --experiment figure3 --mixed_training --n_agents 1000 --n_epochs 100

# Use multiple workers for faster data generation
python scripts/train.py --experiment figure3 --n_workers 8 --n_agents 1000

# Custom alpha values for experiments
python scripts/train.py --experiment figure3 --alpha_values 0.01 0.1 1.0 --n_agents 500
```

### Custom Evaluation and Visualization
```bash
# Custom evaluation with specific metrics
python scripts/evaluate.py --experiment figure3 \
    --output_path result/figure3/custom_evaluation.pkl

# Interactive vs batch visualization
python scripts/visualize_figure3.py --experiment figure3                    # Interactive
python scripts/visualize_figure3.py --experiment figure3 --save_plots      # Save to files
```

## Technical Specifications

### Model Architecture Details
- **Input Dimension**: 726 (11×11×6 flattened state) + 5 (one-hot action)
- **Character Net**: 731 → 128 → 128 → 2 (with ReLU activations)
- **Prediction Net**: 728 (state + char_embedding) → 128 → 128 → 5 (with softmax)
- **Parameters**: ~200K total parameters for Figure 3 configuration

### Production Features

#### 1. Comprehensive Logging System
- Timestamped execution logs for all components
- Process ID tracking for background jobs
- Structured log directories with execution history
- Error logs with stack traces for debugging

#### 2. Robust Error Handling
- Graceful failure recovery with informative error messages
- Prerequisite checking before execution
- Resource availability validation
- Automatic cleanup on failure

#### 3. Scalability Features
- Configurable batch sizes and worker processes
- Memory-efficient data loading with parallel processing
- GPU memory management and device optimization
- Background execution for long-running processes

#### 4. Reproducibility Guarantees
- Fixed random seeds for consistent results
- Complete configuration preservation in checkpoints
- Deterministic data generation and model initialization
- Version tracking and experiment metadata

### Performance Optimization

#### 1. Memory Management
- Dynamic batch sizing based on available memory
- Gradient accumulation for large effective batch sizes
- Efficient data loading with multiprocessing

#### 2. GPU Utilization
- Automatic device detection and optimization
- Mixed precision training support (when available)
- Efficient tensor operations and memory pooling

#### 3. Computational Efficiency
- Parallel data generation with configurable workers
- Vectorized operations for batch processing
- Optimized model architectures with minimal parameters

## Implementation Validation

### Expected Results
The implementation should reproduce these key findings from the original paper:

1. **Learning Curve**: Action likelihood increases with more past observations (Figure 3a)
2. **Character Clustering**: 2D embeddings show clear clusters by dominant action (Figure 3b)  
3. **Cross-Species Transfer**: Lower KL divergence for within-species vs between-species prediction (Figure 3c)
4. **Mixed Training Benefit**: Models trained on mixed species generalize better than single-species models (Figure 3d)

### Troubleshooting
- **Memory Issues**: Reduce batch size or number of agents
- **Convergence Problems**: Check learning rate, ensure proper data preprocessing
- **Visualization Errors**: Verify embedding dimensions and agent labeling
- **Cross-species Evaluation**: Ensure model and data paths are correctly generated

## Extensibility

The implementation is designed for easy extension to new experiment types:

```bash
# Add new experiment type
python scripts/train.py --experiment figure4 --n_agents 1000
python scripts/evaluate.py --experiment figure4
python scripts/visualize_figure4.py --experiment figure4
```

This enhanced implementation provides a complete, production-ready system for ToMnet research with enterprise-level automation, comprehensive evaluation capabilities, and publication-quality visualization tools suitable for serious academic work and potential industrial applications.

# ToMnet Architecture Details for Figure 5 (from Appendix A.3.2)

**Data Configuration**:
- Character embedding from many past episodes: Npast ~ U{0, 10}
- **Key difference**: Only single observation-action pair (snapshot) from each past episode
- No full trajectories - just one time point per past episode

**Character Net Architecture**:
- Input: Single state/action pair from each past trajectory τij
- Processing per pair:
  1. Spatialize and concatenate (same as Experiment 1)
  2. 5-layer ResNet with 32 channels, ReLU, batch-norm
  3. Average pooling
  4. Fully-connected layer to echar,ij ∈ R²
- Aggregation: echar,i = ΣNpast(j=1) echar,ij

**Mental Net**: None

**Prediction Net**: Same as Experiment 1 (Section A.3.1)

### Goal-Directed Agent Environment (Section 3.2)

- Gridworld: 11×11 size with randomly-sampled walls (0-4 walls)
- Objects: 4 different consumable objects placed in random locations
- Agent rewards: Each agent has unique reward function ri,a ∈ (0,1) for consuming object a
    - Reward vectors sampled from Dirichlet distribution with α = 0.01
    - Move penalty: -0.01 per step
    - Wall collision penalty: 0.05


- Agent behavior: Plans using value iteration with discount factor γ = 1
- Episode termination: When agent consumes an object or after 31 steps

### Figure 5 Specific Setup

- Partial observations: ToMnet sees only single observation-action pairs from past episodes
- Past episodes: Npast ~ U{0, 10} (uniformly sampled between 0 and 10)
Training data: 1000 agents per experiment, each generating behavioral traces on random POMDPs
- Task: Predict agent's initial action in a new POMDP based on past observations

### Figure 5(b) - Action Prediction vs Npast

- X-axis: Number of past observations (Npast from 0 to 10)
- Y-axis: Average posterior probability assigned to the true action π̂(at|·)
- Metric: Shows how prediction accuracy improves with more past observations
- Baseline: Even at Npast=0, accuracy > chance due to shared policy structure

### Figure 5(d) - Character Embeddings

- Embedding dimension: 2D character embedding space (echar ∈ R²)
- Visualization: 100 different test agents plotted
- Color coding: Agents colored by ground-truth preferred object (4 colors)
- Saturation: Increases with Npast (grey dots at center = Npast=0)
- Architecture: Only character net used (no mental state net)

### Key Implementation Notes

- Character net aggregates past trajectories: echar,i = ΣNpast(j=1) echar,ij
- For partial trajectories, each past observation is just a single state-action pair
- ToMnet learns Bayes-optimal inference specialized to Dirichlet(α=0.01) prior
- Expected to see clustering in embedding space by preferred object
