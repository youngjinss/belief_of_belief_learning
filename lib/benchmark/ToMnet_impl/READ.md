# ToMnet Implementation for Figure 3 Reproduction

## Overview
This implementation reproduces the qualitative results from Figure 3 of the "Machine Theory of Mind" paper (Rabinowitz et al., 2018). The focus is on demonstrating ToMnet's ability to infer character traits of random agents through observation of their behavioral trajectories.

## ToMnet Architecture for Figure 3

### Core Mathematical Framework
The ToMnet implementation for Figure 3 uses a simplified architecture focused on character inference:

1. **Character Net (f_θ)**: 
   - Processes past episode trajectories {τ_ij} into character embeddings
   - Function: e_char,ij = f_θ(τ_ij^(obs))
   - Aggregation: e_char,i = Σ_j e_char,ij
   - Implementation: 3-layer MLP with ReLU activations
   - Output: 2D embeddings for visualization

2. **Mental State Net**:
   - **Omitted in Figure 3 experiments** as specified in the paper
   - This simplification focuses on character-level inference only

3. **Prediction Net**:
   - Outputs action probabilities: π̂(a_t|x_t^(obs), e_char)
   - Function: concatenates current state with character embedding
   - Implementation: 2-layer MLP with softmax output for 5 actions

### Loss Function
```
L_action = -log π̂(a_t^(obs)|x_t^(obs), e_char)
```

## Current Implementation Details

### Environment (environment.py)
**GridWorld Class**:
- **State Representation**: 11×11×6 tensor with channels for walls, 4 object types, and agent position
- **Random Generation**: Each episode generates 0-4 random walls and places 4 consumable objects
- **Action Space**: {up, down, left, right, stay} = {0, 1, 2, 3, 4}
- **Episode Termination**: Ends when agent consumes an object or reaches 31 steps
- **State Flattening**: Converts 3D state to 726-dimensional vector for neural network input

### Agents (agents.py)
**RandomAgent Class**:
- **Policy Generation**: Samples fixed policy π_i ~ Dirichlet(α, α, α, α, α) at initialization
- **Behavior**: Uses fixed policy throughout all episodes (no learning)
- **Species Parameter**: α controls stochasticity (α=0.01: near-deterministic, α=3.0: highly stochastic)
- **Dominant Action**: Tracks argmax(π_i) for embedding visualization coloring

**GoalDirectedAgent Class** (for completeness, not used in Figure 3):
- **Planning**: Uses value iteration to compute optimal policy for each environment
- **Rewards**: Sampled from Dirichlet distribution over 4 object types
- **Policy**: Softmax over Q-values with temperature parameter

## Figure 3 Target Results

### Experimental Conditions
1. **Near-deterministic agents**: α = 0.01 (concentrated policies)
2. **Stochastic agents**: α = 3.0 (uniform-like policies) 
3. **Mixed species training**: Agents from both α values in same dataset (e.g., α=0.01, 3.0)
4. **Full alpha range**: α ∈ {0.01, 0.03, 0.1, 0.3, 1.0, 3.0} for comprehensive evaluation

### ToMnet Implementation (tomnet.py)
**CharacterNet Class**:
- **Input Processing**: Handles variable-length past trajectories
- **Architecture**: 3-layer MLP (state_dim+action_dim → 128 → 128 → embedding_dim)
- **Trajectory Aggregation**: 
1) Step 1: Average embeddings across time steps for each trajectory
2) Step 2: Sum averaged embeddings across past episodes  
3) Step 3: Apply L2 normalization if needed
- **Output**: 2D character embeddings for Figure 3 visualization

**PredictionNet Class**:
- **Input**: Concatenation of current state + character embedding (+ mental embedding if used)
- **Multi-head Architecture**: Separate heads for actions, object consumption, and successor representation
- **Action Head**: Linear layers with softmax output for 5-action distribution

**ToMnet Class**:
- **Configuration**: `use_mental_state=False` for Figure 3 experiments
- **Forward Pass**: past_trajectories → CharacterNet → (+ current_state) → PredictionNet → action_probs
- **Loss Computation**: Cross-entropy loss on action predictions

### Training Process (train.py)
**Data Generation**:
- Creates agents with specified α values (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
- Generates N_past past episodes per agent (N_past ~ Uniform{0, 10})
- Samples single (state, action) pair from each past episode
- Creates query episode for action prediction task

**Training Loop**:
- Batch size: 32-64 samples
- Optimizer: Adam with learning rate 1e-3 to 1e-4
- Training episodes: 50,000-100,000 per species
- Validation: 80/20 split, early stopping on validation accuracy
- Metrics: Action prediction accuracy, KL divergence vs true policy
- Model saving: Best model based on validation accuracy

### Evaluation Process (evaluate.py)
**Cross-Species Testing**:
- Tests models trained on one α value against agents from different α values
- Computes action prediction likelihood and KL divergence matrices
- Generates data for Figure 3a (likelihood vs N_past) and 3c (cross-species generalization)

**Character Embedding Analysis**:
- Extracts 2D character embeddings for visualization
- Colors embeddings by dominant action for Figure 3b
- Compares embedding clusters between different α species

### Evaluation Metrics
1. Action Likelihood: L_action = π̂(a_t^(obs)|x_t^(obs), e_char)
2. KL Divergence: D_KL(π||π̂) = Σ_a π(a) log(π(a)/π̂(a))
   - where π is the true policy and π̂ is the predicted policy.

### Target Outputs
1. **Figure 3a**: Action likelihood vs number of past observations
   - X-axis: Number of past observations (N_past = 0 to 10)
   - Y-axis: Action prediction likelihood
   - 3 lines showing results for N_past = 0, 1, 5
   - Comparison with Bayes-optimal baseline
2. **Figure 3b**: 2D character embedding scatter plot colored by dominant action  
3. **Figure 3c**: Cross-species generalization heatmap
   - Rows: Training species (α ∈ {0.01, 0.03, 0.1, 0.3, 1.0, 3.0})
   - Columns: Test species (α ∈ {0.01, 0.03, 0.1, 0.3, 1.0, 3.0})
   - 6 lines showing KL divergence for each trained α value
   - Shows generalization capabilities across species
4. **Figure 3d**: Mixed species training performance
   - X-axis: Test α values {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
   - Y-axis: KL divergence between predicted and true policies
   - 3 lines: trained on α=0.01, α=3.0, and mixed (α=0.01 & 3.0)

## Data Flow and File Structure

### Data Generation Pipeline (data_generation.py)
**Trajectory Collection**:
1. **Agent Sampling**: Creates agents with specified α parameters
2. **Environment Sampling**: Generates random 11×11 gridworlds for each episode
3. **Trajectory Recording**: Records (state, action, reward) tuples until episode termination
4. **Data Splitting**: Separates past episodes (for character inference) from query episodes (for prediction)
5. **Batch Formation**: Creates batches with variable N_past for meta-learning

**Data Format**:
```python
batch = {
    'past_trajectories': torch.Tensor,  # (batch_size, n_past, seq_len, state_dim + action_dim)
    'current_state': torch.Tensor,      # (batch_size, state_dim)
    'true_actions': torch.Tensor,       # (batch_size,) target actions
    'agent_ids': List[int],             # For tracking agent species
}
```

## Key Implementation Features

### Bayes-Optimal Baseline
The implementation includes a Bayes-optimal baseline that:
- Maintains Dirichlet posterior over action probabilities
- Updates posterior with observed actions: α_posterior[a] += 1 for each observed action a
- Computes expected policy from posterior parameters
- Provides theoretical upper bound for Figure 3 comparisons

### Model Configuration
For Figure 3 reproduction:
- Character embedding dimension: 2 (for visualization)
- Mental state net: Disabled (`use_mental_state=False`)
- Hidden dimensions: 128 across all MLPs
- Training episodes: 50,000-100,000 per species


## Visualization and Results (visualize_figure3.py)

### Generated Plots
**Figure 3a**: Action Likelihood vs Past Observations
- X-axis: Number of past observations (0-10)
- Y-axis: Action prediction likelihood
- Compares ToMnet vs Bayes-optimal baseline
- Shows improvement with more observations

**Figure 3b**: Character Embeddings  
- 2D scatter plot of character embeddings
- Points colored by dominant action (argmax of agent's policy)
- Demonstrates clustering by behavioral similarity

**Figure 3c**: Cross-Species Generalization
- Heatmap of KL divergence between predicted and true policies
- Rows: Training species (α values)
- Columns: Test species (α values) 
- Shows generalization capabilities across species

**Figure 3d**: Mixed Species Training
- Performance comparison for models trained on mixed vs single species
- Demonstrates hierarchical inference capabilities

## Usage Instructions

### Quick Start
```bash
# Complete Figure 3 reproduction workflow
bash shell/run_experiment.sh all

# Or run individual steps:
# 1. Train models for different alpha values
python scripts/train.py --experiment figure3 --n_agents 1000 --n_epochs 100

# 2. Evaluate cross-species performance (auto-generated script)
bash result/figure3/run_cross_species_evaluation.sh

# 3. Generate Figure 3 visualizations
bash shell/visualize_figure3.sh --save --output_dir plots

# 4. View detailed analysis in Jupyter
jupyter notebook notebook/visualize_figure3.ipynb
```

### Generated Files
- **Models**: `models/figure3_{alpha}_best.pth` - Trained ToMnet models
- **Data**: `data/figure3_alpha_{alpha}.pkl` - Agent trajectory datasets  
- **Results**: `result/figure3/figure3_cross_species_results.pkl` - Evaluation metrics
- **Plots**: `plots/figure3{a,b,c,d}_{description}.png` - Reproduction figures

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

## Technical Specifications

### Model Architecture Details
- **Input Dimension**: 726 (11×11×6 flattened state) + 5 (one-hot action)
- **Character Net**: 731 → 128 → 128 → 2 (with ReLU activations)
- **Prediction Net**: 728 (state + char_embedding) → 128 → 128 → 5 (with softmax)
- **Parameters**: ~200K total parameters for Figure 3 configuration

### Training Configuration
- **Dataset Size**: 1000 agents × 100 episodes × variable N_past
- **Batch Processing**: Dynamic batching with padding for variable-length sequences
- **Validation**: 80/20 split, early stopping on validation accuracy
- **Reproducibility**: Fixed random seeds for consistent results across runs


## Repository Structure
```
ToMnet_impl/
├── scripts/                    # Core implementation
│   ├── tomnet.py              # ToMnet architecture (CharacterNet, PredictionNet, ToMnet)
│   ├── environment.py         # GridWorld environment with random generation
│   ├── agents.py              # RandomAgent and GoalDirectedAgent classes
│   ├── data_generation.py     # Trajectory collection and batch formation
│   ├── train.py               # Training loop with multi-species support
│   ├── evaluate.py            # Cross-species evaluation and metrics
│   └── visualize_figure3.py   # Figure generation and analysis
├── shell/                     # Automation scripts
│   ├── run_experiment.sh      # Complete workflow automation
│   └── visualize_figure3.sh   # Visualization pipeline
├── notebook/                  # Interactive analysis
│   └── visualize_figure3.ipynb # Detailed figure reproduction
├── data/                      # Generated datasets
├── models/                    # Trained model checkpoints
├── result/                    # Evaluation results and configs
└── plots/                     # Generated Figure 3 reproductions
```

This implementation provides a complete reproduction of the ToMnet Figure 3 experiments, demonstrating the network's ability to learn character embeddings that capture agent behavioral patterns and enable cross-species generalization in theory of mind tasks.


