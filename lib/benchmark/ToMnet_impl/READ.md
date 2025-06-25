# ToMnet Implementation Guide

## Overview
This document provides the essential information to implement the Theory of Mind Network (ToMnet) and reproduce experiments from Figures 3 and 5 of the "Machine Theory of Mind" paper.

## ToMnet Architecture

### Core Components
The ToMnet consists of three main modules:

1. **Character Net**: 
   - Parses past episode trajectories {τ_ij} into character embeddings
   - Function: `e_char,ij = f_θ(τ_ij^(obs))`
   - Aggregation: `e_char,i = Σ_j e_char,ij`

2. **Mental State Net**:
   - Processes current episode trajectory up to time t-1
   - Function: `e_mental,i = g_φ([τ_ij^(obs)]_0:t-1, e_char,i)`
   - Note: Not used in Figure 3 experiments

3. **Prediction Net**:
   - Outputs behavioral predictions using character and mental embeddings
   - Predictions include:
     - Next-step action probabilities: π̂(·|x_t^(obs), e_char, e_mental)
     - Object consumption probabilities: ĉ
     - Successor representations: SR

### Training
- End-to-end training with multiple loss components
- Character embedding dimensionality: 2D for Figure 3, 8D for Figure 5

## Environment Setup

### Gridworld Specifications
- Size: 11×11
- Walls: 0-4 random walls (Sections 3.1-3.2)
- Objects: 4 consumable terminal objects (colored squares)
- Episode length: Max 31 steps for random/algorithmic agents

## Figure 3: Random Agents

### Agent Design
- **Species**: S(α) where α is Dirichlet concentration parameter
- **Policy**: Stochastic with fixed action probabilities π_i(·) = π_i ~ Dir(α)
- **Action space**: up/down/left/right/stay

### Experimental Conditions
1. **Near-deterministic agents**: α = 0.01
2. **Stochastic agents**: α = 3
3. **Mixed species**: Train on both α = 0.01 and α = 3

### Training Setup
- 1000 agents per species
- Variable past episodes: N_past ~ U{0, 10}
- Single observation/action pair per past trajectory
- Task: Predict initial action in new POMDP

### Key Results to Reproduce
1. Action likelihood vs number of past observations
2. 2D character embeddings colored by most frequent action
3. KL-divergence between true and predicted policies
4. Hierarchical inference on mixed species

## Figure 5: Goal-Directed Agents

### Agent Design
- **Reward function**: r_i,a ∈ (0,1) for consuming object a
- **Reward distribution**: r_i ~ Dir(α = 0.01)
- **Movement penalty**: -0.01 per step
- **Wall penalty**: 0.05
- **Planning**: Value iteration with optimal policy π*_i

### Experimental Variations
1. **Standard agents**: Low movement cost (0.01)
2. **Greedy agents**: High movement cost (0.5) - seek closest object

### Training Setup
- Observe full trajectories in past MDPs
- Variable past episodes: N_past ~ U{0, 10}
- Task: Predict behavior in current MDP given partial trajectory

### Predictions to Implement
1. Next action given current state
2. Which object will be consumed
3. Successor representation (discounted state occupancy)

### Key Results to Reproduce
1. Policy predictions for different starting positions
2. Effect of N_past on prediction accuracy
3. 2D character embeddings showing reward preferences
4. Inference of cost-reward balance from single trajectories

## Implementation Notes

### Loss Functions
```
L_action = -log π̂(a_t^(obs)|x_t^(obs), e_char, e_mental)
L_consumption = Σ_k -log p_ck(c_k|x_t^(obs), e_char, e_mental)
L_SR = Σ_γ Σ_s -SR_γ(s) log ŜR_γ(s)
```

### Evaluation Metrics
- Action prediction accuracy
- KL divergence: D_KL(π||π̂)
- Jensen-Shannon divergence for policy comparisons

### Data Generation
1. Sample agents from species distribution
2. Generate behavioral traces on random gridworlds
3. Split into training (past) and query (current) episodes
4. Ensure variety in N_past for meta-learning

## Critical Implementation Details

### For Figure 3
- Omit mental state net
- Use 2D character embeddings for visualization
- Compare with Bayes-optimal inference baseline

### For Figure 5
- Include full past trajectories
- Implement value iteration for ground truth agents
- Visualize goal inference through policy heatmaps

## Detailed Neural Network Specifications

### Character Net (f_θ)
- Input: Flattened trajectory (state, action) pairs
- Architecture: 2-3 layer MLP
- Hidden units: 64-128
- Activation: ReLU
- Output dimension: 2 (Figure 3), 8 (Figure 5)

### Mental State Net (g_φ) - Figure 5 only
- Input: Current trajectory + character embedding
- Architecture: LSTM or GRU
- Hidden units: 64-128
- Output dimension: 64

### Prediction Net
- Input: Concatenated embeddings + current state
- Architecture: 2-3 layer MLP with separate heads
- Action head: Softmax over 5 actions
- Consumption head: Sigmoid for each object
- SR head: Linear output for state occupancy

## Code Structure Recommendation

```python
# 1. Environment Module
class GridWorld:
    def __init__(self, size=11):
        # Initialize grid, walls, objects
    def reset(self):
        # Generate new random layout
    def step(self, action):
        # Execute action, return new state

# 2. Agent Module  
class RandomAgent:
    def __init__(self, alpha):
        # Sample policy from Dirichlet(alpha)
    def act(self, state):
        # Return action based on fixed policy

class GoalDirectedAgent:
    def __init__(self, rewards):
        # Initialize with reward function
    def plan(self, gridworld):
        # Run value iteration
    def act(self, state):
        # Return optimal action

# 3. ToMnet Module
class CharacterNet(nn.Module):
    # Process past trajectories
class MentalStateNet(nn.Module):
    # Process current trajectory  
class PredictionNet(nn.Module):
    # Output predictions

# 4. Training Loop
def train_tomnet(agents, episodes):
    # Sample agent and trajectories
    # Forward pass through ToMnet
    # Compute losses and backprop
```

## Data Format Specifications

### State Representation
- 11×11 binary arrays for: walls, objects (4 channels), agent position
- Flattened to 1D vector for neural network input

### Trajectory Format
```python
trajectory = {
    'states': np.array(...),    # Shape: (T, 11, 11, 6)
    'actions': np.array(...),    # Shape: (T,) with values 0-4
    'rewards': np.array(...),    # Shape: (T,)
}
```

## Training Hyperparameters

### General Settings
- Batch size: 32-64
- Learning rate: 1e-3 to 1e-4
- Optimizer: Adam
- Training episodes: 50,000-100,000
- Validation split: 80/20

### Figure 3 Specific
- Training agents: 1000 per species
- Episodes per agent: 100
- Embedding regularization: None

### Figure 5 Specific  
- Training agents: 40
- Episodes per agent: 1000
- Value iteration: γ=0.99, convergence threshold=1e-6

## Baseline Implementation

### Bayes-Optimal Inference (Figure 3)
```python
def bayes_optimal_update(prior_alpha, observed_actions):
    # Update Dirichlet posterior
    posterior_alpha = prior_alpha.copy()
    for action in observed_actions:
        posterior_alpha[action] += 1
    # Compute expected policy
    return posterior_alpha / posterior_alpha.sum()
```

## Visualization Code Examples

### Figure 3 Plots
```python
# 1. Action likelihood plot
def plot_action_likelihood(n_past, likelihoods):
    # Compare ToMnet vs Bayes-optimal

# 2. 2D embedding visualization  
def plot_character_embeddings(embeddings, action_labels):
    # Scatter plot colored by dominant action

# 3. KL divergence heatmap
def plot_kl_matrix(train_alphas, test_alphas, kl_values):
    # Heatmap of cross-species generalization
```

### Figure 5 Plots
```python
# 1. Policy vector field
def plot_policy_field(grid, policy):
    # Arrow plot showing action probabilities

# 2. Goal inference visualization
def plot_consumption_probs(grid, probs):
    # Heatmap of object consumption predictions
```

## Implementation Checklist

1. [ ] Implement GridWorld environment with random generation
2. [ ] Create RandomAgent class with Dirichlet policies
3. [ ] Create GoalDirectedAgent with value iteration
4. [ ] Build ToMnet architecture (3 modules)
5. [ ] Implement data generation pipeline
6. [ ] Set up training loop with proper batching
7. [ ] Add evaluation metrics (accuracy, KL divergence)
8. [ ] Create visualization functions
9. [ ] Implement Bayes-optimal baseline
10. [ ] Reproduce experiments and compare results

## Experiment Reproduction Steps

### Figure 3 Reproduction
1. **Generate Random Agents**
   - Create 1000 agents with α=0.01 (deterministic)
   - Create 1000 agents with α=3 (stochastic)
   - Create mixed dataset with both types

2. **Training Process**
   - For each agent, generate trajectories on random grids
   - Train separate ToMnets for each α value
   - Train one ToMnet on mixed data

3. **Expected Results**
   - Panel (a): Action likelihood should increase with N_past
   - Panel (b): 2D embeddings should cluster by action preference
   - Panel (c): Lower KL when test α matches train α
   - Panel (d): Mixed training should handle both species

### Figure 5 Reproduction
1. **Generate Goal-Directed Agents**
   - Create agents with random reward vectors r_i ~ Dir(0.01)
   - Run value iteration to get optimal policies
   - Include 20% high-cost agents (movement cost 0.5)

2. **Data Collection**
   - Full trajectories showing goal-seeking behavior
   - Partial observation-action pairs for N_past experiments

3. **Expected Results**
   - Panel (a): Prediction accuracy increases with N_past
   - Panel (b): 2D embeddings cluster by preferred object color
   - Panel (c): Policy predictions show goal-directed movement
   - Panel (d): ToMnet infers cost-reward tradeoffs

## Debugging Tips

1. **Convergence Issues**
   - Start with smaller grid (5×5) for faster debugging
   - Ensure value iteration converges before using agents
   - Check that loss decreases steadily

2. **Embedding Quality**
   - Use PCA to verify embedding structure
   - Check that similar agents have similar embeddings
   - Ensure regularization isn't too strong

3. **Prediction Accuracy**
   - Verify baseline (random) performance first
   - Check that more observations improve predictions
   - Ensure proper masking for variable-length sequences

### Training Batch Format
```python
batch = {
    'past_trajectories': [...],   # List of N_past trajectories
    'current_state': np.array(...),
    'true_action': int,
    'true_consumption': np.array(...),  # One-hot
    'true_sr': np.array(...),
}
```

### Usage:

# Train models
python train.py --experiment both --n_agents 100 --n_epochs 100

# Evaluate models
python evaluate.py --experiment figure3 --model_path models/figure3_best.pth --data_path data/figure3_data.pkl
python evaluate.py --experiment figure3 --model_path models/figure3_0.01_best.pth --data_path data/figure3_alpha_0.01.pkl

# Run visualization notebooks
jupyter notebook visualize_figure3.ipynb
jupyter notebook visualize_figure5.ipynb