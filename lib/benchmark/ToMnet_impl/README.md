# ToMnet Implementation for Figure 3 Reproduction

## Overview
This implementation reproduces the qualitative results from Figure 3 of the "Machine Theory of Mind" paper (Rabinowitz et al., 2018). The focus is on demonstrating ToMnet's ability to infer character traits of random agents through observation of their behavioral trajectories.

## ToMnet Architecture for Figure 3

### Core Mathematical Framework
The ToMnet implementation for Figure 3 uses a simplified architecture focused on character inference:

1. **Character Net ($f_\theta$)**: 
   - Processes past episode trajectories $\{\tau_{ij}\}$ into character embeddings
   - Function: $e_{char,ij} = f_\theta(\tau_{ij}^{obs})$
   - Aggregation: $e_{char,i} = \sum_j e_{char,ij}$
   - Implementation: 3-layer MLP with ReLU activations
   - Output: 2D embeddings for visualization

2. **Mental State Net**:
   - **Omitted in Figure 3 experiments** as specified in the paper
   - This simplification focuses on character-level inference only

3. **Prediction Net**:
   - Outputs action probabilities: $\hat{\pi}(a_t \mid x_t^{obs}, e_{char})$
   - Function: concatenates current state with character embedding
   - Implementation: 2-layer MLP with softmax output for 5 actions

### Loss Function

$$ L_{action} = -\log \hat{\pi}(a_t^{obs} \mid x_t^{obs}, e_{char}) $$


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
- **Policy Generation**: Samples fixed policy $\pi_i ~ Dirichlet(\alpha, \alpha, \alpha, \alpha, \alpha)$ at initialization
- **Behavior**: Uses fixed policy throughout all episodes (no learning)
- **Species Parameter**: $\alpha$ controls stochasticity ($\alpha=0.01$: near-deterministic, $\alpha=3.0$: highly stochastic)
- **Dominant Action**: Tracks $\argmax(\pi_i)$ for embedding visualization coloring

**GoalDirectedAgent Class** (for completeness, not used in Figure 3):
- **Planning**: Uses value iteration to compute optimal policy for each environment
- **Rewards**: Sampled from Dirichlet distribution over 4 object types
- **Policy**: Softmax over Q-values with temperature parameter

## Figure 3 Target Results

### Experimental Conditions
1. **Near-deterministic agents**: $\alpha = 0.01$ (concentrated policies)
2. **Stochastic agents**: $\alpha = 3.0$ (uniform-like policies) 
3. **Mixed species training**: Agents from both $\alpha$ values in same dataset (e.g., $\alpha=0.01, 3.0$)
4. **Full alpha range**: $\alpha \in {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}$ for comprehensive evaluation

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
- Creates agents with specified $\alpha$ values (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
- Generates $N_{past}$ past episodes per agent $ N_{past} \sim Uniform \left( 0, 10 \right) $
- Samples single (state, action) pair from each past episode
- Creates query episode for action prediction task

**Training Configuration**:
- **Batch size**: 32-64 samples
- **Optimizer**: Adam with learning rate 1e-3 to 1e-4
- **Training episodes**: 50,000-100,000 per species
- **Dataset size**: 1000 agents × 100 episodes × variable $N_{past}$
- **Batch processing**: Dynamic batching with padding for variable-length sequences
- **Validation**: 80/20 split, early stopping on validation accuracy
- **Metrics**: Action prediction accuracy, KL divergence vs true policy
- **Model saving**: Best model based on validation accuracy
- **Reproducibility**: Fixed random seeds for consistent results across runs

### Evaluation Process (evaluate.py)
**Cross-Species Testing**:
- Tests models trained on one $\alpha$ value against agents from different $\alpha$ values
- Computes action prediction likelihood and KL divergence matrices
- Generates data for Figure 3a (likelihood vs $N_{past}$) and 3c (cross-species generalization)

**Character Embedding Analysis**:
- Extracts 2D character embeddings for visualization
- Colors embeddings by dominant action for Figure 3b
- Compares embedding clusters between different $\alpha$ species

### Evaluation Metrics
1. Action Likelihood: $ L_{action} = \hat{\pi}(a_t^{obs} \mid x_t^{obs}, e_{char})$
2. KL Divergence: $D_{KL}(\pi || \hat{\pi}) = \sum_a \pi(a) log(\pi(a)/\hat{\pi}(a))$
   - where $\pi$ is the true policy and $\hat{\pi}$ is the predicted policy.

## Data Flow and File Structure

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

## Key Implementation Features

### Bayes-Optimal Baseline
The implementation includes a Bayes-optimal baseline that:
```python
class BayesOptimalBaseline:
    def __init__(self, alpha):
        # Prior: Dirichlet($\alpha, $\alpha, $\alpha, $\alpha, $\alpha)
        self.prior = np.array([alpha] * 5)
        
    def update(self, observed_actions):
        # Posterior update
        posterior = self.prior.copy()
        for action in observed_actions:
            posterior[action] += 1
        return posterior
    
    def predict(self, posterior):
        # Expected policy from posterior
        return posterior / posterior.sum()
```

### Model Configuration
For Figure 3 reproduction:
- Character embedding dimension: 2 (for visualization)
- Mental state net: Disabled (`use_mental_state=False`)
- Hidden dimensions: 128 across all MLPs
- Training episodes: 50,000-100,000 per species


## Visualization and Results (visualize_figure3.py)
1. **Figure 3a**: Trained $\alpha$ vs Action likelihood
   - X-axis: Trained $\alpha$ with ($\alpha \in \{0.01, 0.03, 0.1, 0.3, 1.0, 3.0 \}$)
   - Y-axis: Action prediction likelihood
   - 3 lines showing results for $N_{past} = 0, 1, 5$
   - Comparison with Bayes-optimal baseline
2. **Figure 3b**: 2D character embedding scatter plot colored by dominant action  
   - X-axis: Normalized $e_1$
   - Y-axis: Normalized $e_2$
   - Scatters are $N_{past} = 10$ past episodes
   - Darker the higher that count
3. **Figure 3c**: Test $\alpha$ vs Average KL-divergence between agents’ true and predicted policies
   - Rows: Training species ($\alpha \in {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}$)
   - Columns: $D_{KL}(\pi || \hat{\pi})$
   - 6 lines showing KL divergence for each trained $\alpha$ value {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
   - Shows generalization capabilities across species
   - Constraint with $N_{past} = 1$
4. **Figure 3d**: Figure 3c + Mixed species training performance
   - X-axis: Test $\alpha$ values {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
   - Y-axis: $D_{KL}(\pi || \hat{\pi})$
   - 3 lines: trained on $\alpha=0.01$, $\alpha=3.0$, and mixed ($\alpha=0.01 & 3.0$)

## Usage Instructions

### Quick Start
```bash
# Complete Figure 3 reproduction workflow
bash shell/run_experiment.sh all

# Or run individual steps:
# 1. Train models for different alpha values
python scripts/train.py --experiment figure3 --n_agents 1000 --n_epochs 100

# 2. Evaluate cross-species performance
python scripts/evaluate.py
bash result/figure3/run_cross_species_evaluation.sh

# 3. Generate Figure 3 visualizations
bash shell/visualize_figure3.sh --save --output_dir plots

# 4. View detailed analysis in Jupyter
jupyter notebook notebook/visualize_figure3.ipynb
```

### Cross-Species Evaluation Script Generation
```python
# scripts/evaluate.py
alpha_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
with open('result/figure3/run_cross_species_evaluation.sh', 'w') as f:
    for train_alpha in alpha_values:
        for test_alpha in alpha_values:
            cmd = f"python scripts/evaluate.py --train_alpha {train_alpha} --test_alpha {test_alpha} --n_past 1$\n"
            f.write(cmd)
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
│   ├── visualize_figure3.py   # Figure generation and analysis
│   └── generate_cross_species_evaluation.py  # Script generation for evaluation
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


