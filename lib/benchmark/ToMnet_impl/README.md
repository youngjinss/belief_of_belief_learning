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

3. **Prediction Net** (only action prediction head):
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
The negative log-likelihood of the true action taken by the agent under the predicted policy:
$$ L_{action} = -\log \hat{\pi}(a_t^{obs} \mid x_t^{obs}, e_{char}) $$

### Consumption Prediction Loss
For each object, k, the negative log-likelihood that the object is/isn’t consumed:
$$  L_{consumption,i} = \sum_k -\log p_{c_k}(c_k \mid x_t^{obs}, e_{char,i}, e_{mental,i}) $$

### Successor Representation Loss
The loss here is then the cross-entropy between the predicted successor representation and the empirical one:
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

### Data Generation Pipeline (data_generation.py)
**Trajectory Collection**:
1. **Agent Sampling**: Creates agents with specified $\alpha$ parameters
2. **Environment Sampling**: Generates random 11×11 gridworlds for each episode
3. **Trajectory Recording**: Records (state, action, reward) tuples until episode termination
4. **Data Splitting**: Separates past episodes (for character inference) from query episodes (for prediction)
5. **Batch Formation**: Creates batches with variable $N_{past}$ for meta-learning

### Evaluation Metrics
1. **Action Likelihood**: $Likelihood = \hat{\pi}(a_t^{obs} \mid x_t^{obs}, e_{char})$
2. **KL Divergence**: $D_{KL}(\pi || \hat{\pi}) = \sum_a \pi(a) \log(\pi(a) / \hat{\pi}(a))$
   - where $\pi$ is the true policy and $\hat{\pi}$ is the predicted policy
3. **Jensen-Shannon Divergence**: Alternative symmetric distance metric
4. **Character Embedding Analysis**: 2D visualization and clustering metrics

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

**Prediction Net**: 
In this and subsequent experiments, we make three predictions: next-step action, which objects are consumed by the end of the episode, and successor representations. We use a shared torso for these predictions, from which separate heads branch off.

- Shared Torso: 
    - Spatialize echar,i and concatenate with the query state.
    - Pass through 5-layer ResNet with 32 channels, ReLU nonlinearities, and batch-norm

- Action Prediction Head:
    - From the torso output: 1-layer convnet with 32 channels and ReLUs Average pooling
    - Fully-connected layer to 5-dim logits
    - Softmax to get predicted policy π̂

- Consumption Prediction Head:
    - From the torso output: 1-layer convnet with 32 channels and ReLUs
    - Average pooling
    - Fully-connected layer to 4-dims
    - Sigmoid to get Bernoulli probabilities that each of the four objects will be consumed by the end of the episode (ĉ)

- Successor Representation Prediction Head:
    - From the torso output: 1-layer convnet with 32 channels and ReLUs
    - 1-layer convnet with 3 channels
    - Softmax over each channel independently
    - Gives predicted normalized SRs for three discount factors: γ = 0.5, 0.9, 0.99

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
