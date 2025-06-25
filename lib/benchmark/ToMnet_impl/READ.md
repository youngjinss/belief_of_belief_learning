# ToMnet Implementation Guide

## Overview
This document provides the essential information to implement the Theory of Mind Network (ToMnet) and reproduce experiments from Figure 3 of the "Machine Theory of Mind" paper.

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

### Training
- End-to-end training with multiple loss components
- Character embedding dimensionality: 2D for Figure 3

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

## Implementation Notes

### Loss Functions
```
L_action = -log π̂(a_t^(obs)|x_t^(obs), e_char, e_mental)
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


## Detailed Neural Network Specifications

### Character Net (f_θ)
- Input: Flattened trajectory (state, action) pairs
- Architecture: 2-3 layer MLP
- Hidden units: 64-128
- Activation: ReLU
- Output dimension: 2 (Figure 3)

### Prediction Net
- Input: Concatenated embeddings + current state
- Architecture: 2-3 layer MLP with separate heads
- Action head: Softmax over 5 actions

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


## Implementation Checklist

1. [ ] Implement GridWorld environment with random generation
2. [ ] Create RandomAgent class with Dirichlet policies
3. [ ] Build ToMnet architecture (3 modules)
5. [ ] Implement data generation pipeline
6. [ ] Set up training loop with proper batching
7. [ ] Add evaluation metrics (accuracy, KL divergence)
8. [ ] Create visualization functions
9. [ ] Implement Bayes-optimal baseline
10. [ ] Reproduce experiments and compare results

## Experiment Reproduction Steps

# Cross-Species Evaluation for Figure 3
## 개요

논문의 Figure 3는 다음과 같은 실험을 포함합니다:
- **Figure 3a**: 훈련된 alpha 값 vs action likelihood (N_past=1)
- **Figure 3b**: 2D character embeddings
- **Figure 3c**: 테스트 alpha 값 vs KL divergence (교차 종 일반화)
- **Figure 3d**: 혼합 종 훈련 성능 (N_past=5)

## 자동화된 워크플로우

### 1. 모델 훈련

여러 alpha 값으로 모델을 훈련하고 교차 종 평가 파일을 자동 생성합니다:

```bash
# 기본 alpha 값들로 훈련 (0.01, 0.1, 0.5, 1.0, 3.0)
python train.py --experiment figure3

# 사용자 정의 alpha 값들로 훈련
python train.py --experiment figure3 --alpha_values 0.01 0.1 1.0 3.0

# 혼합 종 모델도 함께 훈련
python train.py --experiment figure3 --mixed_training

# 더 많은 에이전트와 에피소드로 훈련
python train.py --experiment figure3 --n_agents 1000 --n_episodes_per_agent 100 --n_epochs 50
```

### 2. 자동 생성된 파일들

훈련 완료 후 `evaluation_configs/` 디렉토리에 다음 파일들이 자동 생성됩니다:

```
evaluation_configs/
├── model_paths.json                 # 훈련된 모델 경로들
├── data_paths.json                  # 테스트 데이터 경로들
├── run_cross_species_evaluation.sh  # 실행 스크립트
└── evaluation_summary.json          # 요약 정보
```

**model_paths.json 예시:**
```json
{
  "alpha_0.01": "/absolute/path/to/models/figure3_0.01_best.pth",
  "alpha_0.1": "/absolute/path/to/models/figure3_0.1_best.pth",
  "alpha_3.0": "/absolute/path/to/models/figure3_3.0_best.pth",
  "mixed": "/absolute/path/to/models/figure3_mixed_best.pth"
}
```

**data_paths.json 예시:**
```json
{
  "alpha_0.01": "/absolute/path/to/data/figure3_alpha_0.01.pkl",
  "alpha_0.1": "/absolute/path/to/data/figure3_alpha_0.1.pkl", 
  "alpha_3.0": "/absolute/path/to/data/figure3_alpha_3.0.pkl"
}
```

### 3. 교차 종 평가 실행

자동 생성된 스크립트로 평가를 실행합니다:

```bash
# 자동 생성된 스크립트 사용 (권장)
bash evaluation_configs/run_cross_species_evaluation.sh

# 또는 직접 실행
python evaluate.py \
    --experiment figure3 \
    --model_paths_json evaluation_configs/model_paths.json \
    --data_paths_json evaluation_configs/data_paths.json \
    --output_path result/figure3_cross_species_results.pkl \
    --device cuda
```

### 4. 결과 시각화

Jupyter notebook으로 결과를 시각화합니다:

```bash
jupyter notebook visualize_figure3.ipynb
```

## 생성되는 평가 데이터

교차 종 평가는 다음 구조의 데이터를 생성합니다:

```python
{
    "figure3a": {
        "trained_alphas": [0.01, 0.1, 0.5, 1.0, 3.0, ...],
        "action_likelihoods": [0.85, 0.82, 0.78, ...],
        "bayes_optimal": [0.90, 0.85, 0.80, ...]
    },
    "figure3c": {
        "train_alphas": [0.01, 0.1, 3.0],
        "test_alphas": [0.01, 0.1, 0.5, 1.0, 3.0],
        "kl_matrix": [[0.2, 1.5, 2.8], [1.2, 0.3, 2.1], ...],
        "bayes_kl_matrix": [[0.1, 1.3, 2.5], ...]
    },
    "figure3d": {
        "mixed_species": {
            "alpha_0.01": {"action_accuracy": 0.75, "mean_kl_divergence": 1.2},
            "alpha_3.0": {"action_accuracy": 0.73, "mean_kl_divergence": 1.4}
        }
    },
    "character_embeddings": {
        "alpha_0.01": {"embeddings": [...], "agent_ids": [...]}
    }
}
```

## 고급 사용법

### 커스텀 평가 설정

```bash
# 특정 모델과 데이터로만 평가
python evaluate.py \
    --experiment figure3 \
    --model_paths_json my_custom_models.json \
    --data_paths_json my_custom_data.json \
    --output_path my_results.pkl
```

### 개별 모델 평가

```bash
# 단일 모델 평가 (레거시 방식)
python evaluate.py \
    --experiment figure3 \
    --model_path models/figure3_0.01_best.pth \
    --data_path data/figure3_alpha_0.01.pkl \
    --output_path single_model_results.pkl
```

## 문제 해결

### 1. 모듈 import 오류
```bash
# 올바른 디렉토리에서 실행 확인
cd /path/to/ToMnet_impl/
python train.py --experiment figure3
```

### 2. 메모리 부족
```bash
# 배치 크기 줄이기
python train.py --experiment figure3 --batch_size 16 --n_agents 500
```

### 3. GPU 오류
```bash
# CPU 사용하기
python train.py --experiment figure3 --device cpu
```

### 4. 평가 파일 없음
훈련이 완료되었는지 확인하고 `evaluation_configs/` 디렉토리가 존재하는지 확인하세요.

## 파일 구조

```
ToMnet_impl/
├── train.py                         # 훈련 스크립트 (JSON 파일 자동 생성)
├── evaluate.py                      # 평가 스크립트 (교차 종 평가 지원)
├── visualize_figure3.ipynb          # 시각화 노트북 (수정된 Figure 3)
├── data/                            # 생성된 데이터
│   ├── figure3_alpha_0.01.pkl
│   ├── figure3_alpha_0.1.pkl
│   └── ...
├── models/                          # 훈련된 모델
│   ├── figure3_0.01_best.pth
│   ├── figure3_0.1_best.pth
│   └── ...
├── evaluation_configs/              # 자동 생성된 평가 설정
│   ├── model_paths.json
│   ├── data_paths.json
│   ├── run_cross_species_evaluation.sh
│   └── evaluation_summary.json
└── result/                          # 평가 결과
    └── figure3_cross_species_results.pkl
```

## 다음 단계

1. 모델 훈련: `python train.py --experiment figure3`
2. 평가 실행: `bash evaluation_configs/run_cross_species_evaluation.sh`  
3. 결과 확인: `jupyter notebook visualize_figure3.ipynb`

이 워크플로우를 통해 논문의 Figure 3를 정확하게 재현할 수 있습니다.


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
}
```

### Usage:

# Train models
python train.py --experiment figure3 --n_agents 100 --n_epochs 100

# Evaluate models
python evaluate.py --experiment figure3 --model_path models/figure3_best.pth --data_path data/figure3_data.pkl
python evaluate.py --experiment figure3 --model_path models/figure3_0.01_best.pth --data_path data/figure3_alpha_0.01.pkl

# Run visualization notebook
jupyter notebook visualize_figure3.ipynb