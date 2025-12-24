# Belief MultiRL

Multi-agent reinforcement learning with belief modeling (ToMnet).

## Contributions

1. **ToMnet Benchmark Re-implementation** (`lib/benchmark/`)
   - Re-implemented ToMnet and ToMnet-family (ToMnetF)

2. **Multi-Agent Environment** (`lib/env/gym_minigrid/`)
   - Custom AchieverBlocker environment for competitive multi-agent scenarios
   - Built on gymnasium and gym-minigrid

3. **Second Belief Embedding** (`script/exp7/`, `script/exp8/`)
   - Novel `e_opp2` embedding: models "what agents believe others believe"
   - Extends 2-stage ToMnet to 3-stage architecture with cross-attention

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `config/` | Configuration files (hyperparameters, environment settings) |
| `data/` | Generated/processed dataset storage |
| `lib/` | Core library modules (environments, models, utilities) |
| `notebook/` | Jupyter notebooks for analysis and experiments |
| `script/` | Experiment scripts (exp1~exp8) |
| `shell/` | Shell scripts for batch execution |
| `visualize/` | Visualization tools and outputs |

## Experiments (`script/`)

| Exp | Name | Description |
|-----|------|-------------|
| exp1 | HBT Trading | Binance data preprocessing & HBT model training (archive) |
| exp2 | Synthetic Simulation | Alon et al. (2023) replication with synthetic data |
| exp3 | KeyDoor ToMnet | Single-agent ToMnet on 9x9 KeyDoor environment |
| exp4 | AchieverBlocker | Multi-agent ToMnet with Achiever/Blocker competition |
| exp5 | Enhanced Multi-Agent | 2/3-stage ToMnet, vectorized SR, parallel data generation |
| exp6 | Unified Framework | Single/Multi-agent unified codebase with config-driven mode |
| exp7 | Second-Order Belief | e_opp2 embedding for "belief about others' beliefs" |
| exp8 | Second-Order Belief v2 | Modular agent structure with second-order belief |

### Core Components (exp3~exp8)
- `config.py` - Experiment configuration
- `generate.py` - Trajectory data generation
- `train.py` - ToMnet model training
- `evaluate.py` - Model evaluation
- `tomnet.py` - ToMnet architecture
- `visualize.py` - Result visualization


## References

- Rabinowitz, N., et al. "Machine Theory of Mind." *ICML*, 2018. [[paper]](https://arxiv.org/abs/1802.07740)
- gym-minigrid: https://github.com/mit-acl/gym-minigrid
- gymnasium: https://gymnasium.farama.org/
