# Belief of Belief in Multiplayer Simulation employing RL

Multi-agent reinforcement learning with belief modeling, built on the Theory of Mind
Network (ToMnet; Rabinowitz et al., 2018) architecture and extended with a second-order
belief embedding.

## Contributions

1. **ToMnet benchmark re-implementations** (`lib/benchmark/`)
   - `ToMnet_impl/` — an implementation of the paper's Figure 3 and Figure 5 experiments
   - `ToMnetF_impl/` — a ToMnet-family variant adapted from
     [ToMnet-N](https://github.com/Nik-Kras/ToMnet-N), organised as five staged experiments

2. **AchieverBlocker multi-agent environment** (`lib/env/gym_minigrid/envs/achiever_blocker.py`)
   - A competitive two-agent grid task: an Achiever pursues a colour-preferred door while a
     Blocker tries to break it first
   - Built on gymnasium and gym-minigrid

3. **Second-order belief embedding** (`script/exp7/`, `script/exp8/`)
   - `e_opp2`: models what an agent believes *others* believe
   - Extends the 2-stage ToMnet into a 3-stage architecture combined by cross-attention

## Directory Structure

| Directory | Contents |
|-----------|----------|
| `beliefrl/` | Shared core: agents, ToMnet model, config base, data and training helpers |
| `script/` | Experiment code, one directory per experiment (exp1–exp8) |
| `script/tools/` | Data-inspection scripts shared by every experiment |
| `tests/` | Golden-output regression harness (`pytest` from the repo root) |
| `lib/env/` | Environment implementations, including AchieverBlocker (v1 and v2) |
| `lib/benchmark/` | ToMnet and ToMnetF re-implementations |
| `lib/policy/` | Archived policy code from exp1 |
| `data/` | Generated trajectory datasets, named by environment variant |
| `shell/` | Batch run scripts (exp1–exp7) |
| `config/` | YAML configuration — used only by exp1's trading pipeline |
| `notebook/` | Analysis notebooks for exp1 and exp2 |
| `visualize/` | Standalone visualization tools (version1–version3) |

Note that `config/` does **not** hold hyperparameters for the ToMnet experiments. Each of
exp3–exp8 carries its own `config.py`, now subclassing `beliefrl.config.BaseConfig`.

### The `beliefrl` core

exp3–exp8 previously carried near-identical copies of the same code. What is provably
identical between them now lives in one place:

| Package | Contents |
|---------|----------|
| `beliefrl/agents/` | Achiever and blocker policies, including `BaseValueAgent` |
| `beliefrl/model/` | ToMnet architecture and its tensor helpers |
| `beliefrl/config/` | `BaseConfig` — the 21 accessor methods every experiment shared |
| `beliefrl/data/` | Trajectory generation helpers and dataset loading |
| `beliefrl/train/` | Early stopping, epoch reporting, training-curve plots |
| `beliefrl/viz/` | SR heatmaps, shared embedding/metric plots, and two-player trajectory replay (`replay.py`) |
| `beliefrl/env/` | Single bootstrap making `lib/env` importable |

Each experiment's `Config` subclasses `BaseConfig` and its modules re-export the shared
names, so `from utils import set_seed` and `from blockers import RandomAgent` still work.

Every removal was gated the same way: a definition moved only when its AST was identical
to the core's, and the golden harness had to stay green afterwards without re-recording.

What did **not** move is as informative as what did. `tomnet.py`, `data_generation.py`,
`train.py`, `visualize.py`, and most agent classes genuinely differ between experiments —
they are not copy-paste, so each experiment keeps its own. In particular exp7 and exp8
share a parameter count and architecture hash but their `forward` implementations differ.

Run `pip install -e .` to put `beliefrl` on the path; without it each experiment's
`config.py` and `utils.py` add the repo root to `sys.path` themselves.

## Experiments

| Exp | Name | Description | README |
|-----|------|-------------|--------|
| exp1 | HBT Trading | Binance data preprocessing and HBT model training (archived) | — |
| exp2 | Synthetic Simulation | Alon et al. (2023) replication on synthetic data | — |
| exp3 | KeyDoor ToMnet | Single-agent ToMnet on a 9×9 multi-colour key-door grid | [link](script/exp3/README.md) |
| exp4 | AchieverBlocker | First multi-agent ToMnet on the Achiever/Blocker task | [link](script/exp4/README.md) |
| exp5 | Enhanced Multi-Agent | Toggleable 2/3-stage ToMnet, vectorized SR, parallel generation | [link](script/exp5/README.md) |
| exp6 | Unified Framework | Single- and multi-agent modes from one config-driven codebase | [link](script/exp6/README.md) |
| exp7 | Second-Order Belief | `e_opp2` embedding with `SecondBeliefNet` and cross-attention | [link](script/exp7/README.md) |
| exp8 | Second-Order Belief v2 | exp7's architecture plus a modular `agents/` package and partial observability | [link](script/exp8/README.md) |

exp7 and exp8 share a near-identical `tomnet.py`. The substantive difference is in
`data_generation.py`, where exp8 adds partial-observation masking.

### Common layout (exp3–exp8)

| File | Role |
|------|------|
| `config.py` | Experiment parameters; exp5–exp8 subclass `beliefrl.config.BaseConfig` |
| `generate.py` | Trajectory data generation |
| `train.py` | ToMnet training |
| `evaluate.py` | Evaluation and metrics |
| `tomnet.py` | Model architecture (exp8 uses `beliefrl.model` instead) |
| `visualize.py` | Result plots |

## Tests

```bash
pytest                         # from the repo root
python tests/golden.py check   # same checks, no pytest dependency
```

Eighteen golden-output checks across exp3–exp8: each experiment's configuration, model,
agent rollouts, and a one-step training smoke test are compared against recordings made
before the refactor. See [tests/README.md](tests/README.md), which also lists the defects
the harness uncovered and the open issues it documents.

## Running Experiments

Each experiment directory documents its own commands. One repo-wide caveat applies to
exp3–exp8: the scripts apply their command-line flags **only when `--config_override` is also
passed**. Without it, flags such as `--epochs` or `--batch_size` are parsed and silently
ignored, and the configured defaults are used instead.

```bash
# flags below are ignored
python script/exp5/train.py --epochs 100 --batch_size 1024

# flags below take effect
python script/exp5/train.py --config_override --epochs 100 --batch_size 1024
```

Batch scripts live in `shell/exp1` through `shell/exp7`; exp8 has none.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` is a full pinned freeze of the development environment, so it includes
packages unrelated to the RL experiments.

## References

- Rabinowitz, N., et al. "Machine Theory of Mind." *ICML*, 2018. [[paper]](https://arxiv.org/abs/1802.07740)
- gym-minigrid: https://github.com/mit-acl/gym-minigrid
- gymnasium: https://gymnasium.farama.org/
