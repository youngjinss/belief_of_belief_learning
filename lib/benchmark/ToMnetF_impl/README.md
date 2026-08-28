# ToMnetF Re-implementation

CNN/ResNet-based Theory of Mind Network that predicts a gridworld agent's next action, terminal object consumption, and successor representation from observed trajectories, following the character-net / prediction-net split of "Machine Theory of Mind" (Rabinowitz et al., 2018, ICML). Environment, A* agent, and value agent are adapted from [Nik-Kras/ToMnet-N](https://github.com/Nik-Kras/ToMnet-N) (Nikita Krasnytskyi); the CNN architecture, training pipeline, evaluation, and visualization tooling were rewritten by Filip Borowiak.

## Purpose

Given a handful of an agent's past episodes in a 13x13 gridworld with four differently-valued goals, ToMnetF infers a character embedding and uses it, together with the current state, to predict:
- the agent's next action,
- which of the four goals it will end up consuming,
- its successor representation (discounted future state-occupancy) under three discount factors.

The directory holds five successive experiments (`experiment1` through `experiment5`), each a self-contained copy of the pipeline with an evolving agent type, environment configuration, and set of prediction heads.

## Architecture

### CharNet
Processes only past episodes (the current/query episode is not fed to CharNet). For each past episode:
1. `Conv2d(in=10, out=32, 3x3)` on every timestep of the episode (episodes with all-zero content are treated as padding and skipped)
2. A stack of `ResidualBlock`s (Conv-BN-ReLU, Conv-BN, skip-add, ReLU) — 5 blocks by default
3. Spatial average pooling per timestep, then a single-layer LSTM (hidden size 64) over the time axis, keeping the last hidden state
4. A linear layer to the character embedding size (`N_echar`, default 8)

Per-episode embeddings are averaged over the valid (non-padded) episodes in the batch to produce `e_char`. When `use_n_past` is disabled or no past episodes are supplied, a learned default embedding is broadcast instead.

### PredNet
Takes the current-state tensor (6 channels: wall, player, 4 goals) concatenated with `e_char` spatially broadcast to every grid cell, and runs a shared torso:
1. `Conv2d(6 + N_echar -> 32, 3x3)`
2. The same `ResidualBlock` stack used in CharNet (5 blocks by default)
3. `Conv2d(32 -> 32, 3x3)` + ReLU

From the torso, three heads branch off:
- **Action head**: global average pool -> 2 FC layers (32->32, ReLU) -> `Linear(32, 4)` logits over UP/RIGHT/DOWN/LEFT
- **Consumption head**: same pooled features -> `Linear(32, 4)` logits (sigmoid applied at the loss, one per goal)
- **SR head**: `Conv2d(32->32, 1x1)` + ReLU -> `Conv2d(32->3, 1x1)`, softmax independently over each of the 3 discount-factor channels (spatial 13x13 successor-representation maps)

```
past episodes ──► CharNet (Conv+ResBlocks per step ──► avg-pool ──► LSTM ──► FC) ──► e_char
                                                                                       │
current state (6ch) ──────────────────────────────────► concat, broadcast e_char ─────┘
                                                                    │
                                                              PredNet torso
                                                        (Conv + N ResBlocks + Conv)
                                                          │        │        │
                                                     action-head  cons-head  SR-head
                                                     (4 logits)  (4 logits) (3x13x13)
```

### Environment and agents
`scripts/environment.py` builds a `labmaze`-backed 13x13 gridworld with a wall layer, a player, and four goals (A-D). Two agent types generate trajectories, both in `scripts/experiment{N}/agents.py`:
- `AgentStar`: A* search toward the highest-value goal.
- `ValueAgent` (introduced in experiment5): value iteration over movement cost, wall-collision penalty, and goal rewards, with a temperature-scaled softmax policy for stochastic behavior.

## Differences from ToMnet

Both `ToMnetF_impl` and `../ToMnet_impl` share the same upstream origin (Nik-Kras/ToMnet-N) but diverge in the character/prediction network:

| | ToMnet (`../ToMnet_impl`) | ToMnetF (this directory) |
|---|---|---|
| CharNet spatial processing | MLP / small convnet per the paper's Figure 3 and Figure 5 specs | Full `ResidualBlock` stack (Conv-BN-ReLU x2 + skip) before pooling |
| Temporal aggregation | Sum of per-episode embeddings (paper's Figure 5 recipe) | LSTM over per-episode features, averaged across valid past episodes |
| Organization | Single flat `scripts/` directory, one experiment line reproducing paper figures 3 and 5 | Numbered `experiment1/` .. `experiment5/` packages, each with its own `config.py`, `agents.py`, `tomnet.py`, `generate.py`, `train.py`, `evaluate.py`, `visualize.py` |
| Agents | `RandomAgent` (Dirichlet policy), `GoalDirectedAgent` (value iteration) | `AgentStar` (A* planner) in experiments 1-4; `ValueAgent` (value iteration + softmax policy) added in experiment5 |
| Prediction targets | Action likelihood (Figure 3), action + consumption + SR (Figure 5 spec) | Action, consumption, and SR heads present from experiment2 onward; experiment1 predicts action only |

Both codebases implement all three loss terms described in the paper (action NLL, consumption BCE, SR cross-entropy); ToMnetF wires all three into a shared PredNet torso rather than a Figure 3/Figure 5 split.

## Files

```
ToMnetF_impl/
├── scripts/
│   ├── environment.py           # labmaze gridworld (shared across experiments)
│   ├── data_generation.py       # generic trajectory loading/tensorization (used by experiment1)
│   └── experiment{1..5}/
│       ├── config.py            # Config object: env, model, and training hyperparameters
│       ├── agents.py            # AgentStar and/or ValueAgent
│       ├── tomnet.py            # CharNet / PredNet / ToMnet definitions
│       ├── generate.py          # trajectory generation CLI (writes gameN.txt files)
│       ├── data_generation.py   # experiment-local trajectory loader (experiment3-5 only)
│       ├── train.py             # training loop, early stopping, checkpointing
│       ├── evaluate.py          # accuracy / cross-species evaluation CLI
│       └── visualize.py         # plotting CLI
├── shell/
│   └── run_exp{1..5}.sh         # per-experiment pipeline runner (data_generation/train/evaluate/visualize/all)
└── (data/, models/, result/, plots/, log/ are created at run time under each experiment's directories)
```

`experiment1` reuses `scripts/data_generation.py` for preprocessing; `experiment3`, `experiment4`, and `experiment5` each carry their own `data_generation.py` because they add goal-rank labels and/or the n_past sampling logic described below. `experiment2` carries its own copy alongside its `tomnet.py`.

## Running

Each experiment's shell script drives the full pipeline (checking for existing outputs and skipping completed steps):

```bash
bash shell/run_exp5.sh all               # data_generation -> test_data_generation -> train -> evaluate -> visualize
bash shell/run_exp5.sh data_generation
bash shell/run_exp5.sh train
bash shell/run_exp5.sh evaluate
bash shell/run_exp5.sh visualize
```

Manual invocation of the experiment5 pipeline (adjust `experiment5` to the target experiment number):

```bash
cd scripts/experiment5
python generate.py --n_games 20000 --save_dir ../../data/experiment5
python train.py --experiment_no 5 --epochs 200 --batch_size 512 --lr 1e-4 --device cuda:0
python evaluate.py --experiment_no 5 --device cuda:0
python visualize.py --plot_type all
```

`config.py` in each experiment directory holds the defaults (grid size, agent type, `N_GAMES`/`BATCH_SIZE`/`EPOCHS` env-var overrides, model hyperparameters); CLI flags on `generate.py`/`train.py`/`evaluate.py` override the corresponding `Config` fields when passed.

### Key hyperparameters (experiment5 defaults, `config.py`)

- Grid: 13x13, `max_moves`: 50, `time_step`: 20
- Model: `residual_blocks=5`, `e_char=8`, `out_channels=32`, input depth 10 (1 wall + 1 player + 4 goals + 4 action one-hots)
- Training: `batch_size=512`, `lr=1e-4`, `epochs=200`, early stopping (`patience=50`, `min_delta=0.001`, restores best weights)
- `n_past_min=0` / `n_past_max=4`: number of past episodes sampled per training example for the character embedding
- `rank_threshold`: how many top-ranked goals must match between two trajectories for one to be usable as a "past episode" source for the other, when past episodes are sampled from goal-rank labels within a batch (`train.py:generate_past_episodes_from_batch`)

## Outputs

Written under each experiment's `models/`, `result/`, and `plots/` directories (paths configured in `config.py`):

- **Models**: `exp{N}_best.pth` (best validation checkpoint), `exp{N}_final.pth` (final epoch)
- **Results**: `exp{N}_training_history.json`, `exp{N}_results.json`, `cross_species_evaluation_exp{N}.json`, `predictions.pkl`, `n_past_evaluation_results.json`
- **Plots**: `training_curves_exp{N}.png`, `confusion_matrix_exp{N}.png`, `action_likelihood_exp{N}.png`, `character_embeddings_by_goal_exp{N}.png`, `cross_species_results_exp{N}.png`, `accuracy_by_n_past.png`, `accuracy_heatmap_by_n_past.png`, plus per-experiment action/consumption/SR/past-episode training visualizations

## Notes and Limitations

- `experiment1` predicts action only (single-head `PredNet`); consumption and SR heads appear starting with `experiment2`.
- `experiment2`'s `CharNet` does not implement n_past-based character embedding aggregation: its `forward` takes a single trajectory, with no loop over a variable number of past episodes. The n_past mechanism (`n_past_min`/`n_past_max`, per-episode averaging) is introduced starting with `experiment3`.
- The five experiment directories are largely independent copies rather than a shared library — changes made to one (e.g. a bugfix in `tomnet.py`) do not automatically propagate to the others.
- `config.py` hardcodes `device = "cuda:3"` as the default in several experiments; override with `--device` for single-GPU or CPU machines.
- Requires `torch`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pandas`, and `dm-labmaze` (`pip install dm-labmaze` for the maze generator).
