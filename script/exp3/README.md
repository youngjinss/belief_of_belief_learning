# Experiment 3 — KeyDoor Single-Agent ToMnet

Single-agent ToMnet (Theory of Mind network) on a 9x9 multi-colored KeyDoor gridworld: an observer model learns to infer an actor's goal preferences and predict its next action and successor representation from past and current trajectories.

## Purpose

The KeyDoor environment gives each episode a randomly sampled preference over four colored goals and a randomly sampled cost per door. A scripted actor (A*, value-iteration, or random) plays the episode according to those preferences. `exp3` trains a ToMnet observer to watch a handful of the actor's past episodes (its "character"), then predict the actor's action, goal, key-consumption, and spatial successor representation (SR) on a new, unseen episode — testing whether the network can infer latent preferences from behavior alone.

## Environment

- Custom MiniGrid environment `MiniGrid-KeyDoor-{size}-v0`, registered in `lib/env/gym_minigrid/` (this directory only consumes it, does not define it).
- Grid sizes are configurable (`3x3`, `5x5`, `9x9`, `11x11` via `config.env_variants`); the default and primary setting used throughout this experiment is **9x9**, `max_steps=500`.
- **4 colored keys** (red/A, green/B, blue/C, yellow/D) and **4 matching colored doors** (a, b, c, d). Stepping onto a key or an unlocked-matching door triggers automatic pickup / toggle — the agent does not need to issue explicit pickup/toggle actions for the intended sequence to work, though those actions exist in the action space.
- **7-dimensional action space**: `0=up, 1=right, 2=down, 3=left, 4=stay, 5=pickup, 6=toggle`.
- **Goal rewards**: per episode, `Config.generate_random_goal_rewards()` draws 4 values from `Uniform(0,1)` and forces the maximum to `1.0` (ToMnetF-style single dominant preference), or uses fixed `default_rewards` when `goal_reward_settings["use_random_rewards"]` is `False`.
- **Door costs**: per episode, `Config.generate_random_costs()` draws a random split of the 4 door costs that sums to `1.0`, clamped to `[min_cost, max_cost] = [0.05, 0.7]`, or uses fixed `default_costs`.
- **Successor representation (SR)**: computed per timestep at three discount factors, `gammas = [0.5, 0.9, 0.99]` (`config.sr_settings`), and stored per-trajectory alongside the standard action/state data.
- Text-format trajectory files encode the maze with a fixed symbol-to-integer mapping (`data_generation.DataReader.OBJECT_ENCODING`): `#`=wall, `-`=empty, `A-D`=keys, `a-d`=doors, `O`=agent.

## Architecture

Implemented in `tomnet.py`. The model supports two selectable architectures via `model_config["use_mentalnet"]`:

```
use_mentalnet=True  (default, 3-stage):
  past episodes ──► CharNet ──► character_embedding ──┐
                                                        ├──► PredNet ──► action_logits, goal_logits,
  recent trajectory ──► MentalNet ──► mental_state ────┘              consumption_logits, sr_pred
  current_state ─────────────────────────────────────────────────────────┘

use_mentalnet=False (2-stage, "experiment5-style", bypasses the mental-state bottleneck):
  past episodes ──► CharNet ──► character_embedding ──┐
                                                        ├──► PredNet ──► (same outputs)
  current_state ──────────────────────────────────────┘
```

- **`CharNet`**: convolutional residual blocks (`ResidualBlock`, default 5) over each past episode's per-timestep grid, followed by an LSTM (`ConvLSTM2d`/`LSTM` depending on branch) that aggregates across the episode and across `n_past` episodes into a fixed-size `character_embedding` (`n_echar`, default 128).
- **`MentalNet`**: only instantiated when `use_mentalnet=True`; consumes the "recent" trajectory and current-episode actions through residual/conv-LSTM blocks to produce a `mental_state` embedding (`n_ement`, default 128).
- **`PredNet`**: takes `character_embedding` (+ `mental_state` if present) and the current-state grid, and outputs four heads: action logits (7-way), goal logits (4-way), key-consumption logits (multi-label), and an SR prediction map.
- **`ToMnetLoss`**: weighted sum of `CrossEntropyLoss` (action, goal), `BCEWithLogitsLoss` (consumption), and a KL-divergence SR loss (`calculate_sr_loss_kl_divergence`, in `train.py`), with per-term weights `action_weight/goal_weight/consumption_weight/sr_weight` (all default 1.0).
- Input grids use 9 channels: 8 base game-state channels (walls, keys, doors, agent position, etc.) + 1 heading-direction channel for `CharNet`'s input; `PredNet`'s current-state input uses only the 8 base channels (`current_state_channels`).

## Files

```
script/exp3/
├── config.py                # Config class: all environment/model/training/eval parameters
├── agents.py                 # Scripted actors: AStarAgent, ValueAgent, RandomAgent
├── data_generation.py        # DataReader: parses trajectory text files into tensors
├── generate.py                # Generates trajectories by rolling out an agent in the env
├── train.py                   # Data prep, training loop, ToMnet loss, checkpointing
├── evaluate.py                 # Loads a trained model, computes accuracy/F1/N_past metrics
├── visualize.py                # Training curves, confusion matrices, embedding plots (t-SNE/PCA)
├── visualize_sr.py             # Standalone SR-heatmap plotting helpers (no CLI, see Notes)
├── simulate_game.py            # Live GUI rollout of one agent in the environment
├── simulate_trajectory.py      # Replays a saved trajectory file and renders/animates it
└── tomnet.py                    # Model architecture (CharNet, MentalNet, PredNet, ToMnet, ToMnetLoss)
```

## Running

All commands assume the repo root as the working directory.

### 1. Data generation (`generate.py`)

```bash
# Generate training data with the config defaults (agent_type=value, 9x9, 100k games)
python script/exp3/generate.py

# Override parameters — note --config_override is required for the other flags to take effect
python script/exp3/generate.py --config_override --n_games 2000 --agent_type astar --env_size 9x9

# Generate test data (written to a test/ subdirectory)
python script/exp3/generate.py --config_override --test_data --n_games 500
```
`--random_seed` and `--n_processes` (parallel workers, default CPU count − 1) apply without `--config_override`. Output path is auto-derived as `data/<env_name>/<agent_type>/` (or `.../test/`).

### 2. Simulation and inspection

```bash
# Live GUI rollout of a single agent (args apply unconditionally, no --config_override needed)
python script/exp3/simulate_game.py --agent_type value --episodes 3 --pause 0.5

# Replay a saved trajectory file
python script/exp3/simulate_trajectory.py --data_file data/MiniGrid-KeyDoor-9x9-v0/value/test0.txt
```
`visualize_sr.py` has no CLI — see Notes and Limitations.

### 3. Training (`train.py`)

```bash
# Train with config defaults (--save_dir has its own default and does not require --config_override)
python script/exp3/train.py --save_dir ./results/exp3/

# Override hyperparameters (requires --config_override for these to be applied)
python script/exp3/train.py --config_override \
    --epochs 100 --batch_size 1024 --lr 0.0001 --device cuda:0 --save_dir ./results/exp3/
```
`--data_dir` and `--save_dir` are read directly from `args` (no `--config_override` needed); every other flag (`--epochs`, `--batch_size`, `--lr`, `--device`, `--residual_blocks`, `--n_echar`, `--n_ement`, `--action_weight`, etc.) is only applied to the config when `--config_override` is also passed — without it, `train.py` silently trains with `config.py`'s defaults.

### 4. Evaluation and visualization

```bash
python script/exp3/evaluate.py --config_override \
    --model_path ./results/exp3/best_model.pth --result_dir ./results/exp3/ --plot_type all

python script/exp3/visualize.py --config_override \
    --result_dir ./results/exp3/ --plot_dir ./results/exp3/plots --plot_type all
```
`evaluate.py --plot_type` accepts `basic|embeddings|n_past|all`; `visualize.py --plot_type` accepts `training|confusion|likelihood|embeddings|n_past|all`. If `--model_path` is omitted, evaluation looks for `<config.model_dir>/best_model.pth`.

## Configuration

All parameters live in `config.py`'s `Config` class, grouped into dict attributes retrieved via getters (`get_model_config()`, `get_training_config()`, `get_data_config()`, etc.). Selected defaults:

```python
agent_type = "value"           # "astar" | "random" | "value"
width = height = 9
max_steps = 500

model_config = {
    "use_mentalnet": True,     # True: CharNet→MentalNet→PredNet; False: CharNet→PredNet
    "residual_blocks": 5,
    "n_echar": 128, "n_ement": 128,
    "action_space": 7, "goal_space": 4,
    "channels_in": 9, "current_state_channels": 8,
}

training_config = {
    "batch_size": 1024, "epochs": 200, "lr": 0.0001,
    "device": "cuda:3", "optimizer": "adam",
}

data_config = {
    "max_n_past": 1,            # past episodes fed to CharNet
    "rank_threshold": 4,        # top-N goal ranks considered a "match" for character labeling
}
```
`agent_configs` also holds per-agent hyperparameters used by `ValueAgent` (`movement_cost`, `wall_penalty`, `gamma`, `temperature`), including `value_deterministic`/`value_stochastic` presets (both instantiate the same `ValueAgent` class with `temperature=0.0` or `0.5`). Any script called with `--config_override` applies matching CLI flags on top of these defaults via `Config.update_from_args()`.

## Agents

`agents.py` implements three scripted actors used to generate training/test trajectories:

- **`AStarAgent`** — deterministic A* pathfinding; first collects the target key, then navigates to the matching door (`strategy_phase`: `collect_key` → `open_door`). No stochasticity.
- **`ValueAgent`** — value iteration over the grid with `movement_cost`, `wall_penalty`, and `gamma` shaping the value function, and a **temperature-controlled stochastic policy** for action sampling. Same two-phase key-then-door strategy as `AStarAgent`. This is the default agent for data generation (`config.agent_type = "value"`).
- **`RandomAgent`** — samples uniformly among movement actions (`up/right/down/left/stay`) with probability `movement_prob` (default `0.9` in the class, `0.8` via `agent_configs["random"]`) and among `pickup/toggle` otherwise; no strategic reasoning.

All three expose `get_action(obs)` / `update_observation(obs)` / `reset()` and rely on the environment's automatic key-pickup and door-toggle behavior when the agent steps onto the relevant tile.

## Outputs

`train.py` writes to `save_dir` (flat, no auto-generated timestamp subfolder — pass a distinct `--save_dir` per run to avoid overwriting):
- `best_model.pth` — state dict at the best validation loss (checkpointed during training)
- `final_model.pth` — state dict at the end of training
- `training_history.json` — per-epoch losses/metrics
- `model_config.json`, `full_config.json` — the config used, for reproducibility and for `evaluate.py`/`visualize.py` to reload model hyperparameters
- `data_statistics.json` — dataset statistics gathered during training

`evaluate.py` writes to `result_dir`:
- `evaluation_results_exp<N>.json` — accuracy/F1/precision/recall and per-action accuracy
- `n_past_evaluation_results.json` — accuracy as a function of the number of past episodes available to `CharNet`
- `action_likelihood_stats.json` — probability-distribution analysis over predicted actions
- prediction/plot artifacts under the given output directories when `plot_type` requests them

`generate.py` writes trajectory text files (`test0.txt`, `test1.txt`, …) plus a cached `processed_data_exp3.pkl` (or `processed_test_data_exp3.pkl` under `test/`) to `data/<env_name>/<agent_type>/`.

## Notes and Limitations

- `visualize_sr.py` has no `argparse` CLI; its `if __name__ == "__main__":` block hardcodes an absolute path (`/Users/youngjins/.../belief_trading/data/exp3/test98.txt`, from a different, earlier repo) and calls `analyze_test_file()` on it directly. Its plotting functions (`parse_maze`, `parse_sr_data`, etc.) are usable as a library from other scripts, but the file is not runnable as-is without editing that path.
- Most scripts gate CLI overrides behind `--config_override`: without that flag, flags other than each script's directly-read path arguments (e.g. `train.py --save_dir`/`--data_dir`, `evaluate.py --model_path`) are silently ignored and `config.py`'s defaults are used instead.
- `training_config["device"]` defaults to `"cuda:3"` — adjust via `--device` (with `--config_override`) for machines without that GPU index.
- The KeyDoor environment itself (`MiniGrid-KeyDoor-{size}-v0`) is defined outside this directory, under `lib/env/gym_minigrid/`.
