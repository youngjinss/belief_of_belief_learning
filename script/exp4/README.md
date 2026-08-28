# Experiment 4 — AchieverBlocker Multi-Agent ToMnet

This experiment trains a single ToMnet on trajectories from a two-agent competitive environment: an Achiever that seeks a preferred key/door pair, and a Blocker that must infer the Achiever's goal from observed behavior and race to block it.

## Purpose

Experiments 1–3 model a single agent acting in an environment. Experiment 4 extends the ToMnet to a competitive two-agent setting where a correct theory of mind is not just descriptive but instrumentally useful: the Blocker's only way to win is to infer the Achiever's target door color before the Achiever reaches it. The same ToMnet is used to model both roles, with an added agent-classification head so the network also learns to distinguish "I'm watching an achiever" from "I'm watching a blocker" from behavior alone.

## Environment

The environment lives outside this directory, at `lib/env/gym_minigrid/envs/achiever_blocker.py` (`AchieverBlockerEnv`, registered as `MiniGrid-AchieverBlocker-{5x5,9x9,11x11}-v1`; exp4's default is 9x9). It is a `MiniGridEnv` subclass with 4 colored keys/doors (red, green, blue, yellow) and two agents that share full observability and cannot occupy the same cell.

**Actions.** The Achiever has 7 actions: up, right, down, left, stay, pickup, toggle. The Blocker has 6: up, right, down, left, stay, and `broken` (an explicit "attempt to block here" action). `pickup` and `toggle` are actually handled automatically on movement (see below); the Achiever's action space still reserves those two slots.

**Goal assignment.** Each episode randomizes a `preference` value per color (one random color is set to 1.0, the rest drawn uniform in [0, 1)) and a `cost` value per color (four values summing to 1.0). The color with the highest preference becomes `target_door_color` — the door the Achiever is trying to open and the Blocker is trying to block.

**Reward structure** (`AchieverBlockerEnv.step`):
- Auto key pickup (stepping onto a key): `+0.5` if the key matches `target_door_color`, otherwise `-cost[color]`.
- Auto door open (stepping onto a locked door while holding its key): `+preference[door_color]`.
- Achiever reaches the target door once it is open: `+10.0` to the Achiever, `-5.0` to the Blocker, episode terminates.
- Blocker plays `broken` while standing on the target door: `+1.0` to the Blocker, `-1.0` to the Achiever, episode terminates. Playing `broken` on the wrong door: `-1.0` to the Blocker, episode continues (no termination).
- Shaping reward: `+0.1` to the Blocker each step it stays within Chebyshev distance 1 of the target door's position.

## Architecture

`tomnet.py` defines a `ToMnet` that supports two architecture variants via `use_mentalnet`:
- `use_mentalnet=True` (default in `config.py`): the original three-stage pipeline, `CharNet → MentalNet → PredNet`.
- `use_mentalnet=False`: a two-stage pipeline, `CharNet → PredNet`, where the character embedding is broadcast spatially and concatenated directly with the current state (no recent-trajectory encoding step).

`CharNet` encodes past episodes into a character embedding (`n_echar`); `MentalNet` (when enabled) encodes the current episode's history into a mental-state embedding (`n_ement`) via a `ConvLSTM2d`; `PredNet` combines these with the current-state grid (a `ResidualBlock`-based CNN) to produce five outputs:
- `action_logits` — next-action distribution (achiever: 7-way, blocker: 6-way, sharing one head sized to the larger space)
- `goal_logits` — 4-way predicted target-door color
- `agent_logits` — 2-way: is this trajectory an Achiever (0) or a Blocker (1)?
- `consumption_logits` — binary, whether a key/door interaction occurs
- `sr_pred` — predicted successor representation over the grid

`ToMnetLoss` combines these five heads with independently weighted losses (`action_weight`, `goal_weight`, `agent_weight`, `consumption_weight`, `sr_weight`, all default to 1.0 in `config.py`'s `training_process_config`): cross-entropy for action/goal/agent, BCE-with-logits for consumption, and a KL-divergence term (`calculate_sr_loss_kl_divergence`, defined in `train.py`) for the SR prediction.

## Configuration

`config.py`'s `Config` class centralizes every setting used across the pipeline; scripts read it directly and only accept CLI overrides behind `--config_override` (see Running). Selected defaults:

```python
# Environment / agents
achiever_type = "value"       # "astar", "value", "random"
blocker_type  = "goal_direct" # "random", "goal_direct"
width, height = 9, 9
max_steps     = 500

# model_config
use_mentalnet = True   # False switches to the 2-stage CharNet→PredNet variant
residual_blocks = 5
n_echar, n_ement = 256, 256
achiever_action_space, blocker_action_space = 7, 6
goal_space = 4

# training_config
batch_size = 1024
epochs = 300
lr = 0.0001
device = "cuda:3"
use_parallel = True        # DataParallel across training_config["device_ids"]
use_amp = True              # mixed precision

# training_process_config (loss weights, all default 1.0)
action_weight = goal_weight = agent_weight = consumption_weight = sr_weight = 1.0
early_stopping_patience = 30
```

`Config.get_data_path(is_test=False)` derives the trajectory directory as `{save_dir}/{env_name}/{achiever_type}_{blocker_type}/[test/]`, and `Config.get_agent_pair_name()` returns the `{achiever_type}_{blocker_type}` string used throughout as a directory/label suffix.

## Files

```
script/exp4/
├── config.py              # Central Config class: env, agent, model, training,
│                           #   data-processing, evaluation settings
├── achievers.py            # Achiever policies: AStarAgent, ValueAgent, RandomAgent
├── blockers.py              # Blocker policies: GoalDirectAgent, RandomAgent
├── generate.py              # Multi-process self-play trajectory generation
├── data_generation.py       # DataGenerator: parses trajectory .txt files into
│                           #   achiever/blocker training samples
├── tomnet.py                # ToMnet model, ToMnetLoss, create_model()
├── train.py                 # Training loop, SR-loss utilities, checkpoint saving
├── evaluate.py               # Loads a checkpoint, scores it on held-out data
├── visualize.py              # Plots from training history / evaluation output
├── visualize_sr.py           # Standalone SR-parsing helpers for a trajectory file
├── simulate_trajectory.py    # Replays a saved trajectory .txt file (GameSimulation)
└── simulate_game.py          # Live single-agent rollout viewer (see Limitations)
```

### Achiever policies (`achievers.py`)
- **`AStarAgent`** — shortest-path planner: navigate to the preferred key, then to its door.
- **`ValueAgent`** — value-iteration planner with a movement cost, wall penalty, and temperature-controlled stochastic action selection (`config.achiever_configs["value"]`).
- **`RandomAgent`** — movement-probability-controlled random baseline.

### Blocker policies (`blockers.py`)
- **`GoalDirectAgent`** — stays put until the Achiever picks up its first key, infers `target_door_color` from that key's color, paths to the corresponding door, then plays `broken`.
- **`RandomAgent`** — samples uniformly from all 6 actions, including `broken`.

## Running

Every entry point built on `Config` (`generate.py`, `train.py`, `evaluate.py`, `visualize.py`) only applies its CLI flags when `--config_override` is passed — without it, the script runs with `config.py`'s defaults (`achiever_type="value"`, `blocker_type="goal_direct"`, `env_size="9x9"`) regardless of other flags given.

**1. Generate trajectory data**
```bash
python script/exp4/generate.py --config_override \
    --achiever_type value --blocker_type goal_direct --n_games 100000

python script/exp4/generate.py --config_override \
    --achiever_type astar --blocker_type random --n_games 2000 --test_data
```
Generation is parallelized with `multiprocessing.Pool` (`--n_processes`, default: CPU count − 1).

**2. Train**
```bash
python script/exp4/train.py --config_override \
    --save_dir ./results/exp4/ --batch_size 1024 --epochs 300 --lr 0.0001 --device cuda:0
```
`--data_dir` is optional; if omitted, `train_tomnet` derives it from `config.get_data_path()` (i.e. `./data/{env_name}/{achiever_type}_{blocker_type}/`). Multi-GPU (`DataParallel`) is available via `--use_parallel --device_ids 2 3` (also configurable through `training_config` in `config.py`).

**3. Evaluate**
```bash
python script/exp4/evaluate.py --config_override \
    --model_path ./results/exp4/best_model.pth --result_dir ./results/exp4/ --plot_type all
```

**4. Visualize**
```bash
python script/exp4/visualize.py --config_override \
    --result_dir ./results/exp4/ --plot_dir ./results/exp4/plots --plot_type all
```
`--plot_type` accepts `training`, `confusion`, `likelihood`, `embeddings`, `n_past`, or `all`.

**5. Replay a saved trajectory**
```bash
python script/exp4/simulate_trajectory.py --data_file data/MiniGrid-AchieverBlocker-9x9-v1/value_goal_direct/test0.txt --summary
```
Drop `--summary` to render the animation; add `--save_gif` to export it.

## Outputs

**Data** (`generate.py`, under `config.save_dir`, default `data/`):
```
data/MiniGrid-AchieverBlocker-9x9-v1/value_goal_direct/
├── test0.txt, test1.txt, ...      # one trajectory per file
└── test/                          # test-split trajectories (from --test_data)
    └── test0.txt, ...
```
Each trajectory yields two training samples once processed by `DataGenerator.process_directory` — one Achiever sample, one Blocker sample — cached to `processed_data_exp4.pkl` (train split) / `processed_test_data_exp4.pkl` (test split) inside the same directory so re-runs skip re-parsing.

**Training** (`train.py`, under `--save_dir`):
- `best_model.pth`, `final_model.pth` — model state dicts
- `training_history.json`, `training_history.png`
- `model_config.json`, `full_config.json`

**Evaluation** (`evaluate.py`, under `--result_dir`):
- `evaluation_results_exp4.json` — accuracy/precision/recall/F1 for action, goal, and agent-type predictions
- `predictions.pkl`
- `n_past_evaluation_results.json`, `action_likelihood_analysis.pkl`, `action_likelihood_stats.json`

## Notes and Limitations

- `simulate_game.py` does `from agents import AStarAgent, RandomAgent, ValueAgent`, but no `agents.py` module exists in this directory (the actual policy classes live in `achievers.py`/`blockers.py`). Running it currently raises `ModuleNotFoundError: No module named 'agents'`. It also only instantiates a single Achiever policy and calls `env.step()` with what appears to be a single action, while `AchieverBlockerEnv.step()` expects an `(achiever_action, blocker_action)` pair — this script looks like an unported carryover from an earlier single-agent experiment and is not currently usable for AchieverBlocker rollouts.
- `visualize_sr.py` has no `argparse` interface; its `if __name__ == "__main__":` block calls `analyze_test_file()` with a hardcoded absolute path from a different machine/repo layout. Its parsing functions (`parse_maze`, `parse_sr_data`, `analyze_test_file`) are usable when imported directly, but the file cannot be run as a CLI tool as-is.
- The `action_logits` head is sized for the larger of the two action spaces; Blocker samples (6-way) and Achiever samples (7-way) share this single head, so action-loss/accuracy figures should be interpreted per agent type rather than pooled, if pooled numbers are reported anywhere downstream.
