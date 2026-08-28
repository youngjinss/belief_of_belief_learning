# Experiment 5 — Enhanced Multi-Agent ToMnet

A competitive Achiever/Blocker MiniGrid environment paired with a ToMnet-style network that predicts both agents' actions, goals, and successor representations from observed trajectories.

## Purpose

An **Achiever** agent navigates a grid to collect a preferred key and open the matching door, while a **Blocker** agent observes the achiever's behavior, infers its goal, and tries to physically block the target door first. This adversarial setup requires the Blocker (and the ToMnet trained on the resulting data) to perform theory-of-mind inference: predicting what the achiever wants from partial observations of its movement.

Compared to earlier experiments, exp5 adds:
- A configurable **2-stage or 3-stage** ToMnet architecture (`use_mentalnet` flag)
- Joint prediction of action, goal, agent identity (achiever vs. blocker), behavior type (Level-0 vs. Level-1), consumption, and successor representation (SR)
- Vectorized SR computation and multiprocessing-based trajectory generation for throughput

## Environment

The environment is `MiniGrid-AchieverBlocker-{size}-v1`, defaulting to a 9x9 grid (`max_steps=50`; 5x5 and 11x11 variants are also defined in `config.py`).

- **Turn-based**, fully observable by default (`observability="full"`); both agents see each other's positions and states.
- **Achiever** actions (7): up, right, down, left, stay, pickup, toggle.
- **Blocker** actions (6): up, right, down, left, stay, broken.
- Agents cannot occupy the same cell.
- The episode ends when the achiever reaches and opens its target door, or when the blocker uses "broken" on the correct target door. Breaking the wrong door gives a penalty but the episode continues, so a blocker can make multiple attempts.
- Each of the 4 colored keys/doors carries a randomly generated preference reward and pickup cost per episode (`goal_reward_settings`, `cost_settings` in `config.py`), following the ToMnetF reward-randomization pattern.

### Agent implementations

**Achievers** (`achievers.py`, `value_agent.py`):
- `AStarAgent` (`astar`) — deterministic shortest-path planner: collect key, then reach door.
- `Level0ValueAchiever` (`lv0va`) — vectorized value iteration with a stochastic, temperature-controlled policy; collects the preferred key then opens its door.
- `Level1ValueAchiever` (`lv1va`) — extends Level-0 with deception: collects a decoy key first and switches to its true target if the blocker moves toward any door.
- `random` — `AchieverRandomAgent`, biased random movement.
- The `value` CLI choice reuses `Level0ValueAchiever` with a different cost profile from `achiever_configs["value"]` (there is no separate `ValueAgent` class); `BaseValueAgent` in `value_agent.py` holds the shared value-iteration logic both Level-0/1 achievers and blockers inherit.

**Blockers** (`blockers.py`):
- `Level0ValueBlocker` (`lv0vb`) — value-iteration pathfinding to a randomly chosen door; retries other doors on a miss.
- `Level1ValueBlocker` (`lv1vb`) — tracks the achiever's distance to each key over time to predict its target door before the key is collected, then moves to block it.
- `RandomlySelectedAgent` (`randomly_selected`) — Level-0-style reasoning: BFS pathfinding to a randomly picked door, cycling through untried doors on misses.
- `RuleBasedAgent` (`rule_based`) — Level-1-style reasoning: blocks a random door first, then infers the target from keys it has observed the achiever collect.
- `GoalDirectAgent` (`goal_direct`) — waits until the achiever picks up a key, then infers and moves to block the corresponding door.
- `RandomAgent` (`random`) — baseline, no inference.

## Architecture

`tomnet.py` implements a single `ToMnet` module that supports two architecture modes, selected by `model_config["use_mentalnet"]`:

```
use_mentalnet=True  (default): CharNet -> MentalNet -> PredNet   (3-stage, with a mental-state bottleneck)
use_mentalnet=False:            CharNet -> PredNet                (2-stage, MentalNet bypassed)
```

- **CharNet**: consumes `n_past` past episode trajectories (residual conv blocks + an LSTM over time) and produces a character embedding `e_char` summarizing an agent's persistent behavioral style.
- **MentalNet** (3-stage only): a `ConvLSTM2d`-based recurrent module that consumes the current episode's recent trajectory and actions to produce a spatial mental-state embedding `e_ment`, conditioned on `e_char`.
- **PredNet**: takes the current observation plus (`e_ment`, `e_char`) — or, in 2-stage mode, `e_char` mixed directly with the current state via `forward_direct` — and outputs five heads:
  - action logits (`achiever_action_space`=7 / `blocker_action_space`=6, model uses the max, 7)
  - goal logits (4 colored doors)
  - agent-type logits (achiever vs. blocker, 2-way)
  - behavior-type logits (Level-0 vs. Level-1 reasoning, 2-way)
  - consumption logits (8-dim binary vector: 4 keys + 4 doors)
  - a successor-representation map, output per discount factor (γ = 0.5, 0.9, 0.99)

Input maze tensors have 9 channels: 8 game-state channels (walls, keys, doors, both agents' positions/state) plus 1 heading-direction channel; PredNet's current-state input drops the heading channel (8 channels). `ToMnetLoss` combines cross-entropy losses for action/goal/agent/type, BCE for consumption, and KL divergence for SR, weighted by `training_process_config`.

## Files

```
script/exp5/
├── config.py               # Central Config class: env, agent, model, training, data settings
├── achievers.py             # Achiever agents: AStarAgent, Level0/1ValueAchiever, RandomAgent
├── blockers.py               # Blocker agents: Level0/1ValueBlocker, RandomlySelected, RuleBased, GoalDirect, Random
├── value_agent.py           # BaseValueAgent: shared vectorized value-iteration logic
├── generate.py               # Parallel multi-agent trajectory generation
├── data_generation.py       # Trajectory-file parsing and sample construction (regex-based, multiprocessing worker)
├── utils.py                  # Data loading/slicing, SR loss, memory-mapped datasets, training loop helpers
├── tomnet.py                  # ToMnet model (CharNet/MentalNet/PredNet), ToMnetLoss
├── train.py                    # Training entry point (always trains on combined data from all configured agent-type combinations)
├── evaluate.py                # Evaluation: accuracy/F1/confusion metrics, N_past sweep, action-likelihood analysis
├── visualize.py                # Plots: training curves, confusion matrices, action likelihood, character-embedding PCA/t-SNE
├── simulate_game.py            # Interactive/GUI game simulation between an achiever and blocker
├── simulate_trajectory.py      # Trajectory playback / GIF export
└── results/                     # Default output location for trained models and plots
```

## Running

All entry-point scripts read defaults from `Config` in `config.py`; passing `--config_override` is required for most CLI flags (e.g. `--achiever_type`, `--batch_size`) to actually override the config — without it, only the flags each script always honors (like `--test_data`, `--save_dir`, `--debug`) take effect.

### 1. Data generation

```bash
# Generate data using the current config.py agent-type settings
python script/exp5/generate.py

# Override achiever/blocker type for this run
python script/exp5/generate.py --config_override --achiever_type lv0va --blocker_type lv0vb

# Generate held-out test data, with parallel worker control
python script/exp5/generate.py --config_override --achiever_type lv1va --blocker_type lv1vb --test_data --n_processes 8
```

The number of games generated per achiever/blocker type combination is controlled by `Config.achiever_types` / `Config.blocker_types` (dicts mapping type name to game count), not by a CLI flag.

### 2. Game / trajectory simulation

```bash
# Watch one episode with rendering
python script/exp5/simulate_game.py --render

# Save an animated GIF of several episodes
python script/exp5/simulate_trajectory.py --save_gif

# Visualize successor-representation heatmaps from a saved trajectory file
python beliefrl/viz/sr.py data/MiniGrid-AchieverBlocker-9x9-v1/lv0va_lv0vb/test0.txt
```

### 3. Training

```bash
# Train with config.py defaults (3-stage ToMnet, combined data across all configured agent types)
python script/exp5/train.py --save_dir ./results/exp5

# Override training hyperparameters
python script/exp5/train.py --config_override --epochs 300 --batch_size 512 --lr 0.0001 --device cuda:0 --save_dir ./results/exp5
```

Training always aggregates data across every achiever-type/blocker-type combination present in `Config.achiever_types` × `Config.blocker_types` — there is no per-pair training mode.

### 4. Evaluation

```bash
python script/exp5/evaluate.py --model_path ./results/exp5/best_model.pth --result_dir ./results/exp5

python script/exp5/visualize.py --result_dir ./results/exp5 --plot_dir ./results/exp5/plots --plot_type all
```

`visualize.py --plot_type` accepts `training`, `confusion`, `likelihood`, `embeddings`, `n_past`, or `all`.

## Outputs

`train.py` writes directly into `--save_dir` (default `./results/exp5`), with no automatic per-run timestamp subdirectory:
- `best_model.pth`, `final_model.pth`, periodic `checkpoint_epoch_N.pth`
- `training_history.json`, `model_config.json`, `full_config.json`
- Training-curve plots via `save_training_plots`

`evaluate.py` writes metrics and, when `--save_predictions` is set, `predictions.pkl` (predictions, targets, probabilities, metrics) under `--result_dir`.

`visualize.py` writes PNG plots (training curves, confusion matrices, action-likelihood, PCA/t-SNE character-embedding plots) under `--plot_dir`.

Generated trajectory data is written under `data/MiniGrid-AchieverBlocker-9x9-v1/{achiever_type}_{blocker_type}/`, with per-episode `testN.txt` trajectory files, a cached `processed_data_exp5_{achiever_type}_{blocker_type}.pkl`, and a `test/` subdirectory holding the equivalent held-out split. Each trajectory file records, per timestep: the maze grid, both agents' actions and positions, an 8-dim consumption vector (4 keys + 4 doors), the blocker's inferred goal, and SR values for γ ∈ {0.5, 0.9, 0.99}.


## Shared core

Code that was byte-identical across experiments now lives in
[`beliefrl`](../../beliefrl): the config accessors (`Config` subclasses
`beliefrl.config.BaseConfig`), trajectory-generation helpers, dataset loading,
seeding, early stopping, epoch reporting, and SR plotting. This directory keeps
only what is specific to this experiment. Existing imports such as
`from utils import set_seed` still work — the shared names are re-exported here.

Run `pytest` from the repo root to check every experiment against the golden
recordings made before the refactor.

## Notes and Limitations

- `use_mentalnet` defaults to `True` (3-stage CharNet→MentalNet→PredNet); the 2-stage `forward_direct` path exists in the same model and is toggled by setting `use_mentalnet=False` in `config.py`.
- `generate.py` has no `--n_games` flag; the per-type episode count is fixed by `Config.achiever_types` / `Config.blocker_types` values in `config.py`.
- `train.py` has no `--agent_weight` (or other individual loss-weight) CLI flag beyond `--action_weight` and `--goal_weight`; other loss weights (`agent_weight`, `type_weight`, `consumption_weight`, `sr_weight`) are set only through `training_process_config` in `config.py`.
