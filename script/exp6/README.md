# Experiment 6 — Unified Single/Multi-Agent Framework

Experiment 6 runs the KeyDoor (single-agent) and AchieverBlocker (multi-agent) key-and-door tasks through one ToMnet codebase, switching between them purely by whether `Config.blocker_types` is empty.

## Purpose

Exp6 was built on top of the exp5 multi-agent codebase so that single-agent and multi-agent Theory-of-Mind experiments can share one implementation instead of two. The goals are:

1. **Unified framework** — switch between single-agent and multi-agent experiments through configuration, not separate codebases.
2. **Direct comparison** — the same ToMnet architecture, training loop, evaluation, and plotting code run against both environments, so results are comparable.
3. **Code reuse** — Level-k reasoning agents, successor-representation (SR) computation, and character/type prediction from the multi-agent pipeline are reused as-is for the single-agent case.

## Environment

Two `MiniGridEnv` subclasses back the two modes, both defined under `lib/env/gym_minigrid/envs/`:

- **`KeyDoorEnv`** (`keydoor.py`) — single agent on an `N×N` grid with 4 colored keys and 4 colored doors. The agent has a `preference` (reward) and `cost` per door color (defaults: `{red: 1.0, green: 0.8, blue: 0.6, yellow: 0.4}` preference, `{red: 0.1, green: 0.2, blue: 0.3, yellow: 0.4}` cost) and must collect a key of a color to open the matching door.
- **`AchieverBlockerEnv`** (`achiever_blocker.py`, instantiated per size as `AchieverBlocker5x5Env` / `AchieverBlocker9x9Env` / `AchieverBlocker11x11Env`) — two agents on the same key/door grid. The achiever pursues its preferred door; the blocker tries to infer the achiever's preference and block door access. Both agents have full observability and cannot occupy the same cell.

Supported grid sizes are `5x5`, `9x9`, and `11x11` — these are the only sizes wired into `generate.py`'s environment-selection tables, even though `config.env_variants` also defines a `3x3` preset (see Notes).

Agent policies, all implemented in `achievers.py` / `blockers.py`:

| Role | Types | Notes |
|---|---|---|
| Achiever | `astar`, `random`, `value` (→ `lv0va`/`lv1va`) | `lv0va`/`lv1va` map to `Level0ValueAchiever`/`Level1ValueAchiever`, value-iteration agents with 0- and 1-step reasoning. |
| Blocker | `random`, `goal_direct`, `randomly_selected`, `rule_based`, `lv0vb`/`lv1vb` | `lv0vb`/`lv1vb` map to `Level0ValueBlocker`/`Level1ValueBlocker`, which reason about the achiever's inferred preference to choose which door to block. |

`value_agent.py` holds a shared `BaseValueAgent` value-iteration base class used by both the value-based achievers and blockers.

## Architecture

### Mode switch

`Config.is_single_agent_mode()` (`config.py`) returns `True` exactly when `blocker_types` is empty:

```python
def is_single_agent_mode(self):
    return not self.blocker_types or len(self.blocker_types) == 0
```

Everything else derives from that one check:

- `get_env_name()` returns `MiniGrid-KeyDoor-{size}-v1` in single-agent mode, `MiniGrid-AchieverBlocker-{size}-v1` otherwise.
- `get_agent_pair_name(achiever_type, blocker_type)` collapses to just `achiever_type` in single-agent mode, or `{achiever_type}_{blocker_type}` in multi-agent mode; `get_data_path()` / `get_training_data_path()` build on top of this.
- `generate.py` re-derives the same condition per worker process (`blocker_type is None or len(blocker_types) == 0`) to decide whether to construct a `KeyDoorEnv` or the size-specific `AchieverBlockerNxNEnv`, and to handle the resulting scalar (single-agent) vs. dict-keyed (multi-agent) reward.
- `data_generation.py` parses trajectory text files with two regexes — a multi-agent pattern `[x1,y1][x2,y2] : action1,action2 : interaction1,interaction2` and a single-agent pattern `[x,y] : action : interaction` — and tags each parsed sample `is_single_agent`; a blocker training sample is only built when that flag is false.
- `train.py`, `evaluate.py`, and `visualize.py` are unmodified between modes: they resolve data paths and the env name through `config`, and downstream code (model, loss, plotting) is agnostic to which environment produced the tensors.

### Trajectory tensor

Grid state is encoded as a 10-channel tensor (`config.model_config["channels_in"] = 10`, `config.data_config["maze_depth"] = 10`):

- Channels 0–7: static game state (walls, keys, doors, door/key colors).
- Channel 8: position of the agent whose action is being predicted ("self").
- Channel 9: opponent position — held at all-zero in single-agent mode, since there is no second agent.

### ToMnet model (`tomnet.py`)

```
past trajectories ──► CharNet ──► character embedding ─┐
                                                         ├─► PredNet ──► action / goal / SR predictions
current-state tensor ──────────────────────────────────┘
       (optionally routed through MentalNet first, if use_mentalnet=True)
```

- `CharNet` — convolutional + LSTM encoder that turns a batch of past episodes into a character embedding.
- `MentalNet` — optional convolutional-LSTM ("ConvLSTM2d") stage that turns the character embedding plus current state into a mental-state embedding.
- `PredNet` — takes the current-state tensor and character embedding (plus mental embedding, when enabled) through residual conv blocks and predicts next action, goal, consumption, and a successor-representation (SR) map.
- `ToMnet.use_mentalnet` switches between two supported architectures: `False` uses the exp5-style two-stage `CharNet → PredNet` path (bypassing the mental-state bottleneck); `True` uses the original three-stage `CharNet → MentalNet → PredNet` path.
- `ToMnetLoss` sums weighted action, goal, agent-identity, blocker-type, consumption, and SR (KL-divergence) losses, all configurable via `config.training_process_config`.

The same `ToMnet` / `ToMnetLoss` classes and the same `train_tomnet()` loop run against both single- and multi-agent data — agent-identity and blocker-type prediction simply have no positive class to learn from in single-agent mode.

## Files

| File | Purpose |
|---|---|
| `config.py` | `Config` class: environment/agent/model/training hyperparameters, mode detection (`is_single_agent_mode`), and data-path helpers. |
| `generate.py` | Parallel trajectory generation; picks `KeyDoorEnv` or the size-specific `AchieverBlockerNxNEnv` based on the config. |
| `data_generation.py` | `DataGenerator`: parses raw trajectory `.txt` files into ToMnet training tensors/samples; usable standalone (`--data_dir`/`--output_dir`) or imported by `train.py` via `utils.load_chunked_data_for_training`. |
| `tomnet.py` | Model architecture — `ResidualBlock`, `LSTM`, `CharNet`, `ConvLSTM2d`, `MentalNet`, `PredNet`, `ToMnet`, `ToMnetLoss`. |
| `train.py` | Training loop, checkpointing (`best_model.pth`, periodic `checkpoint_epoch_*.pth`), history/config logging, final model save. |
| `evaluate.py` | Loads a trained model and computes accuracy, an n_past sweep, character/mental embedding extraction, and action-likelihood analysis. |
| `visualize.py` | Plots training curves, confusion matrices, action likelihood, and character/mental embeddings (agent-based, goal-based, type-based). |
| `visualize_sr.py` | Standalone script to render successor-representation heatmaps from a maze text dump. |
| `achievers.py` | Achiever policies: `AStarAgent`, `Level0ValueAchiever`, `Level1ValueAchiever`, `RandomAgent`. |
| `blockers.py` | Blocker policies: `RandomAgent`, `RandomlySelectedAgent`, `GoalDirectAgent`, `Level0ValueBlocker`, `Level1ValueBlocker`, `RuleBasedAgent`. |
| `value_agent.py` | `BaseValueAgent` — shared value-iteration logic used by the value-based achievers and blockers. |
| `simulate_game.py` / `simulate_trajectory.py` | CLI tools to run and render (GIF/step-through) a single episode. |
| `utils.py` | Shared helpers consolidated from `train.py`/`evaluate.py`/`visualize.py` — dataset loading, SR loss, plotting, memory/parallel-processing utilities. |
| `test_integration.py` | Smoke test that exercises the single-agent and multi-agent config paths. |
| `script/tools/` | `analyze_interactions.py`, `find_long_trajectories.py` — ad-hoc data-inspection scripts, shared by every experiment. |
| `shell/exp6/run_exp6.sh` | Pipeline driver (data generation → training → evaluation → visualization); lives outside this directory but is exp6's usual entry point. |

## Running

Mode is selected by `config.blocker_types`: leave it as `{}` for single-agent (KeyDoor), or populate it (e.g. `{"lv0vb": 1000}`) for multi-agent (AchieverBlocker). The same switch is available from the command line via `generate.py --config_override --blocker_type none|<type>`.

```bash
# Generate trajectory data (mode follows config.py, or override on the CLI)
python script/exp6/generate.py --config_override \
    --achiever_type lv0va --blocker_type none --env_size 9x9

python script/exp6/generate.py --config_override \
    --achiever_type lv0va --blocker_type lv0vb --env_size 9x9

# Generate held-out test data
python script/exp6/generate.py --config_override --test_data

# Train (data path/env resolved automatically from config)
python script/exp6/train.py --config_override --save_dir results/exp6/<run>

# Evaluate a trained model
python script/exp6/evaluate.py --config_override \
    --model_path results/exp6/<run>/best_model.pth --plot_type all

# Plot results
python script/exp6/visualize.py --config_override \
    --result_dir results/exp6/<run> --plot_dir results/exp6/<run>/plots --plot_type all
```

Full pipeline via the shell driver:

```bash
bash shell/exp6/run_exp6.sh all      # data generation + test data + train + evaluate + visualize
bash shell/exp6/run_exp6.sh debug    # same pipeline at reduced scale (config.enable_debug_mode())
bash shell/exp6/run_exp6.sh train    # run a single stage
```

`run_exp6.sh` resolves data/result/log paths from `config.py` (`get_data_paths()`), skips stages whose outputs already exist, and logs each stage to `log/exp6/<timestamp>/*.log`.

## Outputs

- **Trajectory data**: `data/{env_name}/{agent_combination}/test*.txt`, where `env_name` is `MiniGrid-KeyDoor-{size}-v1` or `MiniGrid-AchieverBlocker-{size}-v1`, and `agent_combination` is `{achiever_type}` (single-agent) or `{achiever_type}_{blocker_type}` (multi-agent). Test data lands under a `test/` subdirectory.
- **Training results**: `results/exp6/<run>/` — `best_model.pth`, `checkpoint_epoch_*.pth`, `final_model.pth`, `training_history.json`, `model_config.json`, `full_config.json`, plus training-curve plots saved during training.
- **Evaluation/visualization**: plots and metrics written under `results/exp6/<run>/plots/` (confusion matrices, action-likelihood, character/mental embedding scatterplots, n_past accuracy curves).
- **Logs**: `log/exp6/<timestamp>/{execution,train_data_generation,test_data_generation,training,evaluation,visualization}.log` when run through `run_exp6.sh`.

## Notes and Limitations

- `config.env_variants` defines a `3x3` grid preset, but `generate.py`'s size-to-environment lookup tables (for both `KeyDoorEnv` and the `AchieverBlockerNxNEnv` classes) only cover `5x5`, `9x9`, and `11x11` — `3x3` is not currently generatable.
- Default loss weights (`training_process_config["agent_weight"] = 0`, `["type_weight"] = 0`) zero out the agent-identity and blocker-type prediction terms; action, goal, consumption, and SR losses are the ones trained with a nonzero weight out of the box.
- Single-agent runs still compute an (all-zero) opponent-position channel and pass through the agent-identity/blocker-type prediction heads in `ToMnet`/`ToMnetLoss` — the architecture is shared rather than pruned for single-agent mode.
