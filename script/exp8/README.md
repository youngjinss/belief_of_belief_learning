# Experiment 8 — Second-Order Belief (Modular Agents)

exp8 extends the ToMnet architecture used in earlier experiments with a second-order belief
embedding (`e_opp2`), computed by a dedicated `SecondBeliefNet` and fused into prediction via
cross-attention. It also reorganized the hand-coded game agents (achiever/blocker) into a
modular per-agent package, which has since moved into the shared `beliefrl` core.

## Purpose

1. **Second-Order Belief Modeling** — `SecondBeliefNet` produces `e_opp2`, an embedding meant to
   capture what an agent believes about an opponent's beliefs, using the opponent's recent
   trajectory plus the agent's own mental-state embedding.
2. **Three-Embedding Prediction** — `CrossAttentionModule` combines `e_char`, `e_mental`, and
   `e_opp2` via multi-head attention before the shared prediction head.
3. **Opponent Trajectory Integration** — data generation extracts each agent's recent trajectory
   from the *opponent's* perspective so `SecondBeliefNet` has real opponent behavior to condition on.
4. **Modular Agent Architecture** — the achiever/blocker heuristic and value-based policies used
   to generate training data are one file per agent type rather than monolithic
   `achievers.py` / `blockers.py` files. They now live in `beliefrl/agents/`, shared with the
   other experiments.

## Environment

Games run on `AchieverBlocker{5x5,9x9,11x11}EnvV2` (in `lib/env/gym_minigrid/envs/`), a
two-agent competitive MiniGrid variant:

- **Achiever** tries to reach a preferred key/door combination based on a private preference and
  cost vector over four door colors (red, green, blue, yellow).
- **Blocker** tries to infer the achiever's target and block the corresponding door before the
  achiever gets there.
- Both agents support `observability="full"` or `"partial"` (a `partial_view_size`-wide window
  around the agent). exp8's default `Config` runs partial observation on a 9x9 grid.
- `KeyDoor{5x5,9x9,11x11}EnvV2` is also available for single-agent (achiever-only) data.

State tensors have 10 channels: 8 channels for maze/game elements (walls, keys, doors, etc.)
plus a self-position channel and an opponent-position channel (zeroed out in single-agent mode).

## Architecture

`beliefrl/model/tomnet.py` defines the network stack. `ToMnet` composes up to four sub-networks depending on
`use_mentalnet` / `use_second_belief`:

```
Past Episodes ──────────► CharNet ─────────► e_char ─┐
                                                       │
Recent Self Traj+Actions ► MentalNet ────► e_mental ──┼─► CrossAttentionModule ─► PredNet ─► predictions
                                              │        │        (multi-head attn,
                                              ▼        │         query = current state)
                              Opponent Traj+Actions    │
                                    │                  │
                                    ▼                  │
                              SecondBeliefNet ──► e_opp2┘
```

- **CharNet** — processes up to `max_n_past` past episodes (conv + residual blocks + LSTM),
  averaged over valid (non-masked) episodes, into `e_char` (`n_echar`-dim vector).
- **MentalNet** — processes the agent's own recent trajectory (states + spatialized actions)
  through conv/residual/ConvLSTM layers into a *spatial* `e_mental` (`n_ement` channels × H × W).
  Only built when `use_mentalnet=True`.
- **SecondBeliefNet** — mirrors MentalNet's state+action processing pipeline on the opponent's
  recent trajectory, then fuses the resulting features with `e_mental` (concat + conv fusion
  layer) before a ConvLSTM and global average pool produce the vector `e_opp2` (`n_eopp2`-dim).
  Only built when `use_second_belief=True`.
- **CrossAttentionModule** — projects `e_char`, pooled `e_mental`, and `e_opp2` to a shared
  `attention_hidden` dimension, stacks them as attention keys/values, and attends to them using
  the current-state feature vector as the query (`nn.MultiheadAttention`, `attention_heads` heads).
- **PredNet** — a residual conv torso over `[current_state, attended_features]` (when second
  belief is enabled) or `[current_state, e_mental, e_char]` / `[current_state, e_char]` otherwise,
  followed by six prediction heads: action, goal, agent-identity, agent-type, key/door
  consumption, and a per-discount-factor successor representation (SR) map.

`ToMnet.forward` supports three configurations, selected by `use_mentalnet` /
`use_second_belief` in `model_config`:
- 2-stage (`use_mentalnet=False`): CharNet → PredNet directly.
- 3-stage (`use_mentalnet=True`, `use_second_belief=False`): CharNet + MentalNet → PredNet.
- 3-stage + second belief (`use_mentalnet=True`, `use_second_belief=True`, the current default in
  `config.py`): CharNet + MentalNet + SecondBeliefNet → CrossAttentionModule → PredNet.

When `use_second_belief=True` but no opponent trajectory is available for a sample, the model
substitutes a zero vector for `e_opp2` rather than skipping the attention path.

`ToMnetLoss` (also in `beliefrl/model/tomnet.py`) combines the six prediction losses; per-head weights are set
in `Config.training_process_config`.

## Files

```
script/exp8/
├── config.py                # Config(BaseConfig): env, agent, model, training, data settings
├── generate.py               # Runs games between configured achiever/blocker agents, writes
│                              #   raw trajectory files
├── data_generation.py         # Parses raw trajectory files into ToMnet training tensors,
│                              #   including opponent-perspective trajectories for e_opp2
├── train.py                   # Training loop, checkpointing, plotting hooks
├── evaluate.py                 # Loads a trained model, runs evaluation/analysis, produces plots
├── visualize.py                # exp8-specific plots; the shared ones come from beliefrl.viz
├── simulate_game.py             # Runs and renders a single live episode (debug/demo)
├── simulate_trajectory.py        # Replays a saved trajectory file, optionally to GIF
└── utils.py                       # exp8-specific data preparation; shared helpers re-exported
                                   #   from beliefrl
```

The model, the agents, and everything else exp8 shared with the other experiments now live in
the [`beliefrl`](../../beliefrl) core:

| Module | Contents |
|--------|----------|
| `beliefrl/model/tomnet.py` | CharNet, MentalNet, SecondBeliefNet, CrossAttentionModule, PredNet, ToMnet, ToMnetLoss |
| `beliefrl/agents/value_agent.py` | `BaseValueAgent` — vectorized value iteration, memory of discovered keys/doors, clockwise wall-following exploration |
| `beliefrl/agents/achiever/` | `astar.py`, `random.py`, `level0value.py`, `level1value.py` |
| `beliefrl/agents/blocker/` | `random.py`, `randomlyselect.py`, `goaldirected.py`, `rulebased.py`, `level0value.py`, `level1value.py` |
| `beliefrl/config/base.py` | `BaseConfig` — the config accessors every experiment shared |
| `beliefrl/viz/` | `sr.py` (SR heatmaps) and `plots.py` (embedding and metric plots shared with exp7) |
| `beliefrl/data/`, `beliefrl/train/`, `beliefrl/utils.py` | Generation helpers, dataset loading, early stopping, epoch reporting, seeding |

Data-inspection scripts are shared across experiments and live in `script/tools/`:
`find_long_trajectories.py` (finds generated trajectory files above a length threshold)
and `analyze_interactions.py` (computes interaction/win-rate stats over generated data).
Both scan the whole `data/` tree, so they are not specific to exp8.

Each agent module exposes one policy class. `Level0*` agents plan with value iteration and no
opponent modeling; `Level1*` agents add opponent inference (blocker) or deceptive movement
(achiever). All `Level*Value*` agents inherit shared exploration/memory logic from
`BaseValueAgent` in `beliefrl/agents/value_agent.py`, including clockwise wall-following when exploring
unknown areas under partial observation.

## Running

The scripts are invoked directly with `python`; there is no wrapper shell script in this
directory. Typical pipeline:

```bash
# 1. Generate raw game trajectories (agent types configured in config.py, or overridden via CLI)
python script/exp8/generate.py --config_override --achiever_type lv1va --blocker_type lv1vb \
    --env_size 9x9 --observability partial

# 2. Convert trajectories into training tensors (opponent-perspective trajectories included)
python script/exp8/data_generation.py --data_dir <trajectory_dir> --output_dir <processed_dir>

# 3. Train ToMnet (second belief enabled by default via config.py's model_config)
python script/exp8/train.py --config_override --data_dir <processed_dir> --save_dir results/exp8

# 4. Evaluate a trained checkpoint
python script/exp8/evaluate.py --config_override --model_dir results/exp8 --plot_type all
```

`--config_override` enables CLI-argument overrides on top of `Config()` defaults for `train.py`,
`evaluate.py`, and `generate.py`. Each script also has its own `--help` listing available flags
(batch size, learning rate, architecture dimensions, device, seed, etc.). `data_generation.py`'s
own CLI takes `--data_dir`, `--output_dir`, `--time_step`, `--maze_width/height/depth` directly
and does not expose a `--config_override` flag despite one being read from `args` in its `main`
block (see Notes and Limitations).

`simulate_game.py` and `simulate_trajectory.py` are standalone debugging/demo scripts for
watching or replaying a single episode; they are not part of the training pipeline.

## Outputs

- `generate.py` writes raw per-episode trajectory files (maze layout, positions, actions,
  interactions) under `config.save_dir`.
- `data_generation.py` writes a pickled `processed_samples.pkl` of per-sample tensors, including
  `oppo_states` / `oppo_actions` for second-belief training.
- `train.py` writes model checkpoints and training-curve plots under `results/exp8` (path from
  `Config.model_dir` / `--save_dir`).
- `evaluate.py` writes evaluation metrics and, depending on `--plot_type`, calls into
  `visualize.py` to produce embedding plots (`plot_second_belief_embeddings` and its per-agent /
  per-goal variants for `e_opp2`), attention visualizations, and SR-map plots under
  `Config.plot_dir`.

## Notes and Limitations

- Second-order belief (`e_opp2`) is fully implemented end-to-end — `SecondBeliefNet`,
  `CrossAttentionModule`, opponent-perspective data extraction in `data_generation.py`, and
  dedicated visualization functions in `visualize.py` — and is enabled by default in
  `config.py`'s `model_config` (`use_second_belief: True`). It is not a pending item.
- When `use_second_belief=True` but `oppo_states` is `None` for a sample, `ToMnet.forward` uses a
  zero vector in place of `e_opp2`; no separate "no-opponent" learned embedding exists.
- `CrossAttentionModule.forward` builds a fresh `nn.Linear` query-projection layer on every call
  when the query feature dimension does not match `hidden_dim`; that projection is not registered
  as a persistent submodule, so its weights are not part of the checkpoint and are re-initialized
  randomly on the next call/reload.
- `data_generation.py`'s `if __name__ == "__main__"` block checks `args.config_override`, but its
  own argument parser never defines `--config_override`, so running that script directly with a
  config override flag will fail; it always runs with `Config()` defaults plus its own explicit
  CLI flags.
- No `shell/exp8/` wrapper script exists in this repository; run each stage with `python` directly
  as shown above.

## Known agent issues

Found by the golden-output harness in `tests/` (`pytest` from the repo root).

**Fixed.** `Level1ValueAchiever` and `Level1ValueBlocker` used to raise
`AttributeError: 'dict' object has no attribute 'width'` on every `get_action`
call, so `lv1va` and `lv1vb` could not be generated at all. Their
`_update_grid_reference` assigned the raw observation dict to `self.grid`
instead of decoding it; they now use `Grid.decode(obs[role]["image"])`, matching
`level0value.py`. This is why no `lv1` dataset exists under `data/`.

**Open — `self.grid` is read in two coordinate frames.** `BaseValueAgent`
indexes it with world coordinates at `value_agent.py:216` and `:766` but with
local egocentric coordinates at `:678` and `:728`, while
`Grid.decode(obs[role]["image"])` returns an agent-relative view. The world
call sites therefore read the wrong frame. This predates the refactor and
affects `Level0` and `Level1` equally. The symptom is a degenerate policy:
under full observability both `Level0ValueAchiever` and `Level1ValueAchiever`
emit one repeated action for 30 steps, while the blockers behave normally.
Resolving it is a modelling decision, not a mechanical fix.

**Open — four agents still treat the observation dict as a grid.**
`achiever/astar.py:120`, `blocker/randomlyselect.py:118`,
`blocker/goaldirected.py:100` and `blocker/rulebased.py:195` assign
`self.grid = obs[role]`. They do not crash, because none reads `.width`, but
`self.grid.get(x, y)` silently resolves to `dict.get` and returns the `y`
argument instead of a cell.

**Datasets.** `data/MiniGrid-AchieverBlocker-{5x5,9x9}-v2/lv0va_lv0vb` was
generated by the `Level0` value agents, which already decoded the grid
correctly, so it is unaffected by the defect fixed above. It is still subject to
the coordinate-frame issue. Whether to regenerate is an open question.
