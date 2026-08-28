# Experiment 7 — Second-Order Belief (e_opp2)

This directory implements a Theory-of-Mind (ToM) network for the Achiever–Blocker multi-agent environment, extending the 3-stage ToMnet (CharNet → MentalNet → PredNet) with a fully implemented `SecondBeliefNet` that produces a second-order belief embedding, `e_opp2` — a representation of what an agent infers about its opponent's recent behavior, fused into prediction via cross-attention.

## Purpose

1. **Second-Order Belief Modeling** — `e_opp2` is a learned embedding derived from the opponent's recent trajectory (states + actions), fused with the agent's own mental state.
2. **Enhanced ToM Architecture** — extends the existing 2-/3-stage ToMnet with an optional cross-attention stage that combines character, mental, and second-belief embeddings before prediction.
3. **Opponent Trajectory Integration** — data generation extracts the opponent's recent trajectory from the same game log via perspective switching (achiever samples use blocker data as "opponent", and vice versa).
4. **Comparative Evaluation** — `use_second_belief` and `use_mentalnet` are independent boolean flags, so 2-stage/3-stage architectures with or without second-order belief can all be trained and compared from the same codebase.

## Environment

- Multi-agent **AchieverBlocker** environment (`MiniGrid-AchieverBlocker-{size}-v1`), 9×9 by default (`config.py`).
- When no blocker types are configured, the code falls back to a single-agent **KeyDoor** environment (`MiniGrid-KeyDoor-{size}-v1`), detected via `Config.is_single_agent_mode()`.
- Achiever types include `lv0va`, `lv1va` (level-k reasoning), `astar`, `random`, `value`; blocker types include `lv0vb`, `lv1vb`, `random`, `goal_direct`, `randomly_selected`, `rule_based`.
- Grid state is encoded as 10 channels: 8 original game-state channels (walls, keys, doors, etc.) plus channel 8 (self position) and channel 9 (opponent position, zero in single-agent mode).
- Action spaces differ per role: achiever has 7 actions (up/right/down/left/stay/pickup/toggle), blocker has 6 (up/right/down/left/stay/broken).

## Architecture

### Embeddings

| Embedding | Network | Input | Output | Status |
|---|---|---|---|---|
| `e_char` | `CharNet` | Past episode trajectories | `(batch, n_echar)` vector | Implemented |
| `e_mental` | `MentalNet` | Recent self states + actions | `(batch, n_ement, H, W)` spatial map | Implemented |
| `e_opp2` | `SecondBeliefNet` | Mental state + opponent's recent states/actions | `(batch, n_eopp2)` vector | Implemented |

`SecondBeliefNet` (`tomnet.py`) mirrors `MentalNet`'s trajectory encoder — a conv stem, 5 residual blocks, and a `ConvLSTM2d` over the opponent's spatialized state+action sequence — then fuses the resulting features with the agent's own spatial `e_mental` via a concatenation + conv fusion layer, projects to `n_eopp2` channels, and global-average-pools to a vector. Dropout (0.1) is applied to the output.

`ToMnet` exposes two independent flags:
- `use_mentalnet`: `False` → 2-stage (`CharNet → PredNet`); `True` → 3-stage (`CharNet → MentalNet → PredNet`).
- `use_second_belief`: adds `SecondBeliefNet` and switches `PredNet` to combine embeddings via `CrossAttentionModule` instead of simple channel concatenation.

When `use_second_belief=True`, `oppo_states`/`oppo_actions` are optional at the `forward()` call site — if omitted (single-agent samples), `second_belief` is set to a zero vector so the rest of the pipeline runs unchanged.

### Cross-Attention (when `use_second_belief=True`)

`CrossAttentionModule` projects `e_char`, pooled `e_mental`, and `e_opp2` to a shared `attention_hidden` dimension, stacks them as a 3-token sequence, and attends to them using a query built from the current-state features (`nn.MultiheadAttention`, `attention_heads` heads). The attended output is broadcast spatially and concatenated with the current state before entering `PredNet`'s convolutional torso.

```
Past Episodes ──────────────► CharNet ──────► e_char ─────────────┐
                                                                    │
Recent Self States/Actions ─► MentalNet ────► e_mental ─┬──────────┤
                                                          │          │
Opponent Recent States/Actions ─► SecondBeliefNet ◄──────┘          │
                              │                                     │
                              └────────────► e_opp2 ────────────────┤
                                                                     ▼
                                              Current State + [e_char, e_mental, e_opp2]
                                                          │
                                                  CrossAttentionModule
                                                          │
                                                        PredNet
                                                          │
                          action / goal / agent / type / consumption logits, SR map
```

When `use_second_belief=False`, `PredNet` falls back to plain channel-concatenation of the current state with `e_mental` (if `use_mentalnet`) and `e_char` — no attention module is built.

### PredNet outputs

A shared conv torso (conv → 3 residual blocks → conv, `out_channels` filters) feeds pooled features into five heads — action logits, goal logits, agent-identity logits (achiever/blocker), type logits (level/depth), and consumption logits (4 keys + 4 doors) — plus a convolutional successor-representation (SR) head producing a softmax-normalized spatial map for 3 discount factors.

### Trajectory file format

`DataGenerator.parse_trajectory_file` (`data_generation.py`) reads raw text logs and matches each step line against a pre-compiled regex. The multi-agent format is:

```
[achiever_x, achiever_y][blocker_x, blocker_y] : achiever_action,blocker_action : achiever_interaction,blocker_interaction
```

A single-agent (KeyDoor) log uses the shorter form `[x, y] : action : interaction`. Perspective switching for `SecondBeliefNet` follows directly from this shared log: an achiever sample's `oppo_states`/`oppo_actions` are built from the same file's blocker positions and actions (and vice versa for blocker samples), via `_create_trajectory_tensor(..., "blocker", ...)` / `(..., "achiever", ...)`.

## Files

- `config.py` — central `Config` class: environment, model, data, and training settings; `get_model_kwargs()` builds the `ToMnet` constructor arguments; `update_from_args()` applies CLI overrides.
- `tomnet.py` — model definitions: `ResidualBlock`, `LSTM`, `CharNet`, `ConvLSTM2d`, `MentalNet`, `SecondBeliefNet`, `CrossAttentionModule`, `PredNet`, `ToMnet`, `ToMnetLoss`, plus `create_model()` and `count_parameters()` helpers. Running the file directly (`python tomnet.py`) exercises all four architecture combinations (2-/3-stage × with/without second belief) with a forward pass and shape checks.
- `data_generation.py` — `DataGenerator` class: parses raw trajectory text files, builds per-timestep state tensors (`_create_trajectory_tensor`), and creates achiever/blocker training samples via `create_achiever_sample` / `create_blocker_sample`, including the opponent's trajectory (`oppo_states`, `oppo_actions`) extracted from the opponent's perspective for `SecondBeliefNet`.
- `generate.py` — CLI entry point that drives simulated games and `DataGenerator` to produce trajectory data on disk.
- `train.py` — training loop: batches data (including `oppo_states`/`oppo_actions`), computes `ToMnetLoss`, runs validation, early stopping, and checkpointing.
- `evaluate.py` — loads a trained checkpoint, computes accuracy/precision/recall/F1 and confusion matrices for each prediction head, and can produce `char_embeddings`/`mental_embeddings`/`n_past` plots (`--plot_type`).
- `visualize.py` — plotting utilities, including a unified `EmbeddingExtractor` (supports `"character"`, `"mental"`, and `"second_belief"` embedding types) and dedicated `plot_second_belief_embeddings*` functions (PCA/t-SNE by agent and by goal).
- `simulate_game.py`, `simulate_trajectory.py` — interactive/episode-level game simulation and rendering.
- `value_agent.py`, `achievers.py`, `blockers.py` — agent policy implementations (value-based, level-k, rule-based, random, etc.) used to generate trajectories.
- `visualize_sr.py` — successor-representation-specific plotting.
- `utils.py` — shared helpers, including `spatialize_action` (turns a discrete action index into a spatial one-hot channel, used identically by `MentalNet` and `SecondBeliefNet`).
- `unit_test/` — `analyze_interactions.py`, `find_long_trajectories.py` — data-inspection scripts, not a pytest suite.

## Running

The pipeline is orchestrated by `shell/exp7/run_exp7.sh`:

```bash
bash shell/exp7/run_exp7.sh all              # full pipeline: data gen → test data gen → train → evaluate → visualize
bash shell/exp7/run_exp7.sh debug            # same pipeline at reduced scale (config.enable_debug_mode())
bash shell/exp7/run_exp7.sh data_generation  # generate training trajectories only
bash shell/exp7/run_exp7.sh train            # train only (expects existing data)
bash shell/exp7/run_exp7.sh evaluate         # evaluate a trained checkpoint
bash shell/exp7/run_exp7.sh visualize        # produce plots only
```

Each stage can also be invoked directly, e.g.:

```bash
python script/exp7/generate.py --config_override
python script/exp7/train.py --config_override --save_dir results/exp7/<run>
python script/exp7/evaluate.py --config_override --model_dir results/exp7/<run> --result_dir results/exp7/<run> --plot_type all
python script/exp7/visualize.py --config_override --result_dir results/exp7/<run> --plot_dir results/exp7/<run>/plots --plot_type all
```

`train.py`, `evaluate.py`, and `visualize.py` all take `--config_override` plus specific flags (batch size, device, residual blocks, embedding dimensions, etc.) that overwrite the corresponding `Config` fields — see each script's `argparse` section for the full list. There is no CLI flag to toggle `use_second_belief`/`use_mentalnet`; they are set directly in `config.py`'s `model_config` (both default to `True`).

### Key config fields (`config.py`, `model_config`)

```python
"use_mentalnet": True,        # 2-stage vs 3-stage
"use_second_belief": True,    # enable SecondBeliefNet + cross-attention
"residual_blocks": 5,
"n_echar": 128,
"n_ement": 128,
"n_eopp2": 128,                # second-belief embedding dimension
"second_belief_hidden": 64,    # SecondBeliefNet hidden width
"attention_hidden": 256,       # cross-attention projection dimension
"attention_heads": 8,
```

### Training config (`config.py`, `training_config`)

```python
"batch_size": 512,
"epochs": 300,
"lr": 0.0001,
"weight_decay": 0.001,
"training_proportion": 0.9,
"use_amp": True,                     # automatic mixed precision
"gradient_accumulation_steps": 2,
"use_parallel": True,                # multi-GPU training
"device_ids": [2, 3],
"optimizer": "adam",
```

`training_process_config` additionally sets early-stopping patience (100 epochs, `min_delta=0.001`), gradient-norm clipping (`max_grad_norm=1.0`), and per-head loss weights (`action_weight`, `goal_weight`, etc.) consumed by `ToMnetLoss`.

## Outputs

- `results/exp7/<run>/best_model.pth` — best checkpoint (by validation loss / early stopping).
- `results/exp7/<run>/evaluation_results_exp7.json` — accuracy/F1/confusion-matrix summary produced by `evaluate.py`.
- `results/exp7/<run>/plots/` — training curves, confusion matrices, and embedding visualizations, including `second_belief_embeddings_by_agent_exp7.png` and `second_belief_embeddings_by_goal_exp7.png` from `visualize.py --plot_type second_belief_embeddings` (or `all`).
- `log/exp7/<run>/*.log` — per-stage logs (data generation, training, evaluation, visualization) written by `run_exp7.sh`.

## Notes and Limitations

- `use_second_belief` and `use_mentalnet` can be combined in any of the four ways, but `SecondBeliefNet` is only meaningfully exercised when opponent trajectory data is available (multi-agent games); in single-agent (KeyDoor) samples `e_opp2` is a zero vector.
- `evaluate.py`'s `--plot_type` choices (`basic`, `char_embeddings`, `mental_embeddings`, `n_past`, `all`) do not include a second-belief-specific option — second-belief embedding plots are generated only through `visualize.py`.
- `CrossAttentionModule.forward` instantiates a fresh `nn.Linear` on the fly whenever the query feature dimension differs from `attention_hidden`, rather than as a registered submodule; this works but means that projection's weights are re-initialized on every call rather than trained.
- Variable naming for trajectory data was standardized in an August 2025 refactor: `trajectory → self_states`, `opponent_trajectory → oppo_states`, `actions → self_actions`, `opponent_actions → oppo_actions`, and `spatialize_action` was consolidated into `utils.py` to avoid duplication between `MentalNet` and `SecondBeliefNet`.
- exp8 (`script/exp8/`) carries a near-identical `tomnet.py` (same class set: `SecondBeliefNet`, `CrossAttentionModule`, etc.) but its `data_generation.py` adds `_apply_partial_observation_masking` and a `_create_state_tensor` method not present in exp7, suggesting exp8 extends this architecture toward partial-observability settings.
