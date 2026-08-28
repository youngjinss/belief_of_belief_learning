# ToMnet Re-implementation

Re-implementation of core experiments from Rabinowitz et al., "Machine Theory of Mind" (ICML 2018), reproducing the qualitative results of Figure 3 (random-agent character inference) and Figure 5 (goal-directed agent prediction).

## Purpose

This module trains a ToMnet ("Theory of Mind network") to infer an agent's character from its past behavioral trajectories, then uses the inferred character embedding to predict the agent's behavior in a held-out gridworld. Two experiment configurations are implemented, mirroring two sections of the paper's appendix:

- **Figure 3** (Appendix A.2): random agents with fixed Dirichlet policies; ToMnet predicts only the next action.
- **Figure 5** (Appendix A.3.2): goal-directed agents that plan with value iteration; ToMnet predicts the next action, which objects get consumed by episode end, and successor representations.

Neither configuration uses the paper's Mental State Net. `MentalStateNet` exists in `tomnet.py`, but `create_tomnet` never instantiates it for either experiment.

## Environment

`environment.py` implements `GridWorld`: a grid with random walls, several consumable objects, and one agent.

- Actions: up, down, left, right, stay (5 total).
- State representation: a `(size, size, 6)` array — channel 0 is walls, channels 1-4 are one per object type (only as many as `n_objects` get populated), channel 5 is the agent's position.
- An episode ends when the agent steps onto an object or `max_steps` is reached.
- Movement carries a small negative reward (-0.01); hitting a wall or the boundary subtracts an additional -0.05.
- Module-level defaults (`SIZE`, `MAX_WALLS`, `MAX_STEPS`, `N_OBJECTS` in `environment.py`) are `3`, `2`, `31`, and `2` — a smaller board than the paper's 11×11 grid with 4 objects, likely kept small for faster experiments. All four are also constructor arguments, so the board can be scaled back up.

`agents.py` provides two agent types:

- **`RandomAgent`**: samples a fixed categorical policy from `Dirichlet(α, α, α, α, α)` at construction and never updates it. `α` controls how peaked the policy is (near-deterministic at `α=0.01`, near-uniform at `α=3.0`).
- **`GoalDirectedAgent`**: samples a reward vector over 4 object types from `Dirichlet(α_reward)`, then plans with value iteration (`plan()`) to obtain a softmax-over-Q policy for a given `GridWorld` instance. A move penalty, a wall-collision penalty, and an optional higher "greedy" movement cost (assigned to a `high_cost_ratio` fraction of agents) shape the reward. `get_successor_representation()` estimates the SR by Monte Carlo rollout, with both a serial and a multiprocessing-parallel implementation.

## Architecture

`tomnet.py` defines the network building blocks and a factory, `create_tomnet(experiment_type, ...)`, that assembles the right combination for `"figure3"` or `"figure5"`.

```
past trajectories (N_past episodes, 1 state+action pair each)
        │
        ▼
  CharacterNet  (Figure3CharacterNet | Figure5CharacterNet)
        │  per-episode embedding, summed over N_past → e_char
        ▼
  PredictionNet (Figure3PredictionNet | Figure5PredictionNet)
        │  spatializes e_char, concatenates with the query state
        ▼
  action logits  [, consumption logits, successor-representation logits]
```

- **`Figure3CharacterNet`**: a 1-layer conv (8 feature maps) over the spatialized state+action, followed by an LSTM (hidden size 128) and a fully-connected layer to a character embedding (default 2-D, for direct visualization). Per-episode embeddings are summed over past episodes.
- **`Figure5CharacterNet`**: an initial conv + batch-norm to 32 channels, then **3** `ResidualBlock`s, average pooling, and a fully-connected layer to an embedding (default 8-D), also summed over past episodes. The class docstring cites the paper's 5-layer ResNet, but an in-code comment notes the block count was "reduced for simple 3x3 maze".
- **`Figure3PredictionNet`**: concatenates the spatialized character embedding with the query state, runs a 2-layer conv (32 channels), average-pools, and outputs 5 action logits (softmax).
- **`Figure5PredictionNet`**: a shared torso (conv + batch-norm + 3 `ResidualBlock`s, same reduction as above) feeding three heads — action (conv → avgpool → FC → softmax over 5 actions), object consumption (conv → avgpool → FC → sigmoid over `n_objects`), and successor representation (two convs → 3 output channels, one per discount factor γ ∈ {0.5, 0.9, 0.99}, softmaxed over spatial position).
- **`ToMnet`** wraps a character net, an optional mental-state net, and a prediction net. `forward()` returns a dict of predictions plus the character (and mental, if enabled) embedding; `compute_loss()` sums whichever of action cross-entropy, consumption binary cross-entropy, and SR cross-entropy losses have matching targets in the batch.
- The generic `CharacterNet` and `PredictionNet` classes, plus `MentalStateNet`, are implemented but unused by `create_tomnet` — both `figure3` and `figure5` set `use_mental_state=False`.
- For all three character nets (`CharacterNet`, `Figure3CharacterNet`, `Figure5CharacterNet`), `N_past=0` returns a zero embedding. Two of them also define an unused `no_past_embedding` learnable parameter whose docstring claims it replaces the zero embedding — the forward pass does not reference it.

## Files

```
ToMnet_impl/
├── scripts/
│   ├── environment.py            # GridWorld
│   ├── agents.py                 # RandomAgent, GoalDirectedAgent
│   ├── data_generation.py        # DataGenerator, ToMnetDataset, collate_fn
│   ├── tomnet.py                 # network modules + create_tomnet factory
│   ├── train.py                  # Figure 3 training (per-alpha + mixed-species)
│   ├── train_enhanced.py         # Figure 3 training variant with larger datasets
│   ├── train_figure5.py          # Figure 5 training (action + consumption + SR)
│   ├── generate_figure5_data.py  # standalone Figure 5 data-generation CLI
│   ├── evaluate.py               # Bayes-optimal baseline, KL/JS metrics, cross-species eval (Figure 3)
│   ├── evaluate_figure5.py       # Figure 5 evaluation
│   ├── visualize_figure3.py      # Figure 3a/3b/3c plots
│   ├── visualize_figure5.py      # Figure 5b/5d plots
│   ├── visualize_env_random.py   # standalone GridWorld rendering smoke test
│   ├── visualize_env_goal.py     # standalone GridWorld + GoalDirectedAgent smoke test
│   └── debug/debug_embeddings.py # embedding-inspection script (stale paths, see Notes)
├── shell/
│   ├── run_exp3.sh, run_exp3_parallel.sh  # train/evaluate/visualize Figure 3 end to end
│   ├── run_exp5.sh                        # same, for Figure 5
│   ├── train_enhanced.sh                  # wraps train_enhanced.py
│   ├── visualize_figure3.sh, visualize_figure5.sh
│   ├── show_structure.sh                  # prints/logs the directory tree
│   └── debug/test_likelihood_exp3.sh
├── notebook/
│   └── visualize_figure3.ipynb   # interactive companion to visualize_figure3.py
└── README.md
```

## Running

Figure 3 (random agents):

```bash
python scripts/train.py --experiment figure3 --n_agents 100 --n_episodes_per_agent 100 \
    --alpha_values 0.01 0.03 0.1 0.3 1.0 3.0 [--mixed_training]
python scripts/visualize_figure3.py --results_path result/figure3/cross_species_results.pkl --save_plots
```

or via the wrapper script: `bash shell/run_exp3.sh all` (`bash shell/run_exp3.sh help` lists sub-commands; `run_exp3_parallel.sh` trains the alpha sweep concurrently instead of sequentially).

Figure 5 (goal-directed agents):

```bash
python scripts/generate_figure5_data.py --n_agents 100 --n_episodes_per_agent 100
python scripts/train_figure5.py --n_agents 100 --n_episodes_per_agent 100 --n_epochs 100
python scripts/visualize_figure5.py --results_path result/figure5/figure5_results.pkl --save_plots
```

or `bash shell/run_exp5.sh all`.

`train.py` defaults: `--batch_size 64`, `--learning_rate 1e-3`, `--n_epochs 100`; device is auto-selected (`mps` on macOS if available, `cuda:3` on Linux with CUDA, otherwise `cpu`). `train_figure5.py` defaults: `--batch_size 512`, `--learning_rate 1e-1`, `--device cuda:3`. Per-experiment settings — character embedding dimension (10 for Figure 3, 8 for Figure 5), dropout rate (0.3 for both), and early-stopping patience (30 epochs for Figure 3, 10 for Figure 5) — are fixed in `ExperimentConfig` / `Figure5ExperimentConfig` inside `train.py` / `train_figure5.py` rather than exposed as CLI flags.

## Outputs

Scripts write into directories created at runtime; none of these are checked into the repository:

- `data/<experiment>/*.pkl` — generated trajectory datasets.
- `models/<experiment>/*.pth` — trained checkpoints (e.g. `0.01_best.pth`, `mixed_best.pth`).
- `result/<experiment>/` — `training_results.json`, evaluation pickles (`cross_species_results.pkl`, `figure5_results.pkl`), and, for Figure 3, an auto-generated `run_cross_species_evaluation.sh`.
- `plots/<experiment>/` — PNGs: `figure3a_action_likelihood.png`, `figure3b_character_embeddings.png`, `figure3c_cross_species_kl.png`; `figure5b_n_past_vs_likelihood.png`, `figure5d_embedding_space.png`.
- `log/...` — timestamped logs written by the shell wrapper scripts.

## Notes and Limitations

- The Mental State Net is implemented but not wired into either experiment; both currently run character-only inference.
- `Figure5CharacterNet` and the `Figure5PredictionNet` torso use 3 residual blocks rather than the paper's 5, per an in-code comment about the smaller default grid.
- Figure 3d ("mixed species training") has a plotting function defined but commented out in `visualize_figure3.py`; only 3a, 3b, and 3c are currently generated by that script.
- The default environment (3×3 grid, 2 objects, up to 2 walls) is much smaller than the paper's 11×11/4-object setup. `create_goal_directed_agents` still samples reward vectors over 4 components regardless of the environment's actual `N_OBJECTS`.
- `scripts/debug/debug_embeddings.py` hardcodes `state_dim = 11 * 11 * 6` and a `models/figure3_0.01_best.pth` path, neither of which matches the current default 3×3 environment or the `models/<experiment>/` output layout.
- No dependency manifest (`requirements.txt`, etc.) is present in this directory; scripts import `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, and `tqdm` directly.
