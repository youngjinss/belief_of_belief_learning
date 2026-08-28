# Golden-output regression harness

This harness exists to make the exp5–exp8 refactor verifiable. The refactor
collapses four near-duplicate experiment directories onto one shared core, and
these goldens are the contract that migration must not break. They were recorded
against the pre-refactor code, so any drift they report is drift the refactor
introduced.

## Running

```bash
pytest                         # from the repo root
python tests/golden.py check   # same checks, no pytest dependency
```

Both complete in roughly three seconds, so they can be run after every step of
the refactor rather than saved for the end.

## Re-recording

```bash
python tests/golden.py record
```

Only re-record when a change to the recorded values is **intended**, and say so
in the commit message. Re-recording to silence a failure destroys the very
guarantee the harness provides.

## What is captured

Two probes run per experiment.

`tests/golden/<exp>.json` — configuration and model:

| Section | Contents |
|---|---|
| `config.attributes` | Every public attribute of a freshly constructed `Config` |
| `config.getters` | The return value of every zero-argument `get_*` / `is_*` method |
| `model.n_parameters` | Total parameter count of the instantiated `ToMnet` |
| `model.param_shapes_sha256` | Hash over every named parameter's shape |
| `model.forward` | Shape and sum of each output of one deterministic forward pass |

A single changed hyperparameter is traced through all five sections, which makes
failures easy to localise: config drift, architecture drift, and numerical drift
each show up in a distinct place.

`tests/golden/<exp>.agents.json` — agent behaviour:

| Field | Contents |
|---|---|
| `env_version` | `v1` for exp5–exp7, `v2` for exp8, taken from that experiment's `Config` |
| `agents.<module>.<Class>[mode]` | Status plus the 30-step action sequence from a fixed-seed rollout |

Since Phase 2 moved the agents into `beliefrl.agents`, the probe records both
sets for exp5–exp7: that experiment's own flat `achievers`/`blockers` modules
**and** the core agents, run against the same v1 environment and seed. That
side-by-side is the evidence Phase 3 needs, and it already shows where the work
is: six of the ten agents (AStar, both `RandomAgent`s, `GoalDirectAgent`,
`RandomlySelectedAgent`, `RuleBasedAgent`) produce byte-identical action
sequences and migrate for free, while the four value agents
(`Level0`/`Level1` × achiever/blocker) differ, because exp8 rewrote them for
partial observability.

Agent behaviour is invisible to the config/model probe, and Phase 2 rewires
agent imports, so it needs its own contract. Agents that currently raise are
recorded as raising — the goldens pin what the code *does*, not what it should
do.

## Design notes

**Subprocess isolation.** exp5–exp8 each expose flat modules with identical
names (`config`, `tomnet`, `utils`, ...). Importing more than one of them into a
single interpreter collides in `sys.modules`, so `tests/_probe.py` runs once per
experiment in its own interpreter with that experiment's directory as the
working directory.

**Signature adaptation.** The model API forked between exp6 and exp7:

- exp5, exp6 — `forward(past_trajectories, recent_trajectory, current_state)`
- exp7, exp8 — `forward(past_trajectories, self_states, self_actions, current_state, oppo_states, oppo_actions)`

The probe inspects the signature and supplies the arguments each version
accepts. Phase 3 will therefore need a compatibility adapter for exp5/exp6, not
configuration alone.

**Seeding.** `Config.get_costs()` and `Config.get_goal_rewards()` call
`np.random.uniform` without seeding, so they return different values on every
call. The probe reseeds from `cfg.seed` before each getter, which makes the
recorded values reproducible and independent of sweep order. This is a property
of the probe only — it does not change the experiment code, where the
randomisation remains as it was.

**Float tolerance.** Forward-pass sums are compared within `1e-3`
(`FLOAT_TOLERANCE` in `golden.py`). Shapes, parameter counts, and hashes are
compared exactly.

**`PYTHONHASHSEED`.** `golden.py` launches every probe with
`PYTHONHASHSEED=0`. This is required, not cosmetic: several agents branch on
set/dict iteration order, so without a pinned hash seed the same agent produces
a different action sequence on every run even when `random` and `np.random` are
seeded identically. It must be set in the child's environment — `set_seed()`
assigns `os.environ["PYTHONHASHSEED"]` at runtime, which has no effect, because
CPython fixes hash randomisation at interpreter startup.

## Known baseline facts

Recorded on first run, worth knowing before the refactor moves anything:

- exp7 and exp8 produce an **identical** architecture hash
  (`81b2b593…`, 5,134,746 parameters, 229 tensors). Their `tomnet.py` files
  differ by 143 lines that have no effect on the model.
- exp5 and exp6 differ by only 576 parameters (3,808,218 vs 3,808,794).
- `script/exp8/simulate_game.py` used to fail to import: it did
  `from achievers import ...`, but exp8 replaced exp7's flat `achievers.py`
  with the `agents/achiever/` package. Phase 1 repointed those imports, so all
  eleven exp8 modules now import cleanly.

Three defects the agent probe uncovered. All three are now fixed; every one of
the 50 recorded rollouts runs.

- **exp8 `lv1va` / `lv1vb` raised on every step.** `_update_grid_reference` did
  `self.grid = obs["<role>"]`, but that is the MiniGrid observation dict
  (`image`, `direction`, `mission`), not a grid. `BaseValueAgent` then called
  `self.grid.width` — `AttributeError` — while `self.grid.get(x, y)` resolved to
  `dict.get` and silently returned the y argument instead of a cell. Fixed by
  decoding the image, matching what `level0value.py` already did.
- **exp5/exp6/exp7 `rule_based` raised on construction.**
  `RuleBasedAgent.reset()` cleared `distance_history`, `reduction_rates` and
  `consecutive_reductions`, none of which `__init__` created; each name appeared
  exactly once in the class, only in `reset()`. `generate.py` calls `reset()`
  immediately after construction, so the blocker failed every time. Fixed by
  deleting the dead block — nothing ever read those attributes. The repaired
  action sequence matches exp8's already-working `RuleBasedAgent` exactly.
- **Runs were not reproducible across processes.** See the `PYTHONHASHSEED` note
  above.

## Open issues, deliberately not fixed

- **`self.grid` is read in two coordinate frames.** `BaseValueAgent` indexes it
  with world coordinates at `value_agent.py:216` and `:766`, but with local
  egocentric coordinates at `:678` and `:728`. `Grid.decode(obs[role]["image"])`
  yields an agent-relative view, so the world-coordinate call sites are still
  reading the wrong frame. This predates the refactor and affects `Level0` and
  `Level1` alike. It shows up as degenerate policies: under full observability
  both `Level0ValueAchiever` and `Level1ValueAchiever` emit a single repeated
  action for 30 steps, while the blockers vary normally.
- **Four exp8 agents still use the dict form** — `achiever/astar.py:120`,
  `blocker/randomlyselect.py:118`, `blocker/goaldirected.py:100`,
  `blocker/rulebased.py:195` — so they retain the silent `dict.get` corruption.
  They do not crash, because none of them reads `.width`.
- **exp5/exp6/exp7 use the dict form throughout** (eight sites across
  `achievers.py` and `blockers.py`). Left untouched so those frozen experiments
  keep the behaviour that produced their results.
- **The checked-in datasets are not implicated by the above bug.**
  `data/MiniGrid-AchieverBlocker-{5x5,9x9}-v2/lv0va_lv0vb` was generated by
  exp8's `Level0` value agents, which already used `Grid.decode`. They are
  affected by the coordinate-frame issue, like all value agents, but not by the
  dict-as-grid defect. Whether to regenerate is an open question.
