# Golden-output regression harness

Code duplicated across exp3–exp8 moved into the shared [`beliefrl`](../beliefrl)
core. These goldens are the contract that migration must not break: they were
recorded against the pre-refactor code, so any drift they report is drift the
refactor introduced.

Eighteen checks — six experiments, three probes each — in about ten seconds.

## Running

```bash
pytest                          # from the repo root
python tests/golden.py check    # same checks, no pytest dependency
python tests/golden.py record   # re-record (see below)
```

Only re-record when a change to the recorded values is **intended**, and say so
in the commit message. Re-recording to silence a failure destroys the guarantee.

## What is captured

| File | Contents |
|------|----------|
| `golden/<exp>.json` | Every public `Config` attribute and zero-argument getter; the model's parameter count, a hash over every parameter shape, and the shape and sum of each output of one deterministic forward pass |
| `golden/<exp>.agents.json` | A 30-step fixed-seed rollout for every agent class, against that experiment's own environment version |
| `golden/<exp>.train.json` | One training step mirroring `train.py` — forward, `ToMnetLoss`, backward, `optimizer.step()` — recording every loss component before and after, plus gradient norm |

A single changed hyperparameter shows up in a distinct place in each section,
which makes failures easy to localise: config drift, architecture drift, and
numerical drift are separable.

Agents or steps that currently raise are recorded as raising. The goldens pin
what the code *does*, not what it should do.

## Design notes

**Subprocess isolation.** Each experiment exposes flat modules with identical
names (`config`, `tomnet`, `utils`, ...), which collide in `sys.modules`. Every
probe therefore runs in its own interpreter, with that experiment's directory as
the working directory.

**Signature adaptation.** The model API forked between exp6 and exp7:

- exp3–exp6 — `forward(past_trajectories, recent_trajectory, current_state)`
- exp7, exp8 — `forward(past_trajectories, self_states, self_actions, current_state, oppo_states, oppo_actions)`

The probes inspect the signature and supply what each version accepts.

**`PYTHONHASHSEED`.** `golden.py` launches every probe with
`PYTHONHASHSEED=0`. This is required, not cosmetic: several agents branch on
set/dict iteration order, so without a pinned hash seed the same agent produces
a different action sequence on every run even when `random` and `np.random` are
seeded identically. It must be set in the child's environment — `set_seed()`
assigns `os.environ["PYTHONHASHSEED"]` at runtime, which has no effect, because
CPython fixes hash randomisation at interpreter startup.

**Seeding.** `Config.get_costs()` and `get_goal_rewards()` draw from
`np.random` without seeding, so the probe reseeds from `cfg.seed` before each
getter. That is a property of the probe only; the experiment code is unchanged.

**Float tolerance.** Forward-pass and loss values are compared within `1e-3`
(`FLOAT_TOLERANCE` in `golden.py`). Shapes, counts, and hashes are exact.

## Defects the harness found, since fixed

- **exp8's `lv1va` / `lv1vb` raised on every step.** `_update_grid_reference`
  assigned the MiniGrid observation dict to `self.grid`, so `self.grid.width`
  raised `AttributeError` while `self.grid.get(x, y)` silently resolved to
  `dict.get` and returned the y argument instead of a cell. Fixed by decoding
  the image, matching what `level0value.py` already did. This is why no `lv1`
  dataset exists.
- **exp5/exp6/exp7's `rule_based` raised on construction.** `reset()` cleared
  three attributes `__init__` never created; each appeared exactly once in the
  class, only in that `reset()`. Fixed by deleting the dead block. The repaired
  action sequence matches exp8's already-working version exactly.
- **Runs were not reproducible across processes.** See `PYTHONHASHSEED` above.
- **Value agents never moved.** `_is_walkable` always applied the partial-view
  egocentric transform, even under full observability, so it indexed the
  agent-relative observation image at offset coordinates and reported walls
  where the world has open floor. Agents concluded they were boxed in and
  emitted `stay` for the rest of the episode. Fixed by building a world grid
  from the layout the environment reports and indexing it globally when
  observability is full; the egocentric path now runs only for partial views.
  The achiever walks straight to its preferred key again and the blocker to the
  matching door.
- **The agent probe stepped the environment with a dict.** `env.step` unpacks
  `achiever_action, blocker_action = actions`, so a dict silently unpacked its
  *keys* as actions and the world never advanced. Fixed to pass a pair.

## Open issues, deliberately not fixed

- **Four exp8 agents still treat the observation dict as a grid** —
  `achiever/astar.py:120`, `blocker/randomlyselect.py:118`,
  `blocker/goaldirected.py:100`, `blocker/rulebased.py:195`. They do not crash,
  because none reads `.width`, but they retain the silent `dict.get` corruption.
  exp3–exp7 use the same form throughout, left untouched so those frozen
  experiments keep the behaviour that produced their results.
- **The checked-in datasets are stale.**
  `data/MiniGrid-AchieverBlocker-{5x5,9x9}-v2/lv0va_lv0vb` was generated before
  the walkability fix, by agents that froze after a handful of steps: in the 9x9
  episodes the achiever stops moving at step 6 and never reaches its key. They
  should be regenerated before being used for training or reported results.
  `python -m beliefrl.viz.replay --live` plays a fresh game with the current
  agents without touching `data/`.
- **`lib/benchmark/` was not refactored.** It carries the same
  copy-per-experiment duplication (~3,700 lines between ToMnetF experiment4 and
  experiment5 alone), but it is a self-contained reference re-implementation,
  imported by no live code and covered by no tests.
