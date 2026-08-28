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

Per experiment, in `tests/golden/<exp>.json`:

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

## Known baseline facts

Recorded on first run, worth knowing before the refactor moves anything:

- exp7 and exp8 produce an **identical** architecture hash
  (`81b2b593…`, 5,134,746 parameters, 229 tensors). Their `tomnet.py` files
  differ by 143 lines that have no effect on the model.
- exp5 and exp6 differ by only 576 parameters (3,808,218 vs 3,808,794).
- `script/exp8/simulate_game.py` does not import: it does
  `from achievers import ...`, but exp8 has no `achievers.py`. It is excluded
  from the harness because it is already dead code, scheduled for deletion in
  Phase 1.
