"""beliefrl — shared core for the ToMnet belief-modelling experiments.

Extracted from script/exp8, the live experiment. exp5..exp8 previously carried
near-identical copies of the same agents, model, data pipeline and training
loop; this package holds one copy, and each experiment keeps only its
configuration and whatever genuinely differs.

Behaviour is pinned by the golden-output harness in tests/: `pytest` from the
repo root compares every experiment's config, model and agent rollouts against
recordings made before the extraction began.
"""

__all__ = ["agents", "env"]
