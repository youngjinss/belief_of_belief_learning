"""Regression guard for the exp5..exp8 refactor.

Each experiment's configuration, model, and agent behaviour must stay identical
to the goldens recorded before the refactor began.
"""

import pytest

from golden import EXPERIMENTS, KINDS, diff, load_golden, run_probe


@pytest.mark.parametrize("experiment", EXPERIMENTS)
@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k or "config_model")
def test_experiment_matches_golden(experiment, kind):
    differences = list(diff(load_golden(experiment, kind), run_probe(experiment, kind)))
    assert not differences, "\n".join(differences[:40])
