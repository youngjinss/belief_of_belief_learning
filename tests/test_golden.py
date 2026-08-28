"""Regression guard for the exp5..exp8 refactor.

Each experiment's configuration and model must stay identical to the golden
recorded before the refactor began.
"""

import pytest

from golden import EXPERIMENTS, diff, load_golden, run_probe


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_matches_golden(experiment):
    differences = list(diff(load_golden(experiment), run_probe(experiment)))
    assert not differences, "\n".join(differences[:40])
