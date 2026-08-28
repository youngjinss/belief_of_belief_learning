"""Cross-cutting helpers shared by every experiment."""

import os
import random
import sys
import warnings

import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all major libraries.

    Args:
        seed (int): Random seed value
    """
    # Python random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # CUDA convolution determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PYTHONHASHSEED deliberately not assigned here. CPython fixes hash
    # randomisation at interpreter startup, so setting it at runtime has no
    # effect and merely looks reproducible. Several agents branch on set/dict
    # iteration order, so without it the same seed yields different action
    # sequences from process to process. Warn rather than pretend.
    import warnings

    if sys.flags.hash_randomization and "PYTHONHASHSEED" not in os.environ:
        warnings.warn(
            f"PYTHONHASHSEED is unset, so hash randomisation is active and runs "
            f"are not reproducible across processes despite seed={seed}. "
            f"Launch with PYTHONHASHSEED=0 for reproducible results.",
            RuntimeWarning,
            stacklevel=2,
        )

    # For DataLoader workers
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker
