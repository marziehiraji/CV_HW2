"""Reproducibility helpers."""
import os
import random
import numpy as np
import torch

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Deterministic cuDNN algorithms are intentionally NOT enabled here
    because the host runs on CPU and the environment-variable switch
    `CUBLAS_WORKSPACE_CONFIG` would change numerical results without
    added benefit.  See the report for a discussion of residual
    non-determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = config.USE_DETERMINISTIC
    torch.backends.cudnn.benchmark = not config.USE_DETERMINISTIC
