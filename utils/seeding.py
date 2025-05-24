import random
import numpy as np
import torch

def seed_all(seed: int) -> None:
    """Set random seeds for reproducibility across modules."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
