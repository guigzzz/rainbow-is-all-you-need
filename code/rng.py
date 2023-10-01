import torch
import numpy as np
import random

seed = 777


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def set_seeds():
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
