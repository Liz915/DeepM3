import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    锁定所有随机种子，确保实验可复现
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"🔒 Global Seed set to: {seed}")