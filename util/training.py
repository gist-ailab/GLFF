import os
import random

import numpy as np
import torch


def seed_for_init(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_requires_grad(model, layers_to_train):
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name for layer in layers_to_train)
