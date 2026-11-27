import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, balanced_accuracy_score
import torch.nn.functional as F
from util.temperature_scaling import ModelWithTemperature
from util.bin_temperature_scaling import ModelWithBinwiseTemperature
from .metric import validation_accuracy_lora





@torch.no_grad()
def validation_accuracy(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    for batch_idx, (inputs, masks, targets) in enumerate(loader):
        inputs  = inputs.to(device)
        masks   = masks.to(device)
        targets = targets.to(device)

        if targets.ndim > 1 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        if targets.ndim > 1:
            targets = targets.view(-1)

        outputs = model(inputs, masks)   # logits
        _, predicted = outputs.max(1)

        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return correct / total


