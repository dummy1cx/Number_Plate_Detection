import torch
import torch.nn.functional as F

## --------------------------------------------------------------
## This code is refercned fro pytorch vision modules
## ---------------------------------------------------------------

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    probs = torch.sigmoid(inputs)
    pt = torch.where(targets == 1, probs, 1 - probs)
    loss = alpha * (1 - pt) ** gamma * BCE_loss
    return loss.sum() if reduction == "sum" else loss.mean()
