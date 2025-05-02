import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

def rmse(y, yhat):
    if yhat.ndim == 3:
        yhat = np.mean(yhat, 0)
    return np.sqrt(np.mean(np.square(y-yhat),0))

def coverage(y, yhat, alpha=0.95):
    yhat_mean = np.mean(yhat, 0)
    yhat_sd = np.std(yhat, 0)
    b = norm.ppf((1 + alpha) / 2)
    c = np.logical_and(yhat_mean-b*yhat_sd <= y, yhat_mean+b*yhat_sd >= y)
    return np.mean(c, 0)

def width(yhat, alpha=0.95):
    yhat_sd = np.std(yhat, 0)
    b = norm.ppf((1 + alpha) / 2)
    return np.mean(b*yhat_sd,0)

def interval_score(y, yhat, alpha=0.95):
    yhat_mean = np.mean(yhat, 0)
    yhat_sd = np.std(yhat, 0)
    w = width(yhat)
    b = norm.ppf((1 + alpha) / 2)
    p1 = 2/alpha * np.sum(y < yhat_mean-b*yhat_sd, 0)
    p2 = 2/alpha * np.sum(y > yhat_mean+b*yhat_sd, 0)
    return w + p1 + p2

def scale_targets(y, log_var):
    return y * torch.exp(-0.5 * log_var)

def freeze_all_but_last_linear(model):
    """
    Freezes all parameters in the model except those in the last nn.Linear layer.
    """
    # Step 1: Find the last nn.Linear layer
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module  # overwrite until the last Linear is found

    if last_linear is None:
        raise ValueError("No Linear layer found in model!")

    # Step 2: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 3: Unfreeze parameters in last linear layer
    for param in last_linear.parameters():
        param.requires_grad = True

    print(f"Unfroze last linear layer: {last_linear}")