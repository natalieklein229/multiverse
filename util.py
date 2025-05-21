import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from collections import defaultdict

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
    w = width(yhat) # mean width
    b = norm.ppf((1 + alpha) / 2)
    p1 = 2/(1-alpha) * np.mean(y < yhat_mean-b*yhat_sd, 0)
    p2 = 2/(1-alpha) * np.mean(y > yhat_mean+b*yhat_sd, 0)
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

def freeze_all_but_last_n_linear(model, n=1):
    """
    Freezes all parameters in the model except those in the last `n` nn.Linear layers.
    
    Parameters:
        model (nn.Module): The model to modify.
        n (int): Number of last Linear layers to keep unfrozen.
    """
    # Collect all Linear layers in the order they appear
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]

    if len(linear_layers) < n:
        raise ValueError(f"Model has only {len(linear_layers)} Linear layers, but n={n} was requested.")

    # Step 1: Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze the last `n` Linear layers
    for layer in linear_layers[-n:]:
        for param in layer.parameters():
            param.requires_grad = True

    print(f"Unfroze last {n} linear layer(s): {linear_layers[-n:]}")

def freeze_all_but_first_layer(model):
    """
    Freezes all parameters in the model except those in the first layer (by order of appearance).
    """
    first_linear = None

    # Step 1: Identify the first nn.Linear or nn.Conv layer
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            first_linear = module
            break

    if first_linear is None:
        raise ValueError("No supported first layer (Linear or Conv) found in model.")

    # Step 2: Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Step 3: Unfreeze only the first layer
    for param in first_linear.parameters():
        param.requires_grad = True

    print(f"Unfroze first layer: {first_linear}")

def unfreeze_only_third_conv(model):
    """
    Freezes all parameters in the model, then unfreezes only the third convolutional layer.
    """
    # Step 1: Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Find and unfreeze the third Conv layer
    conv_layers = [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d))]

    if len(conv_layers) < 3:
        raise ValueError(f"Model has only {len(conv_layers)} Conv layers. Cannot unfreeze the third one.")

    third_conv = conv_layers[2]
    for param in third_conv.parameters():
        param.requires_grad = True

    print(f"Unfroze only the third conv layer: {third_conv}")

def unfreeze_second_to_last_layer(model):
    """
    Freezes all parameters in the model, then unfreezes only the second-to-last parameterized layer.
    """
    # Step 1: Find all layers (modules) with parameters
    param_layers = [m for m in model.modules() if any(p.requires_grad for p in m.parameters(recurse=False))]

    if len(param_layers) < 2:
        raise ValueError("Model has fewer than two parameterized layers.")

    # Step 2: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 3: Unfreeze only the second-to-last layer
    target_layer = param_layers[-2]
    for param in target_layer.parameters():
        param.requires_grad = True

    print(f"Unfroze only the second-to-last layer: {target_layer}")

def get_groups(labels):
    # Get unique rows and group indices
    unique_labels, inverse = np.unique(labels, axis=0, return_inverse=True)

    # Create groups: inverse maps each row to its group index
    from collections import defaultdict

    group_indices = defaultdict(list)
    for i, group_id in enumerate(inverse):
        group_indices[group_id].append(i)
    
    return group_indices
