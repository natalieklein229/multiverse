
import torch
from torch import nn
from bnns.SequentialGaussianBNN import SequentialGaussianBNN
torch.manual_seed(2024)

BNN = SequentialGaussianBNN(
        nn.Linear(2, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 2)
    )
