
import torch
from torch import nn
from bnns.SequentialGaussianBNN import SequentialGaussianBNN
torch.manual_seed(2024)

BNN = SequentialGaussianBNN(
        nn.Unflatten( dim=-1, unflattened_size=(-1,1) ),    # ~~~ in order to accept inputs x of the form x=torch.linspace(-1,1,100)
        nn.Linear(1, 40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 1)
    )
