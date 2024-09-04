import torch
import torch.nn as nn

from tqdm import tqdm

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam

class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=None, l_sizes=None, width=10, depth=5,
                 device=None, activation=None):
        '''
        :param bayes: whether to use a Bayesian Neural Network
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param l_sizes: list of hidden layer sizes
        :param width: width of hidden layers. to be specified when l_sizes is None
        :param depth: number of hidden layers. to be specified when l_sizes is None
        :param w_prior_scale: scale of the normal prior on weights
        :param b_prior_scale: scale of the normal prior on biases
        :param device: device to run on
        :param activation: activation function to use, one of 'leakyrelu' (default), 'relu', 'tanh', 'sigmoid'
        :param heteroskedastic: whether to use a heteroskedastic likelihood
        :param sigma: fixed standard deviation of the likelihood if heteroskedastic is False
        '''
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if out_dim is not None:
            out_dim = out_dim

        if activation is None or activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("activation must be one of 'leakyrelu' (default), 'relu', 'tanh', 'sigmoid'")
        self.output_size = out_dim
        if l_sizes is None:
            self.layer_sizes = [in_dim] + depth * [width] + [out_dim]
        else:
            self.layer_sizes = [in_dim] + l_sizes + [out_dim]

        layer_list = [y for x in [(nn.Linear(self.layer_sizes[idx - 1], self.layer_sizes[idx]).to(self.device) ,self.activation) for idx in
                      range(1, len(self.layer_sizes)-1)] for y in x]
        layer_list.append(nn.Linear(*self.layer_sizes[-2:]))
        self.out = torch.nn.Sequential(*layer_list).to(self.device)

    def forward(self, x):
        out = self.out(x)       
        return out     


