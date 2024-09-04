import torch
import torch.nn as nn

import copy

from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, ch_sizes=None, krnl_sizes = None, stride=None, lin_l_sizes=None, 
                 lin_width=10, lin_depth=5, activation=None, alt_parametrization=True, loss='NLL', device=None):
        '''
        :param bayes: whether to use a Bayesian Neural Network
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param conv_l_sizes: list of convolution layer dimensions
        :param lin_l_sizes: list of linear layer sizes
        :param lin_width: width of linear layers. to be specified when lin_l_sizes is None
        :param lin_depth: number of linear layers. to be specified when lin_l_sizes is None
        :param device: device to run on
        :param activation: activation function to use, one of 'leakyrelu' (default), 'relu', 'tanh', 'sigmoid'
        :param loss: loss function, one of 'NLL' (default), 'DE', 'CE'
        '''
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

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

        # Convolutional layers
        self.output_size = out_dim
        assert len(krnl_sizes) == (len(ch_sizes)) == len(stride), "krnl_sizes, ch_sizes and stride must have the same length"

        conv_layer_size = len(ch_sizes)
        conv_layer_list = [nn.Conv1d(1, out_channels=ch_sizes[0], kernel_size = krnl_sizes[0], stride=stride[0]).to(self.device), self.activation]

        conv_layer_list = [*conv_layer_list, *[y for x in [(nn.Conv1d(ch_sizes[idx], out_channels=ch_sizes[idx+1], kernel_size = krnl_sizes[idx+1], stride=stride[idx+1]).to(self.device), self.activation) for idx in
                      range(conv_layer_size-1)] for y in x]]
        
        ## Linear layers

        # calculate the size of the input to the first linear layer
        first_lin_layer_size = in_dim
        for i in range(conv_layer_size):
            first_lin_layer_size = (first_lin_layer_size - krnl_sizes[i])//stride[i] + 1

        if lin_l_sizes is None:
            # linear layers default to lin_depth layers of size lin_width
            self.lin_layer_sizes = [first_lin_layer_size] + lin_depth * [lin_width] + [out_dim]
        else:
            # possibility to define custom sizes for linear layers
            self.lin_layer_sizes = [first_lin_layer_size] + lin_l_sizes + [out_dim]

        # build the actual linear layers
        lin_layer_list = [y for x in [(nn.Linear(self.lin_layer_sizes[idx - 1], self.lin_layer_sizes[idx]).to(self.device) ,self.activation) for idx in
                      range(1, len(self.lin_layer_sizes)-1)] for y in x]
        lin_layer_list.append(nn.Linear(*self.lin_layer_sizes[-2:]))

        # concatenate convolutional and linear layers
        layers = [*conv_layer_list,*lin_layer_list]
        # combine layers into a sequential model
        self.out = torch.nn.Sequential(*layers).to(self.device)

        self.out = self.out.to(self.device)

    def forward(self, x):
        if x.dim()==2:
            x = x.unsqueeze(1)
        out = self.out(x).squeeze()       
        return out     
    