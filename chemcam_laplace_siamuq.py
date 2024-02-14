"""
Fit Laplace after NN fit. 

"""
# %%
import functools
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from functools import partial
import pyro.distributions as dist
import pyro
pyro.enable_validation(True)

import tyxe

from inference.bnn import *
import inference.guides as guides
import inference.likelihoods as likelihoods
import inference.priors as priors
from inference.util import *
from neural_nets.CNN import CNN

torch.set_float32_matmul_precision('medium')
torch.manual_seed(42)
np.random.seed(42)

vnir_range = [492.427, 849.0]
vio_range = [382.13, 473.184]
uv_range = [246.635, 338.457]
keep_shots = ['shot%d' % i for i in range(5, 50)]
oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
#device='cuda:0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Params
n_epo = 100
# priors
wp = 1.0
nprec = .1**-2

# %% functions
class CCamCNN(L.LightningModule):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.cnn(x.unsqueeze(1))
        loss = nn.functional.mse_loss(y_hat, y.to(device))
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.cnn(x.unsqueeze(1))
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def predict_step(self, batch):
        x, y = batch
        return self(x)

# %% data loading
train_spec = np.load('data/train_spec.npy')
val_spec = np.load('data/val_spec.npy')
test_spec = np.load('data/test_spec.npy')
train_oxides = np.load('data/train_oxides.npy')
val_oxides = np.load('data/val_oxides.npy')
test_oxides = np.load('data/test_oxides.npy')

# %% CNN
train_loader = DataLoader(TensorDataset(torch.from_numpy(train_spec).float(), torch.from_numpy(train_oxides).float()), 
                          batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(val_spec).float(), torch.from_numpy(val_oxides).float()), 
                        batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(test_spec).float(), torch.from_numpy(test_oxides).float()),
                        batch_size=64, shuffle=False)

cnn = CNN(in_dim=train_spec.shape[1], out_dim=len(oxides), ch_sizes=[32,128,1],
          krnl_sizes=[11,5,1], stride=[3,3,3], lin_l_sizes = [20, 20],
          activation='relu', device=device)

model = CCamCNN.load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=114-step=125235.ckpt', cnn=cnn)

# %% Linearized Laplace
cnn_laplace_copy = copy.deepcopy(model.cnn)
prior = priors.IIDPrior((dist.Normal(torch.tensor(0., device=device), torch.tensor(wp ** -0.5, device=device))))
likelihood = likelihoods.HomoskedasticGaussian(len(train_spec), precision=nprec)
laplace_bnn = LaplaceBNN(cnn_laplace_copy.to(device), prior, likelihood, approximation='subnet', S_perc=0.5).to(device)
opt = pyro.optim.ClippedAdam({"lr": 3e-4, "clip_norm": 100.0, "lrd": 0.999})
laplace_hist = laplace_bnn.fit(train_loader, opt, n_epo, num_particles=1, closed_form_kl=True, hist=True)

plt.figure()
plt.plot(laplace_hist)
plt.show()

# %%
cnn_pred = []
laplace_pred = []
for x, y in test_loader:
# TODO use data loader later
    cnn_pred.append(model.cnn(x.to(device)).detach().cpu().numpy())
    laplace_pred_ = laplace_bnn.predict(x.to(device), num_predictions=100, aggregate=False)
    laplace_pred.append(laplace_bnn.likelihood.sample(laplace_pred_).detach().cpu().numpy())
cnn_pred = np.concatenate(cnn_pred, 0)
laplace_pred = np.concatenate(laplace_pred, 0)
laplace_mean = np.mean(laplace_pred, 0)
laplace_sd = np.std(laplace_pred, 0)

for i in range(len(oxides)):
    plt.figure()
    plt.plot(test_oxides[:, i], laplace_mean[:, i], 'ko')
    plt.errorbar(test_oxides[:, i], laplace_mean[:, i], yerr=laplace_sd[:, i], fmt='k.', zorder=-1)
    plt.plot(test_oxides[:, i], cnn_pred[:, i], 'r.')
    plt.axline([0,0], slope=1)
    plt.title(oxides[i])
    plt.show()

# %% save -- maybe some issues saving pyro models... 
np.save('results/laplace_predictions.npy', laplace_pred)