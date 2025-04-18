"""
Fit VI after NN fit. 
TODO: check fits/hyerparams, include nprec in sampling like with laplace

"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import copy
import lightning as L
import pyro
pyro.enable_validation(True)

import torchbnn as bnn
from torchhk import transform_model
from neural_nets.CNN import CNN

torch.set_float32_matmul_precision('medium')
torch.manual_seed(42)
np.random.seed(42)

vnir_range = [492.427, 849.0]
vio_range = [382.13, 473.184]
uv_range = [246.635, 338.457]
keep_shots = ['shot%d' % i for i in range(5, 50)]
oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Params
n_epo = 100
# priors
wp = 100.0 # note: is this just the initialization for variational psterior? is prior fixed?
# fixed noise precision -- note torchbnn does not incorporate this, just does MSE
# Role of KL weight similar?
nprec = .3**-2 # was 0.1
kl_weight = 0.1
lr = 1e-3 
# large lr = jumping to bad place immediately, probably getting stuck there
# note that with this wp=100.0, the pre predictions (essentially prior draws) are goodish
# changing wp has bad effects on convergence
# changing kl weight? seems like higher gets better UQ... possibly worse MSE, noisier training

# linear only: looks kind of reasonable (still UQ too small)
# if include conv, prior result looks kinda weird but ends up ok. 
# note it is overriding the pretrained model but it kind of ends up ok. 
conv_bayes = True # convert conv to bayes or linaer only

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
mars_spec = np.load('data/mars_spec.npy')
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

model = CCamCNN.load_from_checkpoint('lightning_logs/version_1/checkpoints/epoch=74-step=105525.ckpt', cnn=cnn)

cnn_vi_copy = copy.deepcopy(model.cnn)

# %% torchbnn
# Convert Conv1d -> BayesConv1d
if conv_bayes:
    transform_model(cnn_vi_copy, nn.Conv1d, bnn.BayesConv1d, 
                    args={"prior_mu":0.0, "prior_sigma":1/wp, "in_channels" : ".in_channels",
                        "out_channels" : ".out_channels", "kernel_size" : ".kernel_size",
                        "stride" : ".stride", "padding" : ".padding", "bias":".bias"
                        }, 
                    attrs={"weight_mu" : ".weight"})

# Convert Linear -> BayesLinear
transform_model(cnn_vi_copy, nn.Linear, bnn.BayesLinear, 
            args={"prior_mu":0.0, "prior_sigma":1/wp, "in_features" : ".in_features",
                  "out_features" : ".out_features", "bias":".bias"
                 }, 
            attrs={"weight_mu" : ".weight"})

cnn_vi_copy.to(device)

#%% Prior to training, get predictions
vi_pred_pre = []
for x, y in test_loader:
    tmp = []
    for p in range(100):
        pred = cnn_vi_copy(x.to(device)).unsqueeze(0).detach().cpu().numpy() + np.random.normal(scale=1/np.sqrt(nprec))
        tmp.append(pred)
    tmp = np.concatenate(tmp, 0)
    vi_pred_pre.append(tmp)
vi_pred_pre = np.concatenate(vi_pred_pre, 1) 

# %% Fitting
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
optimizer = optim.Adam(cnn_vi_copy.parameters(), lr=lr)

for step in range(n_epo):
    for x, y in train_loader:
        pre = cnn_vi_copy(x.to(device))
        mse = mse_loss(pre, y.to(device))
        kl = kl_loss(cnn_vi_copy)
        cost = mse + kl_weight*kl
    
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    
    print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))

# %% test predictions
vi_pred = []
cnn_pred = []
for x, y in test_loader:
    cnn_pred.append(model.cnn(x.to(device)).detach().cpu().numpy())
    tmp = []
    for p in range(100):
        pred = cnn_vi_copy(x.to(device)).unsqueeze(0).detach().cpu().numpy() + np.random.normal(scale=1/np.sqrt(nprec))
        tmp.append(pred)
    tmp = np.concatenate(tmp, 0)
    vi_pred.append(tmp)
cnn_pred = np.concatenate(cnn_pred, 0)
vi_pred = np.concatenate(vi_pred, 1) 

# %%
vi_mean = np.mean(vi_pred, 0)
vi_sd = np.std(vi_pred, 0)
vi_mean_pre = np.mean(vi_pred_pre, 0)
vi_sd_pre = np.std(vi_pred_pre, 0)

for i in range(len(oxides)):
    plt.figure()
    plt.plot(test_oxides[:, i], vi_mean[:, i], 'k.')
    plt.errorbar(test_oxides[:, i], vi_mean[:, i], yerr=vi_sd[:, i], fmt='k.')
    plt.plot(test_oxides[:, i], cnn_pred[:, i], 'r.')
    #plt.plot(test_oxides[:, i], vi_mean_pre[:, i], 'c.', alpha=0.7)
    #plt.errorbar(test_oxides[:, i], vi_mean_pre[:, i], yerr=vi_sd_pre[:, i], fmt='c.', alpha=0.7)
    plt.axline([0,0], slope=1)
    plt.title(oxides[i])
    plt.savefig('test%d.png' % i)
    plt.show()

# %% Mars perdictions
mars_x = torch.from_numpy(mars_spec).float().to(device)
tmp = []
for p in range(100):
    pred = cnn_vi_copy(mars_x).unsqueeze(0).detach().cpu().numpy() + np.random.normal(scale=1/np.sqrt(nprec))
    tmp.append(pred)
tmp = np.concatenate(tmp, 0)
mars_vi_pred = tmp

# %% save -- maybe some issues saving pyro models... 
np.save('results/vi_predictions.npy', vi_pred)
np.save('results/vi_mars_predictions.npy', mars_vi_pred)

# %%
