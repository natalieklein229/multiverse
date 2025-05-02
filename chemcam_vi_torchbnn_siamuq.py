"""
Fit VI after NN fit using fixed noise variance (via rescaling) and initializing at CNN fit.

"""
# %%
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
import glob
import pickle

import torchbnn as bnn
from torchhk import transform_model
from models import CCamCNN, ScaledModel
from util import scale_targets

torch.set_float32_matmul_precision('medium')
torch.manual_seed(42)
np.random.seed(42)

vnir_range = [492.427, 849.0]
vio_range = [382.13, 473.184]
uv_range = [246.635, 338.457]
keep_shots = ['shot%d' % i for i in range(5, 50)]
oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpath = glob.glob('lightning_logs/version_0/checkpoints/*.ckpt')
batch_size = 256

# %% Params
n_epo = 30 
n_samp = 1000 # samples to save predictions
wp = 100.0 # note: bad results if use 1.0
kl_weight = 1e-4 
lr = 1e-3 
conv_bayes = True # convert conv to bayes or linaer only

# %% data loading
train_spec = np.load('/data/0/chemcam_bnn/train_spec.npy')
val_spec = np.load('/data/0/chemcam_bnn/val_spec.npy')
test_spec = np.load('/data/0/chemcam_bnn/test_spec.npy')
mars_spec = np.load('/data/0/chemcam_bnn/mars_spec.npy')
train_oxides = np.load('/data/0/chemcam_bnn/train_oxides.npy')
val_oxides = np.load('/data/0/chemcam_bnn/val_oxides.npy')
test_oxides = np.load('/data/0/chemcam_bnn/test_oxides.npy')

# %% CNN
orig_model = CCamCNN.load_from_checkpoint(cpath[0]).eval()
noise_prec = torch.exp(-orig_model.log_var).detach().cpu().numpy()
noise_sd = np.squeeze(np.sqrt(1/noise_prec))
log_var = orig_model.log_var.detach().cpu()
log_var.requires_grad = False

train_loader = DataLoader(TensorDataset(torch.from_numpy(train_spec).float(), 
                                        scale_targets(torch.from_numpy(train_oxides).float(),log_var)), 
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(val_spec).float(),
                                      scale_targets(torch.from_numpy(val_oxides).float(),log_var)), 
                        batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(test_spec).float(), 
                                       torch.from_numpy(test_oxides).float()),
                        batch_size=64, shuffle=False)

cnn_vi_copy = copy.deepcopy(orig_model.cnn)
orig_model.to('cpu')
del orig_model
gc.collect()
torch.cuda.empty_cache()

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

model = ScaledModel(cnn_vi_copy.to(device), log_var.to(device))

# %% Fitting
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=True)
optimizer = optim.Adam(model.parameters(), lr=lr)

for step in range(n_epo):
    for x, y in train_loader:
        pre = model(x.to(device))
        mse = mse_loss(pre, y.to(device))
        kl = kl_loss(model.base_model)
        cost = mse + kl_weight*kl
    
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    
    print('%d - MSE : %2.2f, KL : %2.2f' % (step, mse.item(), kl.item()))

# %% test predictions
vi_pred = []
vi_pred_noisy = []
for x, y in test_loader:
    tmp = []
    tmp_noisy = []
    for p in range(n_samp):
        pred = model(x.to(device)).detach()/model.inv_std
        tmp.append(pred[:, None, :].cpu().numpy())
        pred = pred.cpu().numpy() + np.random.normal(scale=noise_sd[None, :], size=pred.shape)
        tmp_noisy.append(pred[:, None, :])
    tmp = np.concatenate(tmp, 1)
    tmp_noisy = np.concatenate(tmp_noisy, 1)
    vi_pred.append(tmp)
    vi_pred_noisy.append(tmp_noisy)
vi_pred = np.concatenate(vi_pred, 0) 
vi_pred_noisy = np.concatenate(vi_pred_noisy, 0) 

# %%
# vi_mean = np.mean(vi_pred, 1)
# vi_sd = np.std(vi_pred, 1)
# #vi_mean_pre = np.mean(vi_pred_pre, 0)
# #vi_sd_pre = np.std(vi_pred_pre, 0)

# for i in range(len(oxides)):
#     plt.figure()
#     plt.plot(test_oxides[:, i], vi_mean[:, i], 'k.')
#     plt.errorbar(test_oxides[:, i], vi_mean[:, i], yerr=vi_sd[:, i], fmt='k.')
#     #plt.plot(test_oxides[:, i], cnn_pred[:, i], 'r.')
#     #plt.plot(test_oxides[:, i], vi_mean_pre[:, i], 'c.', alpha=0.7)
#     #plt.errorbar(test_oxides[:, i], vi_mean_pre[:, i], yerr=vi_sd_pre[:, i], fmt='c.', alpha=0.7)
#     plt.axline([0,0], slope=1)
#     plt.title(oxides[i])
#     plt.savefig('test%d.png' % i)
#     plt.show()

# %% Mars predictions
mars_x = torch.from_numpy(mars_spec).float().to(device)
mars_pred = []
mars_pred_noisy = []
for p in range(n_samp):
    pred = model(mars_x.to(device)).detach()/model.inv_std
    mars_pred.append(pred[:, None, :].cpu().numpy())
    pred = pred.cpu().numpy() + np.random.normal(scale=noise_sd[None, :], size=pred.shape)
    mars_pred_noisy.append(pred[:, None, :])
mars_pred = np.concatenate(mars_pred, 1)
mars_pred_noisy = np.concatenate(mars_pred_noisy, 1)

# %% save 
res = {'pred': vi_pred, 'pred_noisy': vi_pred_noisy,
       'mars_pred': mars_pred, 'mars_pred_noisy': mars_pred_noisy
       }
with open('results/vi_predictions.pkl','wb') as f:
    pickle.dump(res, f)

# %%
