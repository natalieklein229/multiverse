"""
Fit Laplace after NN fit. (using Immer's package)

"""
# %%

from laplace import Laplace
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import glob
import copy
import gc
import pickle

from models import CCamCNN, ScaledModel
from util import scale_targets, freeze_all_but_last_n_linear

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

# %% Params
prior_precision = 1.0 #match MAP training
#prior_precision = 10000.0 # match VI

# %% data loading
train_spec = np.load('/data/0/chemcam_bnn/train_spec.npy')
val_spec = np.load('/data/0/chemcam_bnn/val_spec.npy')
test_spec = np.load('/data/0/chemcam_bnn/test_spec.npy')
train_oxides = np.load('/data/0/chemcam_bnn/train_oxides.npy')
val_oxides = np.load('/data/0/chemcam_bnn/val_oxides.npy')
test_oxides = np.load('/data/0/chemcam_bnn/test_oxides.npy')
mars_spec = np.load('/data/0/chemcam_bnn/mars_spec.npy')

# %% CNN
train_loader = DataLoader(TensorDataset(torch.from_numpy(train_spec).float(), torch.from_numpy(train_oxides).float()), 
                          batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(val_spec).float(), torch.from_numpy(val_oxides).float()), 
                        batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(test_spec).float(), torch.from_numpy(test_oxides).float()),
                        batch_size=64, shuffle=False)

orig_model = CCamCNN.load_from_checkpoint(cpath[0]).eval()
noise_prec = torch.exp(-orig_model.log_var).detach().cpu().numpy()
noise_sd = np.squeeze(np.sqrt(1/noise_prec))
log_var = orig_model.log_var.detach().cpu()
log_var.requires_grad = False

laplace_train_loader = DataLoader(TensorDataset(torch.from_numpy(train_spec).float(), 
                                                scale_targets(torch.from_numpy(train_oxides).float(),log_var)), 
                          batch_size=64, shuffle=True)

cnn_copy = copy.deepcopy(orig_model.cnn)
orig_model.to('cpu')
del orig_model
gc.collect()
torch.cuda.empty_cache()

# %% Linearized Laplace
model = ScaledModel(cnn_copy, log_var.to(device))
# Freeze all but last linear (last-layer Laplace)
freeze_all_but_last_n_linear(model,n=1)

la = Laplace(model, 'regression',
             subset_of_weights='all',
             hessian_structure='diag',
             prior_precision=prior_precision
             )
la.fit(laplace_train_loader)
#eps = 1e-4  # Or tune this if needed
#la.H += eps * torch.eye(la.H.shape[0], device=la.H.device)
#print(torch.linalg.cond(la.H))  # Should now be < 1e6 ideally

# %% Predictions
laplace_mean = []
laplace_var = []
laplace_var_noisy = []
for x, y in test_loader:
    f_mu, f_var = la(x.to(device))
    f_mu = f_mu / model.inv_std
    f_var = f_var / (model.inv_std ** 2)
    f_mu = f_mu.squeeze().detach().cpu().numpy() 
    laplace_mean.append(f_mu)
    f_sigma = f_var.squeeze().cpu().numpy() 
    f_sigma_diag = np.diagonal(f_sigma, axis1=1, axis2=2)
    laplace_var.append(f_sigma_diag) 
    pred_var = f_sigma_diag + np.square(noise_sd[None, :])
    laplace_var_noisy.append(pred_var)
    
laplace_mean = np.concatenate(laplace_mean, 0) 
laplace_var = np.concatenate(laplace_var, 0) 
laplace_var_noisy = np.concatenate(laplace_var_noisy, 0) 
# laplace_sd = np.sqrt(laplace_var)
# for i in range(len(oxides)):
#     plt.figure()
#     plt.plot(test_oxides[:, i], laplace_mean[:, i], 'ko')
#     plt.errorbar(test_oxides[:, i], laplace_mean[:, i], yerr=laplace_sd[:, i], fmt='k.', zorder=-1)
#     #plt.plot(test_oxides[:, i], cnn_pred[:, i], 'r.')
#     plt.axline([0,0], slope=1)
#     plt.title(oxides[i])
#     plt.savefig('test%d.png'%i)
#     plt.show()

# %% predict on Mars data
mars_x = torch.from_numpy(mars_spec).float().to(device)
f_mu, f_var = la(mars_x)
f_sigma = f_var.squeeze().cpu().numpy()
f_sigma_diag = np.diagonal(f_sigma, axis1=1, axis2=2) * np.square(noise_sd[None, :])
pred_var = f_sigma_diag + np.square(noise_sd[None, :])
#pred_std = np.sqrt(f_sigma_diag**2 + la.sigma_noise.item()**2)
mars_laplace_mean = f_mu.cpu().numpy() * noise_sd[None, :]

# %% save 
res = {'mean':laplace_mean, 'sd':np.sqrt(laplace_var), 'sd_noisy':np.sqrt(laplace_var_noisy),
       'mars_mean': mars_laplace_mean, 'mars_sd':np.sqrt(f_sigma_diag), 'mars_sd_noisy':np.sqrt(pred_var)}
with open('results/laplace_predictions.pkl','wb') as f:
    pickle.dump(res, f)

# %%
