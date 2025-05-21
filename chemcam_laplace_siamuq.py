"""
Fit Laplace after NN fit. (using Immer's package)

# TODO: if don't do scaling, what happens?
# TODO: last_layer has better condition (still large but...), but then prediction fails
# TODO: trying unscaled leads to weird shape error with  model -- check what is going on!
# TODO: try old subnetwork code

"""
# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from laplace import Laplace
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from laplace.utils import LargestMagnitudeSubnetMask
import glob
import copy
import gc
import pickle

from models import CCamCNN, ScaledModel
from util import scale_targets, freeze_all_but_last_n_linear, freeze_all_but_first_layer, unfreeze_only_third_conv, unfreeze_second_to_last_layer

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
#hs = 'lowrank'
hs = 'full'
#hs = 'diag'
#hs = 'kron'
#sub = 'last_layer'
#sub = 'all'
sub = 'subnetwork'
s_perc = 0.2
#prior_precision = 1.0 #match MAP training
prior_precision = 10000.0 # match VI
#prior_precision = 1/10000 # try much larger variance...

# %% data loading
train_spec = np.load('/data/0/chemcam_bnn/train_spec.npy')
val_spec = np.load('/data/0/chemcam_bnn/val_spec.npy')
test_spec = np.load('/data/0/chemcam_bnn/test_spec.npy')
train_oxides = np.load('/data/0/chemcam_bnn/train_oxides.npy')
val_oxides = np.load('/data/0/chemcam_bnn/val_oxides.npy')
test_oxides = np.load('/data/0/chemcam_bnn/test_oxides.npy')
mars_spec = np.load('/data/0/chemcam_bnn/mars_spec.npy')

# %% CNN
orig_model = CCamCNN.load_from_checkpoint(cpath[0]).eval()
noise_prec = torch.exp(-orig_model.log_var).detach().cpu().numpy()
noise_sd = np.squeeze(np.sqrt(1/noise_prec))
log_var = orig_model.log_var.detach().cpu()
log_var.requires_grad = False

train_y = scale_targets(torch.from_numpy(train_oxides).float(),log_var)
#y_mean = train_y.mean(0,keepdim=True)
#y_sd = train_y.std(0,keepdim=True)
y_mean = 0
y_sd = 1

train_loader = DataLoader(TensorDataset(torch.from_numpy(train_spec).float(), 
                                        (scale_targets(torch.from_numpy(train_oxides).float(),log_var)-y_mean)/y_sd), 
                          batch_size=64, shuffle=True)
noscl_train_loader = DataLoader(TensorDataset(torch.from_numpy(train_spec).float(), 
                                        torch.from_numpy(train_oxides).float()), 
                          batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(val_spec).float(),
                                      (scale_targets(torch.from_numpy(val_oxides).float(),log_var)-y_mean)/y_sd), 
                        batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(test_spec).float(), 
                                       torch.from_numpy(test_oxides).float()),
                        batch_size=64, shuffle=False)

cnn_copy = copy.deepcopy(orig_model.cnn)
orig_model.to('cpu')
del orig_model
gc.collect()
torch.cuda.empty_cache()

# %% Linearized Laplace
model = ScaledModel(cnn_copy, log_var.to(device))
#model = cnn_copy
# Freeze all but last linear (last-layer Laplace)
if sub != 'subnetwork':
    freeze_all_but_last_n_linear(model,n=1)
    #freeze_all_but_first_layer(model)
    #unfreeze_only_third_conv(model)
    #unfreeze_second_to_last_layer(model)
for name, param in model.named_parameters():
    print(f"Layer: {name}, Frozen: {not param.requires_grad}")
    print(f"Layer: {name}, Parameters: {param.shape}")

# %%
#from laplace.curvature.curvature import GGNInterface, AsdlGGN
if sub == 'subnetwork':
    n_param = 25900
    subnetwork_mask = LargestMagnitudeSubnetMask(model, n_params_subnet=int(s_perc*n_param))
    subnetwork_indices = subnetwork_mask.select().type(torch.LongTensor)

    la = Laplace(model, 'regression',
                subset_of_weights=sub,
                hessian_structure=hs,
                prior_precision=prior_precision,
                subnetwork_indices=subnetwork_indices
    #             backend=GGNInterface
                )
else:
    la = Laplace(model, 'regression',
                subset_of_weights=sub,
                hessian_structure=hs,
                prior_precision=prior_precision
    #             backend=GGNInterface
                )

la.fit(train_loader)

if hs == 'diag':
    print(torch.linalg.cond(torch.diag(la.H)))  # Should now be < 1e6 ideally
    # eps = 1e-4  # Or tune this if needed
    # #la.H += eps * torch.eye(la.H.shape[0], device=la.H.device)
    # print(torch.linalg.cond(torch.diag(la.H)+eps * torch.eye(la.H.shape[0], device=la.H.device)))  # Should now be < 1e6 ideally
    if torch.linalg.cond(torch.diag(la.H)) > 1e6:
        diag_hess_clipped = torch.clip(la.H, 1e-1, 1e5)
        la.H = diag_hess_clipped
        #la.H[la.H < 1e-3] = 1e-3
        print(torch.linalg.cond(torch.diag(la.H)))  # Should now be < 1e6 ideally
elif hs == 'full':
    print(torch.linalg.cond(la.H))  # Should now be < 1e6 ideally
elif hs == 'kron':
    print(torch.linalg.cond(la.H.to_matrix()))

# tuen prior precision
#la.optimize_prior_precision(n_steps=1000,val_loader=laplace_val_loader,method = 'gridsearch',
#                            log_prior_prec_min=-6)
#la.optimize_prior_precision(n_steps=1000,log_prior_prec_min=-8,log_prior_prec_max=10)
#print('optimized prior var: %0.4g' % (1/la.prior_precision))
#print('optimized log prior prec: %0.4g' % (torch.log(la.prior_precision)))


# %% Predictions
def trans_lapred(x):
    f_mu, f_var = la(x.to(device))
    f_mu = f_mu / model.inv_std
    f_mu = f_mu.squeeze().detach().cpu().numpy() 
    f_sigma = f_var.squeeze().cpu().numpy() 
    noise_diag = np.diag(1./model.inv_std.cpu().numpy())[None, :, :]
    f_sigma = noise_diag @ f_sigma @ noise_diag
    f_sigma_diag = np.diagonal(f_sigma, axis1=1, axis2=2)
    return f_mu, f_sigma_diag

laplace_mean = []
laplace_var = []
laplace_var_noisy = []
for x, y in test_loader:
    f_mu, f_sigma_diag = trans_lapred(x)
    laplace_mean.append(f_mu)
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
mars_x = torch.from_numpy(mars_spec).float()
f_mu_mars, f_sigma_diag_mars = trans_lapred(mars_x)
pred_var_mars = f_sigma_diag_mars + np.square(noise_sd[None, :])

# %% save 
res = {'mean':laplace_mean, 'sd':np.sqrt(laplace_var), 'sd_noisy':np.sqrt(laplace_var_noisy)}#,
       #'mars_mean': f_mu_mars, 'mars_sd':np.sqrt(f_sigma_diag_mars), 'mars_sd_noisy':np.sqrt(pred_var_mars)}
with open('results/laplace_predictions.pkl','wb') as f:
    pickle.dump(res, f)

# %%
