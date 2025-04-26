"""
Fit Laplace after NN fit. (using Immer's package)

"""
# %%

from laplace import Laplace, marglik_training
from laplace.curvature.backpack import BackPackGGN, BackPackEF
from laplace.utils import LargestMagnitudeSubnetMask
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from tqdm import tqdm
import glob
import copy
import gc

from neural_nets.CNN import CNN
from models import CCamCNN

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
cpath = glob.glob('lightning_logs/version_0/checkpoints/*.ckpt')

# %% Params
# s_perc=0.1, defaults for prior/noise -- okay, kind of overcovers
# s_perc=0.2, defaults for prior/noise -- very similar to 0.1
# s_perc=0.5, defaults for prior/noise -- bad, nans
# Sticking with s_perc 0.2 and sigma_noise=1.0, try:
# prior_precision 0.1: all nans
# prior_precision 10.0: pretty much same as 1.0
# Keep s_perc 0.2, prior_precision 1.0, try:
# sigma_noise = 0.1: nans
# Keep s_perc 0.2, prior_precision 1.0, try:
# sigma_noise = 0.1: still bad
# sigma_noise = 0.5: still bad
# sigma_noise = 5.0: intervals way too big
# sigma_noise = 0.8: better than 1.0
# sigma_noise = 0.7: better than 0.8
# sigma_noise = 0.6: lots of nans, undercovers
# if turn up prior precision to 100.0,
# sigma_noise = 0.6: looks okish.
# sigma_noise = 0.5: good
# sigma_noise = 0.1: too small
# sigma_noise = 0.25: ok, perhaps a bit small
# sigma_noise = 0.3: close to ensemble. good overall.


#n_epo = 30
s_perc = 0.1
#s_perc = 0.2 # what used for old results
# priors
# s_perc 0.5 original results; n_particles 1
#wp = 100.0 # used in VI; not good for laplace!!!!! 
#wp = 0.01
#nprec = .1**-2
#prior_precision = 100.0 # default 1.0 - what used for old results
#sigma_noise = 0.3 # default 1.0 - what used for old results
prior_precision = 1.0 #match MAP training

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

# cnn = CNN(in_dim=train_spec.shape[1], out_dim=len(oxides), ch_sizes=[32,128,1],
#           krnl_sizes=[11,5,1], stride=[3,3,3], lin_l_sizes = [20, 20],
#           activation='relu', device=device)

orig_model = CCamCNN.load_from_checkpoint(cpath[0]).eval()
noise_prec = torch.exp(-orig_model.log_var).detach().cpu().numpy()
noise_sd = np.squeeze(np.sqrt(1/noise_prec))

class ScaledModel(nn.Module):
    def __init__(self, base_model, log_var):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('inv_std', torch.exp(-0.5 * log_var))  # shape [D]

    def forward(self, x):
        output = self.base_model(x)  # shape [batch_size, D]
        return output * self.inv_std.unsqueeze(0)  # scale predictions

def scale_targets(y, log_var):
    return y * torch.exp(-0.5 * log_var)

log_var = orig_model.log_var.detach().cpu()
log_var.requires_grad = False

laplace_train_loader = DataLoader(TensorDataset(torch.from_numpy(train_spec).float(), 
                                                scale_targets(torch.from_numpy(train_oxides).float(),log_var)), 
                          batch_size=64, shuffle=True)

cnn_copy = copy.deepcopy(orig_model.cnn)
orig_model.to('cpu')
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
print(f"Reserved memory: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
del orig_model
gc.collect()
torch.cuda.empty_cache()
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
print(f"Reserved memory: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

# %% Linearized Laplace
n_param = 25900
model = ScaledModel(cnn_copy, log_var.to(device))
#subnetwork_mask = LargestMagnitudeSubnetMask(model, n_params_subnet=int(s_perc*n_param))
#subnetwork_indices = subnetwork_mask.select().type(torch.LongTensor)
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
print(f"Reserved memory: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

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
freeze_all_but_last_linear(model)


# %%
# def weighted_nll(y_pred, y_true):
#     inv_var = torch.exp(-orig_model.log_var)   # log_var is a global or model attribute
#     return 0.5 * torch.sum(inv_var * (y_pred - y_true)**2 + orig_model.log_var)
#from laplace.curvature import AsdlGGN
la = Laplace(model, 'regression',
             subset_of_weights='all',
             hessian_structure='diag',
             prior_precision=prior_precision,
             #sigma_noise=torch.Tensor(noise_sd),
             #hessian_structure='diag'
             #subset_of_weights='subnetwork',
             #subset_of_weights='last_layer',
             #hessian_structure='lowrank',
             #subnetwork_indices=subnetwork_indices,
             #backend=AsdlGGN
             )
#del model
#gc.collect()
#torch.cuda.empty_cache()
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
print(f"Reserved memory: {torch.cuda.memory_reserved() / 1e6:.2f} MB")


# %%
la.fit(laplace_train_loader)
#eps = 1e-4  # Or tune this if needed
#la.H += eps * torch.eye(la.H.shape[0], device=la.H.device)
#print(torch.linalg.cond(la.H))  # Should now be < 1e6 ideally


# TODO check this, is this even a property
# la.likelihood_variance = 1.0

# %% Try optimizing the hyperparams - does not work with subnet
# log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
# hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
# for i in tqdm(range(n_epo)):
#     hyper_optimizer.zero_grad()
#     neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
#     neg_marglik.backward()
#     hyper_optimizer.step()

# #la.optimize_prior_precision(method='marglik')
#train_loader_tqdm = tqdm(train_loader)
#setattr(train_loader_tqdm, 'dataset', train_loader.dataset)

# la, model, margliks, losses = marglik_training(
#    model=orig_model.cnn, train_loader=train_loader, likelihood='regression',
#    hessian_structure='full', 
#    #backend = BackPackEF,
#    backend=BackPackGGN, 
#    n_epochs=n_epo, 
#    optimizer_kwargs={'lr': 0.1}, prior_structure='scalar'
#)

# plt.figure()
# plt.plot(margliks)
# plt.savefig("marglik.png")
# plt.show()

# plt.figure()
# plt.plot(losses)
# plt.savefig("loss.png")
# plt.show()

# %% TODO look a la.prior_precision or prior_precision_diag; posterior_covariance, posterior_scale; functional_variance
# prior precision ~100 (very small variance)
# posterior scale nan, look at posterior precision
# %% TODO fit and test predictions (GPU issues)
#cnn_pred = []
laplace_mean = []
laplace_var = []
for x, y in test_loader:
#    cnn_pred.append(orig_model.cnn(x.to(device)).detach().cpu().numpy())
    f_mu, f_var = la(x.to(device))
    f_mu = f_mu / model.inv_std
    f_var = f_var / (model.inv_std ** 2)
    f_mu = f_mu.squeeze().detach().cpu().numpy() #* noise_sd[None, :]
    f_sigma = f_var.squeeze().cpu().numpy() 
    f_sigma_diag = np.diagonal(f_sigma, axis1=1, axis2=2) #* np.square(noise_sd[None, :])
    pred_var = f_sigma_diag + np.square(noise_sd[None, :])
    #pred_std = np.sqrt(f_sigma_diag**2 + la.sigma_noise.item()**2)
    #pred_std = np.sqrt(f_sigma_diag**2 + 1/nprec) # fixed nprec
    laplace_mean.append(f_mu)
    laplace_var.append(pred_var)
#cnn_pred = np.concatenate(cnn_pred, 0)

# %%
laplace_mean = np.concatenate(laplace_mean, 0) 
laplace_var = np.concatenate(laplace_var, 0) 
laplace_sd = np.sqrt(laplace_var)


for i in range(len(oxides)):
    plt.figure()
    plt.plot(test_oxides[:, i], laplace_mean[:, i], 'ko')
    plt.errorbar(test_oxides[:, i], laplace_mean[:, i], yerr=laplace_sd[:, i], fmt='k.', zorder=-1)
    #plt.plot(test_oxides[:, i], cnn_pred[:, i], 'r.')
    plt.axline([0,0], slope=1)
    plt.title(oxides[i])
    plt.savefig('test%d.png'%i)
    plt.show()

# %% predict on Mars data
mars_x = torch.from_numpy(mars_spec).float().to(device)
f_mu, f_var = la(mars_x)
f_sigma = f_var.squeeze().cpu().numpy()
f_sigma_diag = np.diagonal(f_sigma, axis1=1, axis2=2) * np.square(noise_sd[None, :])
pred_var = f_sigma_diag + np.square(noise_sd[None, :])
#pred_std = np.sqrt(f_sigma_diag**2 + la.sigma_noise.item()**2)
mars_laplace_mean = f_mu.cpu().numpy() * noise_sd[None, :]
mars_laplace_sd = np.sqrt(pred_var)

# %% save -- maybe some issues saving pyro models... 
np.save('results/laplace_mean_predictions.npy', laplace_mean)
np.save('results/laplace_sd_predictions.npy', laplace_sd)
np.save('results/laplace_mars_mean_predictions.npy', mars_laplace_mean)
np.save('results/laplace_mars_sd_predictions.npy', mars_laplace_sd)
# %%
