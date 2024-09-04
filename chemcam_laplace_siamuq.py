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
s_perc = 0.2
# priors
# s_perc 0.5 original results; n_particles 1
#wp = 100.0 # used in VI; not good for laplace!!!!! 
#wp = 0.01
#nprec = .1**-2
prior_precision = 100.0 # default 1.0
sigma_noise = 0.3 # default 1.0

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

orig_model = CCamCNN.load_from_checkpoint('lightning_logs/version_1/checkpoints/epoch=74-step=105525.ckpt', cnn=cnn)

# %% Linearized Laplace
n_param = 25900
subnetwork_mask = LargestMagnitudeSubnetMask(orig_model.cnn, n_params_subnet=int(s_perc*n_param))
subnetwork_indices = subnetwork_mask.select().type(torch.LongTensor)

la = Laplace(orig_model.cnn, 'regression',
             #subset_of_weights='all',
             hessian_structure='full',
             prior_precision=prior_precision,
             sigma_noise=sigma_noise,
             #hessian_structure='diag'
             subset_of_weights='subnetwork',
             #subset_of_weights='last_layer',
             #hessian_structure='lowrank',
             subnetwork_indices=subnetwork_indices
             )
la.fit(train_loader)

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
# %%
cnn_pred = []
laplace_mean = []
laplace_sd = []
for x, y in test_loader:
    cnn_pred.append(orig_model.cnn(x.to(device)).detach().cpu().numpy())
    f_mu, f_var = la(x.to(device))
    f_mu = f_mu.squeeze().detach().cpu().numpy()
    f_sigma = f_var.squeeze().sqrt().cpu().numpy()
    f_sigma_diag = np.diagonal(f_sigma, axis1=1, axis2=2)
    pred_std = np.sqrt(f_sigma_diag**2 + la.sigma_noise.item()**2)
    #pred_std = np.sqrt(f_sigma_diag**2 + 1/nprec) # fixed nprec
    laplace_mean.append(f_mu)
    laplace_sd.append(pred_std)
cnn_pred = np.concatenate(cnn_pred, 0)

# %%
laplace_mean = np.concatenate(laplace_mean, 0) 
laplace_sd = np.concatenate(laplace_sd, 0) 

for i in range(len(oxides)):
    plt.figure()
    plt.plot(test_oxides[:, i], laplace_mean[:, i], 'ko')
    plt.errorbar(test_oxides[:, i], laplace_mean[:, i], yerr=laplace_sd[:, i], fmt='k.', zorder=-1)
    plt.plot(test_oxides[:, i], cnn_pred[:, i], 'r.')
    plt.axline([0,0], slope=1)
    plt.title(oxides[i])
    plt.savefig('test%d.png'%i)
    plt.show()

# %% predict on Mars data
mars_x = torch.from_numpy(mars_spec).float().to(device)
f_mu, f_var = la(mars_x)
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
f_sigma_diag = np.diagonal(f_sigma, axis1=1, axis2=2)
pred_std = np.sqrt(f_sigma_diag**2 + la.sigma_noise.item()**2)
mars_laplace_mean = f_mu.cpu().numpy()
mars_laplace_sd = pred_std

# %% save -- maybe some issues saving pyro models... 
np.save('results/laplace_mean_predictions.npy', laplace_mean)
np.save('results/laplace_sd_predictions.npy', laplace_sd)
np.save('results/laplace_mars_mean_predictions.npy', mars_laplace_mean)
np.save('results/laplace_mars_sd_predictions.npy', mars_laplace_sd)
# %%
