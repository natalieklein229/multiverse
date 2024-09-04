"""
Fit VI after NN fit. 

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
data_path = '/home/neklein/chemcam_data/calib_2015' # TODO later: get other cal data too
#device='cuda:0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Params
# number of targets for train and test
# n_train = 330
# n_val = 30
# n_test = 30
n_train = 330
n_val = 30
n_test = 30
#n_epo_init = 200 # initial training
n_epo = 30
# priors
wp = 10.0
nprec = .1**-2

# %% functions
def mask_norm3(wav, x):
    uv_mask = np.logical_and(wav>uv_range[0], wav<uv_range[1])
    vio_mask = np.logical_and(wav>vio_range[0], wav<vio_range[1])
    vnir_mask = np.logical_and(wav>vnir_range[0], wav<vnir_range[1])
    mask = np.logical_or.reduce((uv_mask, vio_mask, vnir_mask))
    for m in [uv_mask, vio_mask, vnir_mask]:
        x[:, m] = x[:, m]/np.max(x[:, m], 1, keepdims=True)
    x = x[:, mask]
    wav = wav[mask]
    return wav, x

def load_data(data_path, targets):
    data_dict = {'target':[], 'wav':[]} | {o: [] for o in oxides}
    for i in range(len(targets)):
        f_list = os.listdir('%s/%s' % (data_path, targets[i]))
        for j in range(len(f_list)):
            f = pd.read_csv('%s/%s/%s' % (data_path, targets[i], f_list[j]))
            wav = f['# wave'].values
            x = f[keep_shots].values.T
            wav, x = mask_norm3(wav, x)
            data_dict['wav'].append(x)
            data_dict['target'].append([f_list[j]]*len(x))
            for o in oxides:
                if targets[i] in comp['target'].values:
                    v = comp.loc[comp['target']==targets[i],o].values.astype(float)[0]
                else:
                    v = np.nan
                data_dict[o].append([v]*len(x))
    data_dict['wav'] = np.concatenate(data_dict['wav'], 0)
    data_dict['target'] = np.concatenate(data_dict['target'])
    for o in oxides:
        data_dict[o] = np.concatenate(data_dict[o])
    data = pd.DataFrame(data_dict['wav'], columns=['wav%d' % i for i in range(data_dict['wav'].shape[1])])
    data['target'] = data_dict['target']
    for o in oxides:
        data[o] = data_dict[o]
    data = data.dropna()
    return data

# class CCamDataSet(Dataset):
#     def __init__(self, x, y):
#         super(CCamDataSet, self).__init__()
#         self.x = torch.from_numpy(x).float()
#         self.y = torch.from_numpy(y).float()

#     def __len__(self):
#         return self.x.shape[0]
    
#     def __getitem__(self, index):
#         x = self.x[index, :]
#         y = self.y[index, :]
#         return x, y

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
# TODO: how to merge targets and spectra? some missing.
comp = pd.read_csv('/home/neklein/chemcam_data/ccam_calibration_compositions.csv',nrows=572)
comp['Spectrum Name'] = comp['Spectrum Name'].astype(str)
comp['target'] = comp['Spectrum Name'].apply(lambda x: x.lower().replace('-','').replace('_','').replace(' ',''))

targ_dirs = os.listdir(data_path)
targ_dirs = np.array([a for a in targ_dirs if a in comp['target'].values])

targ_ind = np.random.choice(len(targ_dirs), n_train+n_val+n_test, replace=False)
train_targ = targ_dirs[targ_ind[:n_train]]
val_targ = targ_dirs[targ_ind[n_train:(n_train+n_val)]]
test_targ = targ_dirs[targ_ind[(n_train+n_val):]]

train_data = load_data(data_path, train_targ)
val_data = load_data(data_path, val_targ)
test_data = load_data(data_path, test_targ)

# a little other normalization
wav_cols = [c for c in train_data.columns if 'wav' in c]
train_spec = train_data[wav_cols].values
val_spec = val_data[wav_cols].values
test_spec = test_data[wav_cols].values
spec_max = np.max(train_spec)
train_spec /= spec_max
val_spec /= spec_max
test_spec /= spec_max
train_spec = np.log(train_spec + 0.5)
val_spec = np.log(val_spec + 0.5)
test_spec = np.log(test_spec + 0.5)
train_oxides = train_data[oxides].values / 100.0
oxide_sd = np.std(train_oxides, 0, keepdims=True)
train_oxides /= oxide_sd
val_oxides = (val_data[oxides].values / 100.0) / oxide_sd
test_oxides = (test_data[oxides].values / 100.0) / oxide_sd

# %% PLS baseline
pls = PLSRegression(n_components=15).fit(train_spec, train_oxides)
pls_y_hat = pls.predict(test_spec)

# plt.figure()
# plt.plot(test_oxides[:, 0], pls_y_hat[:, 0], 'k.')
# plt.axline([0,0], slope=1)
# plt.show()

pls_test_mse = np.mean(np.square(pls_y_hat-test_oxides))

# %% CNN

train_loader = DataLoader(TensorDataset(torch.from_numpy(train_spec).float(), torch.from_numpy(train_oxides).float()), 
                          batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(val_spec).float(), torch.from_numpy(val_oxides).float()), 
                        batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(test_spec).float(), torch.from_numpy(test_oxides).float()),
                        batch_size=64, shuffle=False)

cnn = CNN(in_dim=len(wav_cols), out_dim=len(oxides), ch_sizes=[32,128,1],
          krnl_sizes=[11,5,1], stride=[3,3,3], lin_l_sizes = [20, 20],
          activation='relu', device=device)

model = CCamCNN.load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=114-step=125235.ckpt', cnn=cnn)


# %% Variational TyXe - see examples/resnet.py
def callback(b, i, avg_elbo):
    avg_err, avg_ll = 0., 0.
    b.eval()
    for x, y in iter(val_loader):
        err, ll = b.evaluate(x.to(device), y.to(device), num_predictions=20)
        avg_err += err / len(val_loader.sampler)
        avg_ll += ll / len(val_loader.sampler)
    print(f"ELBO={avg_elbo}; val error={100 * avg_err:.2f}%; LL={avg_ll:.4f}")
    b.train()

cnn_vi_copy = copy.deepcopy(model.cnn)
cnn_vi_copy2 = copy.deepcopy(model.cnn)
prior = priors.IIDPrior((dist.Normal(torch.tensor(0., device=device), torch.tensor(wp ** -0.5, device=device))))
likelihood = tyxe.likelihoods.HomoskedasticGaussian(dataset_size=len(train_spec),scale=1/nprec)
#guide = tyxe.guides.ParameterwiseDiagonalNormal
guide = functools.partial(tyxe.guides.AutoNormal,
                          init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(cnn_vi_copy2.to(device), prefix="net"), 
                          init_scale=0.1, #max_guide_scale=100.0, 
                          train_loc=True) # False means scale only

vi_bnn = tyxe.VariationalBNN(cnn_vi_copy.to(device), prior, likelihood, guide)
vi_pred_pre = vi_bnn.predict(test_loader.dataset.tensors[0][::30].to(device), num_predictions=100, aggregate=False).detach().cpu().numpy()

opt = pyro.optim.Adam({"lr": 1e-2})
with tyxe.poutine.local_reparameterization():
    vi_bnn.fit(train_loader, opt, n_epo, callback=callback, device=device)

# %% TODO it seems to at least work better than Variatonal BNN. uhh, or not?
# But the uncertainty is tiny, why?
    # I
cnn_pred = model.cnn(test_loader.dataset.tensors[0][::30].to(device)).detach().cpu().numpy()

vi_pred_ = vi_bnn.predict(test_loader.dataset.tensors[0][::30].to(device), num_predictions=100, aggregate=False)

vi_pred = vi_pred_.detach().cpu().numpy()
#vi_pred = vi_bnn.likelihood.sample(vi_pred_).detach().cpu().numpy()
vi_mean = np.mean(vi_pred, 0)
vi_sd = np.std(vi_pred, 0)

vi_mean_pre = np.mean(vi_pred_pre, 0)
vi_sd_pre = np.std(vi_pred_pre, 0)

for i in range(len(oxides)):
    plt.figure()
    plt.plot(test_oxides[::30, i], vi_mean[:, i], 'k.')
    plt.errorbar(test_oxides[::30, i], vi_mean[:, i], yerr=vi_sd[:, i], fmt='k.')
    plt.plot(test_oxides[::30, i], vi_mean_pre[:, i], 'c.', alpha=0.7)
    plt.errorbar(test_oxides[::30, i], vi_mean_pre[:, i], yerr=vi_sd_pre[:, i], fmt='c.', alpha=0.7)
    plt.plot(test_oxides[::30, i], cnn_pred[:, i], 'r.')
    plt.axline([0,0], slope=1)
    plt.title(oxides[i])
    plt.savefig('test%d.png'%i)
    plt.show()

# # %% Variational TODO still some device errors sometimes, why
# cnn_vi_copy = copy.deepcopy(model.cnn)
# # was learning rate too low?
# opt = pyro.optim.ClippedAdam({"lr": 1e-1, "clip_norm": 100.0, "lrd": 0.999})

# # This guide does not work well! why not?
# #guide = partial(guides.AutoNormal, init_scale=1.0, init_loc_fn=guides.PretrainedInitializer.from_net(cnn_vi_copy.to(device)))
# guide = guides.AutoNormal
# prior = priors.IIDPrior((dist.Normal(torch.tensor(0., device=device), torch.tensor(wp ** -0.5, device=device))))
# likelihood = likelihoods.HomoskedasticGaussian(len(train_spec), precision=nprec)
# vi_bnn = VariationalBNN(cnn_vi_copy.to(device), prior, likelihood, guide).to(device)
# vi_hist, vi_hist_test = vi_bnn.fit(train_loader, opt, n_epo, hist=True, closed_form_kl=True, test_loader=val_loader)

# plt.figure()
# plt.plot(vi_hist)
# plt.plot(vi_hist_test)
# plt.show()

# # %%
# vi_pred_ = vi_bnn.predict(test_loader.dataset.tensors[0][::30].to(device), num_predictions=20, aggregate=False)
# vi_pred = vi_bnn.likelihood.sample(vi_pred_).detach().cpu().numpy()

# vi_mean = np.mean(vi_pred, 0)
# vi_sd = np.std(vi_pred, 0)

# for i in range(len(oxides)):
#     plt.figure()
#     plt.plot(test_oxides[::30, i], vi_mean[:, i], 'k.')
#     plt.errorbar(test_oxides[::30, i], vi_mean[:, i], yerr=vi_sd[:, i], fmt='k.')
#     plt.plot(test_oxides[::30, i], cnn_y_hat[::30, i], 'r.')
#     plt.axline([0,0], slope=1)
#     plt.title(oxides[i])
#     plt.show()
# %%
