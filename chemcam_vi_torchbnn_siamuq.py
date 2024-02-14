"""
Fit VI after NN fit. 

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
data_path = '/home/neklein/chemcam_data/calib_2015' # TODO later: get other cal data too
#device='cuda:0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Params
# number of targets for train and test
n_train = 330
n_val = 30
n_test = 30
#n_train = 50
#n_val = 5
#n_test = 5
#n_epo_init = 200 # initial training
n_epo = 100
# priors
wp = 100.0 # note: is this just the initialization for variational psterior? is prior fixed?
# fixed noise precision -- note torchbnn does not incorporate this, just does MSE
# Role of KL weight similar?
# nprec = .1**-2
kl_weight = 1.0
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

# get test preds now
cnn_pred = model.cnn(test_loader.dataset.tensors[0][::30].to(device)).detach().cpu().numpy()

# %% torchbnn
# Convert Conv1d -> BayesConv1d
if conv_bayes:
    transform_model(model.cnn, nn.Conv1d, bnn.BayesConv1d, 
                    args={"prior_mu":0.0, "prior_sigma":1/wp, "in_channels" : ".in_channels",
                        "out_channels" : ".out_channels", "kernel_size" : ".kernel_size",
                        "stride" : ".stride", "padding" : ".padding", "bias":".bias"
                        }, 
                    attrs={"weight_mu" : ".weight"})

# Convert Linear -> BayesLinear
transform_model(model.cnn, nn.Linear, bnn.BayesLinear, 
            args={"prior_mu":0.0, "prior_sigma":1/wp, "in_features" : ".in_features",
                  "out_features" : ".out_features", "bias":".bias"
                 }, 
            attrs={"weight_mu" : ".weight"})

model.cnn.to(device)

#%% Prior to training get predictions
vi_pred_pre = []
for p in range(100):
   vi_pred_pre.append(model.cnn(test_loader.dataset.tensors[0][::30].to(device)).detach().cpu().numpy())
vi_pred = np.array(vi_pred_pre)

# %% Fitting
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)


optimizer = optim.Adam(model.parameters(), lr=lr)

for step in range(n_epo):
    for x, y in train_loader:
        pre = model(x.to(device))
        mse = mse_loss(pre, y.to(device))
        kl = kl_loss(model)
        cost = mse + kl_weight*kl
    
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    
    print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))

# %% test predictions

vi_pred = []
for p in range(100):
   vi_pred.append(model.cnn(test_loader.dataset.tensors[0][::30].to(device)).detach().cpu().numpy())
vi_pred = np.array(vi_pred)

# %%
vi_mean = np.mean(vi_pred, 0)
vi_sd = np.std(vi_pred, 0)
vi_mean_pre = np.mean(vi_pred_pre, 0)
vi_sd_pre = np.std(vi_pred_pre, 0)

for i in range(len(oxides)):
    plt.figure()
    plt.plot(test_oxides[::30, i], vi_mean[:, i], 'k.')
    plt.errorbar(test_oxides[::30, i], vi_mean[:, i], yerr=vi_sd[:, i], fmt='k.')
    plt.plot(test_oxides[::30, i], cnn_pred[:, i], 'r.')
    plt.plot(test_oxides[::30, i], vi_mean_pre[:, i], 'c.', alpha=0.7)
    plt.errorbar(test_oxides[::30, i], vi_mean_pre[:, i], yerr=vi_sd_pre[:, i], fmt='c.', alpha=0.7)
    plt.axline([0,0], slope=1)
    plt.title(oxides[i])
    plt.savefig('test%d.png'%i)
    plt.show()
# %%
