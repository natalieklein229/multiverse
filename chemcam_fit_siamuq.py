"""
Fit CNN model and approx. Bayes versions to ChemCam.

TODO:
- CV for PLS
- CNN: design, add max pooling? Look at famous nets? (LeNet, AlexNet, VGG)

"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import lightning as L

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
device='cuda:0'

# %% Params
# number of targets for train and test
n_train = 330
n_val = 30
n_test = 30

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

class CCamDataSet(Dataset):
    def __init__(self, x, y):
        super(CCamDataSet, self).__init__()
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        x = self.x[index, :]
        y = self.y[index, :]
        return x, y

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

plt.figure()
plt.plot(test_oxides[:, 0], pls_y_hat[:, 0], 'k.')
plt.axline([0,0], slope=1)
plt.show()

pls_test_mse = np.mean(np.square(pls_y_hat-test_oxides))

# %% CNN

train_loader = DataLoader(CCamDataSet(train_spec, train_oxides), batch_size=64, shuffle=True)
val_loader = DataLoader(CCamDataSet(val_spec, val_oxides), batch_size=64, shuffle=True)
test_loader = DataLoader(CCamDataSet(test_spec, test_oxides), batch_size=64, shuffle=False)

cnn = CNN(in_dim=len(wav_cols), out_dim=len(oxides), ch_sizes=[32,128,1],
          krnl_sizes=[11,5,1], stride=[3,3,3], lin_l_sizes = [20, 20],
          activation='relu', device=device)

model = CCamCNN(cnn)
trainer = L.Trainer(max_epochs=50)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# %%
cnn_y_hat = trainer.predict(dataloaders=test_loader)
cnn_y_hat = torch.cat(cnn_y_hat, 0)

for i in range(len(oxides)):
    plt.figure()
    plt.plot(test_oxides[:, i], cnn_y_hat[:, i], 'ko')
    plt.plot(test_oxides[:, i], pls_y_hat[:, i], 'r.', alpha=0.7)
    plt.axline([0,0], slope=1)
    plt.title(oxides[i])
    plt.show()

cnn_test_mse = np.mean(np.square(cnn_y_hat.detach().numpy()-test_oxides))

print('PLS test mse: %0.3g, CNN test mse: %0.3g' % (pls_test_mse, cnn_test_mse))

# %%
