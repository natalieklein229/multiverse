"""
Evaluate results from CNN, PLS, laplace, VI.

TODO:
- figures and tables

"""
# %% 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torchview import draw_graph

sns.set_theme(context='talk')

from neural_nets.CNN import CNN

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# %%
def rmse(y, yhat):
    if yhat.ndim == 3:
        yhat = np.mean(yhat, 0)
    return np.sqrt(np.mean(np.square(y-yhat),0))

def coverage(y, yhat):
    yhat_mean = np.mean(yhat, 0)
    yhat_sd = np.std(yhat, 0)
    c = np.logical_and(yhat_mean-2*yhat_sd <= y, yhat_mean+2*yhat_sd >= y)
    return np.mean(c, 0)

def width(yhat):
    yhat_sd = np.std(yhat, 0)
    return np.mean(2*yhat_sd,0)

# %%
oxide_sd = np.load('data/oxide_sd.npy')
test = np.load('data/test_oxides.npy')
cnn_pred = np.load('results/cnn_predictions.npy')
pls_pred = np.load('results/PLS_predictions.npy')
laplace_mean = np.load('results/laplace_mean_predictions.npy')
laplace_sd = np.load('results/laplace_sd_predictions.npy')
vi_pred = np.load('results/vi_predictions.npy')
ensemble_pred = []
for i in np.arange(1, 31):
    tmp = np.load('results/cnn%d_predictions.npy' % i)
    ensemble_pred.append(tmp)
ensemble_pred = np.array(ensemble_pred)
laplace_pred = []
for i in range(100):
    laplace_pred.append(np.random.normal(loc=laplace_mean, scale=laplace_sd))
laplace_pred = np.array(laplace_pred) # (100, 253, 9)

for d, n in zip([cnn_pred, pls_pred, vi_pred, ensemble_pred], ['CNN', 'PLS', 'VI', 'Ensemble']):
    print(n)
    print('RMSE')
    print(np.round(rmse(test,d)*oxide_sd*100,2))
    if n in ['VI', 'Ensemble']:
        print('Cov')
        print(np.round(coverage(test,d),2))
        print('Width')
        print(np.round(width(d)*oxide_sd*100,2))
        print('\n')
print('Laplace')
print('nans')
print(np.sum(np.any(np.isnan(laplace_sd),1)))
good_ix = np.logical_not(np.any(np.isnan(laplace_sd),1))
print('RMSE')
print(np.round(np.sqrt(np.mean(np.square(test[good_ix]-laplace_mean[good_ix]),0))*oxide_sd*100,2))
print('Cov')
print(np.round(np.nanmean(np.logical_and(laplace_mean[good_ix]-2*laplace_sd[good_ix] <= test[good_ix], laplace_mean[good_ix]+2*laplace_sd[good_ix] >= test[good_ix]), 0),2))
print('Width')
print(np.round(np.nanmean(2*laplace_sd[good_ix],0)*oxide_sd*100,2))

# %% Mars
mars_spec = np.load('data/mars_spec.npy')
mars_x = torch.from_numpy(mars_spec).float().to(device)
# Run through CNN ensemble to get mars predictions
ensemble_versions = ['version_%d' % d for d in np.arange(1,31)]
mars_ensemble_pred = []
for e in ensemble_versions:
    f = os.listdir('lightning_logs/%s/checkpoints/' % e)

    cnn = CNN(in_dim=mars_spec.shape[1], out_dim=len(oxides), ch_sizes=[32,128,1],
            krnl_sizes=[11,5,1], stride=[3,3,3], lin_l_sizes = [20, 20],
            activation='relu', device=device)

    orig_model = CCamCNN.load_from_checkpoint('lightning_logs/%s/checkpoints/%s' % (e, f[0]), cnn=cnn)

    cnn_mars_pred = orig_model.cnn(mars_x.to(device)).detach().cpu().numpy()
    mars_ensemble_pred.append(cnn_mars_pred)
mars_ensemble_pred = np.array(mars_ensemble_pred) # (30, 253, 9)
mars_ensemble_pred *= oxide_sd
mars_ensemble_mean = np.mean(mars_ensemble_pred, 0)
mars_ensemble_sd = np.std(mars_ensemble_pred, 0)

# each (253, 9)
mars_laplace_mean = np.load('results/laplace_mars_mean_predictions.npy')
mars_laplace_sd = np.load('results/laplace_mars_sd_predictions.npy')
mars_laplace_mean *= oxide_sd
mars_laplace_sd *= oxide_sd
mars_laplace_pred = []
for i in range(100):
    mars_laplace_pred.append(np.random.normal(loc=mars_laplace_mean, scale=mars_laplace_sd))
mars_laplace_pred = np.array(mars_laplace_pred) # (100, 253, 9)

# (100, 253, 9)
mars_vi_pred = np.load('results/vi_mars_predictions.npy')
mars_vi_pred *= oxide_sd
mars_vi_mean = np.mean(mars_vi_pred, 0)
mars_vi_sd = np.std(mars_vi_pred, 0)

for dm, d, n in zip([mars_vi_pred, mars_ensemble_pred, mars_laplace_pred], 
                    [vi_pred, ensemble_pred, laplace_pred],
                    ['VI', 'Ensemble', 'Laplace']):
    print(n)
    print(oxides)
    print('Width (Mars)')
    print(np.round(width(dm)*100,2))
    print('Width (Earth)')
    print(np.round(width(d)*oxide_sd*100,2))
    print('Relative Mars - Earth ')
    rel = 100*(width(dm)*100-width(d)*oxide_sd*100)/(width(d)*oxide_sd*100)
    print(np.round(rel.squeeze(),2))

# %%
np.random.seed(42)
for mars_ix in np.random.choice(253, size=10, replace=False):
    #mars_ix = 1 # which mars data point
    df_samples = []
    df_models = []
    df_oxides = []
    for i, ox in enumerate(oxides):
        # ensemble
        tmp = list(mars_ensemble_pred[:, mars_ix, i])
        df_samples += tmp
        df_models += ['Ensemble']*len(tmp)
        df_oxides += [ox]*len(tmp)
        # laplace
        tmp = list(mars_laplace_pred[:, mars_ix, i])
        df_samples += tmp
        df_models += ['Laplace']*len(tmp)
        df_oxides += [ox]*len(tmp)
        # VI
        tmp = list(mars_vi_pred[:, mars_ix, i])
        df_samples += tmp
        df_models += ['VI']*len(tmp)
        df_oxides += [ox]*len(tmp)
    df = pd.DataFrame({'Model':df_models, 'Oxide':df_oxides, 'Prediction':df_samples})

    plt.figure()
    sns.boxplot(x="Oxide", y='Prediction', hue='Model', data=df)
    plt.show()


# %% view CNN
model_graph = draw_graph(orig_model.cnn, input_size=(64, mars_spec.shape[1]), 
                         device='meta', #graph_dir='LR', 
                         save_graph=True)
model_graph.visual_graph    

# %% Plot some data
train_oxides = np.load('data/train_oxides.npy')
train_oxides_df = pd.DataFrame(data=train_oxides*oxide_sd*100,
                               index=np.arange(len(train_oxides)),
                               columns=oxides)

sns.pairplot(train_oxides_df, diag_kind='kde',
             x_vars=['SiO2','FeOT','MgO'], y_vars=['SiO2','FeOT','MgO'])
plt.savefig('figures/oxide_pairplot.png', dpi=300)
plt.show()
# %%
train_spec_nonorm = np.load('data/train_spec_nonorm.npy')
train_wav = np.load('data/train_wav.npy')

# %%
mean_spec = np.mean(train_spec_nonorm,0)
min_spec = np.min(train_spec_nonorm,0)
max_spec = np.max(train_spec_nonorm,0)
sd_spec = np.std(train_spec_nonorm, 0)
plt.figure()
plt.plot(train_wav, mean_spec)
#plt.fill_between(train_wav, min_spec, max_spec, color='red', alpha=0.7)
plt.fill_between(train_wav, mean_spec-sd_spec, mean_spec+sd_spec, color='red', alpha=0.7)
plt.show()

# %%

vnir_range = [492.427, 849.0]
vio_range = [382.13, 473.184]
uv_range = [246.635, 338.457]
vnir_mask = np.logical_and(train_wav >= vnir_range[0], train_wav <= vnir_range[1])
np.random.seed(42)
plt.figure(figsize=(10,3))
plt.plot(train_wav[vnir_mask], train_spec_nonorm[:, vnir_mask][np.random.choice(len(train_spec_nonorm), 30)].T)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.savefig('figures/train_spec_examples.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
test_oxides = np.load('data/test_oxides.npy')
for i in range(len(oxides)):

    plt.figure(figsize=(9,5))
    ax = plt.subplot(121)
    plt.axline([0,0], slope=1)
    plt.plot(test_oxides[:, i]*oxide_sd[0,i]*100, pls_pred[:, i]*oxide_sd[0,i]*100, 'k.')
    plt.xlabel("Reference")
    plt.ylabel('Predicted')
    plt.title('PLS (RMSE: %0.2f)' % (rmse(test_oxides,pls_pred)[i]*oxide_sd[0,i]*100))
    plt.subplot(122, sharex=ax, sharey=ax)
    plt.axline([0,0], slope=1)
    plt.plot(test_oxides[:, i]*oxide_sd[0,i]*100, cnn_pred[:, i]*oxide_sd[0,i]*100, 'k.')
    plt.xlabel("Reference")
    plt.title('CNN (RMSE: %0.2f)' % (rmse(test_oxides,cnn_pred)[i]*oxide_sd[0,i]*100))
    plt.suptitle(oxides[i])
    plt.tight_layout()
    plt.savefig('figures/cnn_pls_ref_v_predicted_%s.png'%oxides[i], dpi=300, bbox_inches='tight')
    plt.show()

    worst = np.argmax(np.abs(test_oxides-pls_pred)[:, i])
    print(np.round(test_oxides[worst, :]*oxide_sd[0,:]*100,2))

# %%
