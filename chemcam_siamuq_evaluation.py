"""
Evaluate results from CNN, ensemble, laplace, VI. 


"""
# %% 

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from util import rmse, coverage, width, interval_score

sns.set_theme(context='talk')

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% Load data and predictions, scale back as needed
oxide_sd = np.load('/data/0/chemcam_bnn/oxide_sd.npy')
test = np.load('/data/0/chemcam_bnn/test_oxides.npy')*oxide_sd*100
cnn_pred = np.load('results/cnn1_predictions.npy')*oxide_sd*100

with open('results/laplace_predictions.pkl','rb') as f:
    res = pickle.load(f)
    laplace_mean = res['mean']*oxide_sd*100
    laplace_sd = res['sd']*oxide_sd*100
    laplace_sd_noisy = res['sd_noisy']*oxide_sd*100
    laplace_mean_mars = res['mars_mean']*oxide_sd*100
    laplace_sd_mars = res['mars_sd']*oxide_sd*100
    laplace_sd_noisy_mars = res['mars_sd_noisy']*oxide_sd*100

with open('results/vi_predictions.pkl','rb') as f:
    res = pickle.load(f)
    vi_pred = res['pred']*oxide_sd*100
    vi_pred = np.transpose(vi_pred,[1,0,2])
    vi_pred_noisy = res['pred_noisy']*oxide_sd*100
    vi_pred_noisy = np.transpose(vi_pred_noisy,[1,0,2])
    vi_pred_mars = res['mars_pred']*oxide_sd*100
    vi_pred_mars = np.transpose(vi_pred_mars,[1,0,2])
    vi_pred_noisy_mars = res['mars_pred_noisy']*oxide_sd*100
    vi_pred_noisy_mars = np.transpose(vi_pred_noisy_mars,[1,0,2])

with open('results/ensemble_compiled.pkl', 'rb') as f:
    res = pickle.load(f)
    ensemble_pred = res['ens_pred'][res['rmse']<3.0]
    ens_pred_noisy = res['ens_pred_noisy'][res['rmse']<3.0]
    ensemble_pred_noisy = ens_pred_noisy.reshape([-1,ens_pred_noisy.shape[2],ens_pred_noisy.shape[3]])
    ensemble_pred_mars = res['ens_pred_mars'][res['rmse']<3.0]
    ens_pred_noisy_mars = res['ens_pred_noisy_mars'][res['rmse']<3.0]
    ensemble_pred_noisy_mars = ens_pred_noisy_mars.reshape([-1,ens_pred_noisy_mars.shape[2],ens_pred_noisy_mars.shape[3]])

laplace_pred = []
laplace_pred_noisy = []
for i in range(1000):
    laplace_pred.append(np.random.normal(loc=laplace_mean, scale=laplace_sd))
    laplace_pred_noisy.append(np.random.normal(loc=laplace_mean, scale=laplace_sd_noisy))
laplace_pred = np.array(laplace_pred) # (100, 253, 9)
laplace_pred_noisy = np.array(laplace_pred_noisy) # (100, 253, 9)

def print_tex(l):
    latex_row = ' & '.join([f"{item:.2f}" for item in l]) + r' \\'
    return latex_row

for d, n in zip([cnn_pred, vi_pred_noisy, ensemble_pred_noisy, laplace_pred_noisy], ['CNN', 'VI', 'Ensemble', 'Laplace']):
    print(n)
    print('RMSE')
    print(print_tex(rmse(test,d)))
    print(np.round(np.mean(rmse(test,d)),2))
    if n in ['VI', 'Ensemble', 'Laplace']:
        print('Cov')
        print(print_tex(coverage(test,d)))
        print(np.round(np.mean(coverage(test,d)),2))
        print('Width')
        print(print_tex(width(d)))
        print(np.round(np.mean(width(d)),2))
        print('Interval')
        print(print_tex(interval_score(test,d)))
        print(np.round(np.mean(interval_score(test,d)),2))
    print('\n')


# %% TODO: Mars stuff and from here down with new predictions, aleatoric/epistemic, etc.
# %% Mars evaluation
with open('results/ensemble_compiled.pkl', 'rb') as f:
    ens_res = pickle.load(f)
ens_pred = ens_res['ens_pred_noisy_mars'][ens_res['rmse']<3.0]
mars_ensemble_pred = ens_pred.reshape([-1,ens_pred.shape[2],ens_pred.shape[3]])
mars_ensemble_mean = np.mean(mars_ensemble_pred, 0)
mars_ensemble_sd = np.std(mars_ensemble_pred, 0)

# each (253, 9)
mars_laplace_mean = np.load('results/laplace_mars_mean_predictions.npy')*oxide_sd*100
mars_laplace_sd = np.load('results/laplace_mars_sd_predictions.npy')*oxide_sd*100
mars_laplace_pred = []
for i in range(100):
    mars_laplace_pred.append(np.random.normal(loc=mars_laplace_mean, scale=mars_laplace_sd))
mars_laplace_pred = np.array(mars_laplace_pred) # (100, 253, 9)

# (100, 253, 9)
mars_vi_pred = np.load('results/vi_mars_predictions.npy')*oxide_sd*100
mars_vi_mean = np.mean(mars_vi_pred, 0)
mars_vi_sd = np.std(mars_vi_pred, 0)

plt.figure(figsize=(5,5))
for dm, d, n in zip([mars_vi_pred, mars_ensemble_pred, mars_laplace_pred], 
                    [vi_pred, ensemble_pred, laplace_pred],
                    ['VI', 'Ensemble', 'Laplace']):
    print(n)
    print(oxides)
    print('Width (Mars)')
    print(np.round(width(dm),2))
    print('Width (Earth)')
    print(np.round(width(d),2))
    # print('Relative Mars - Earth ')
    # rel = (width(dm)-width(d))/(width(d))
    # print(np.round(rel.squeeze(),2))
    plt.plot(width(d), width(dm), 'o', label=n)
plt.axline([0,0],slope=1,linestyle='dashed')
plt.xlabel('Earth width')
plt.ylabel('Mars width')
plt.legend()
plt.show()



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
