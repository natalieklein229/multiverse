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

from util import rmse, coverage, width, interval_score, get_groups

sns.set_theme(context='talk')

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Load data and predictions, scale back as needed
train_spec = np.load('/data/0/chemcam_bnn/train_spec.npy')
train_oxides = np.load('/data/0/chemcam_bnn/train_oxides.npy')
train_wav = np.load('/data/0/chemcam_bnn/train_wav.npy')
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
laplace_pred_mars = []
laplace_pred_noisy_mars = []
for i in range(1000):
    laplace_pred_mars.append(np.random.normal(loc=laplace_mean_mars, scale=laplace_sd_mars))
    laplace_pred_noisy_mars.append(np.random.normal(loc=laplace_mean_mars, scale=laplace_sd_noisy_mars))
laplace_pred_mars = np.array(laplace_pred_mars) # (100, 253, 9)
laplace_pred_noisy_mars = np.array(laplace_pred_noisy_mars) # (100, 253, 9)

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

# %% TODO Plot aleatoric/epistemic predictions for some targets for Si02, K20
from matplotlib.patches import Patch
def aggregate_predictions(means, variances):
    n = len(means)
    mean_avg = np.mean(means,0)
    var_avg = np.sum(variances,0) / n #**2
    return mean_avg, np.sqrt(var_avg)

np.random.seed(42)
test_targ = get_groups(test)
targ_sel = np.random.choice(len(test_targ.keys()),20)
ox_ind = 0

plt.figure(figsize=(18,12))
counter = 1
for pi, ox_ind in enumerate([0,8]):
    for tmp_, tmp_noisy_, nm in zip([vi_pred, ensemble_pred, laplace_pred],[vi_pred_noisy, ensemble_pred_noisy, laplace_pred_noisy],['VB','Ensemble','Laplace']):
        plt.subplot(2,3,counter)
        min_x = 100
        for i, ti in enumerate(targ_sel):
            ix = test_targ[targ_sel[i]]
            tmp = tmp_[:, ix, :]
            tmp_noisy = tmp_noisy_[:, ix, :]
            tmp_mean, tmp_sd = aggregate_predictions(np.mean(tmp,0), np.var(tmp,0))
            tmp_noisy_mean, tmp_noisy_sd = aggregate_predictions(np.mean(tmp_noisy,0), np.var(tmp_noisy,0))
            plt.errorbar(test[ix][0,ox_ind],tmp_mean[ox_ind],yerr=2*tmp_noisy_sd[ox_ind],  ecolor='blue', fmt='none')
            plt.errorbar(test[ix][0,ox_ind],tmp_mean[ox_ind],yerr=2*tmp_sd[ox_ind], elinewidth=5.0, ecolor='red', fmt='none')
            plt.plot(test[ix][0,ox_ind],tmp_mean[ox_ind],'k.')
            if test[ix][0,ox_ind] < min_x:
                min_x = test[ix][0,ox_ind]
        plt.axline([0.95*min_x,0.95*min_x], slope=1, linestyle='dashed', color='grey')
        if counter > 3:
            plt.xlabel('Reference value (ox. wt. %)')
        if counter == 1 or counter == 4:
            plt.ylabel('Predicted value (ox. wt. %)')
        plt.gca().text(0.03, 0.97, oxides[ox_ind], transform=plt.gca().transAxes,
        horizontalalignment='left', verticalalignment='top')
        custom_lines = [Patch(facecolor='red', edgecolor='none', linewidth=2),
                        Patch(facecolor='blue', edgecolor='none', linewidth=1)]
        custom_labels = ['Epistemic', 'Total']
        plt.legend(custom_lines, custom_labels, loc='lower right')
        plt.title(nm)
        counter += 1
plt.tight_layout()
plt.savefig('figures/ref_v_pred_uq.png',bbox_inches='tight',dpi=300)

# %% Prediction error v uncertainty (across all oxides)
np.random.seed(42)
plt.figure(figsize=(10,5))
counter = 1
test_targ = get_groups(test)
for tmp_, tmp_noisy_, nm in zip([vi_pred, ensemble_pred],
                                [vi_pred_noisy, ensemble_pred_noisy],
                                ['VB','Ensemble',]):
    errs = []
    epi = []
    total = []
    for i, ti in enumerate(test_targ.keys()):
        ix = test_targ[i]
        tmp = tmp_[:, ix, :]
        tmp_noisy = tmp_noisy_[:, ix, :]
        tmp_mean, tmp_sd = aggregate_predictions(np.mean(tmp,0), np.var(tmp,0))
        tmp_noisy_mean, tmp_noisy_sd = aggregate_predictions(np.mean(tmp_noisy,0), np.var(tmp_noisy,0))
        label = test[ix][0]
        err = np.sqrt(np.mean(np.square(label-np.mean(tmp_noisy,0)),0))
        errs.append(err)
        epi.append(tmp_sd)
        total.append(tmp_noisy_sd)
    errs = np.concatenate(errs)
    epi = np.concatenate(epi)
    total = np.concatenate(total)
    if counter == 1:
        ax = plt.subplot(1,2,counter)
    else:
        plt.subplot(1,2,counter,sharey=ax,sharex=ax)
    plt.plot(errs,epi,'r.',label='Epistemic')
    #plt.plot(errs,total,'b.',label='Total')
    plt.xlabel('Prediction RMSE')
    plt.ylabel('Epistemic uncertainty')
    plt.title(nm)
    counter += 1
#plt.legend()
plt.tight_layout()
plt.savefig('figures/err_v_uq.png', bbox_inches='tight', dpi=300)

# %% Calibration plot
n_cal = 10
cal_vals = np.linspace(0,1,n_cal)
cov_vals = {'Laplace':np.zeros(n_cal), 'VB':np.zeros(n_cal), 'Ensemble':np.zeros(n_cal)}
for i, c in enumerate(cal_vals):
    cov_vals['Laplace'][i] = coverage(test,laplace_pred_noisy,alpha=c).mean()
    cov_vals['VB'][i] = coverage(test,vi_pred_noisy,alpha=c).mean()
    cov_vals['Ensemble'][i] = coverage(test,ensemble_pred_noisy,alpha=c).mean()

cov_vals['ref'] = cal_vals
cov_vals = pd.DataFrame(cov_vals).melt(id_vars='ref', var_name='Method', value_name='coverage')
cov_vals = cov_vals.rename(columns={'ref':'Desired Coverage','coverage':'Observed Coverage'})
plt.figure(figsize=(5,5))
sns.lineplot(cov_vals,x='Desired Coverage',y='Observed Coverage',hue='Method')
plt.axline([0,0],slope=1,linestyle='dashed',color='grey')
plt.savefig('figures/calibration_plot.png', bbox_inches='tight', dpi=300)

# %% Epistemic fraction
ens_epi = np.var(ensemble_pred,0)/np.var(ensemble_pred_noisy,0)
vi_epi = np.var(vi_pred,0)/np.var(vi_pred_noisy,0)
laplace_epi = np.var(laplace_pred,0)/np.var(laplace_pred_noisy,0)
df = {'Ensemble':ens_epi.flatten(), 'VB':vi_epi.flatten(), 'Laplace':laplace_epi.flatten(),
      'Oxide':np.tile(oxides,(len(ens_epi),1)).flatten()}
df = pd.DataFrame(df).melt(id_vars='Oxide', var_name='Method', value_name='Epistemic Fraction')
plt.figure(figsize=(12,4))
sns.boxplot(df,x='Oxide',y='Epistemic Fraction',hue='Method', fliersize=1.5)
sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('figures/epistemic_frac.png', bbox_inches='tight', dpi=300)

# %% Mars evaluation of epistemic uncertainty
ens_epi_mars = np.var(ensemble_pred_mars,0)/np.var(ensemble_pred_noisy_mars,0)
vi_epi_mars = np.var(vi_pred_mars,0)/np.var(vi_pred_noisy_mars,0)
laplace_epi_mars = np.var(laplace_pred_mars,0)/np.var(laplace_pred_noisy_mars,0)
ens_df = pd.DataFrame({'Oxide':oxides+oxides, 
                       'Epistemic Fraction':np.concatenate([np.mean(ens_epi,0),np.mean(ens_epi_mars,0)]),
                       'Dataset':['Earth']*len(oxides) + ['Mars']*len(oxides),
                       'Method':'Ensemble'})
vi_df = pd.DataFrame({'Oxide':oxides+oxides, 
                       'Epistemic Fraction':np.concatenate([np.mean(vi_epi,0),np.mean(vi_epi_mars,0)]),
                       'Dataset':['Earth']*len(oxides) + ['Mars']*len(oxides),
                       'Method':'VB'})
laplace_df = pd.DataFrame({'Oxide':oxides+oxides, 
                       'Epistemic Fraction':np.concatenate([np.mean(laplace_epi,0),np.mean(laplace_epi_mars,0)]),
                       'Dataset':['Earth']*len(oxides) + ['Mars']*len(oxides),
                       'Method':'Laplace'})
df = pd.concat([ens_df, vi_df],axis=0)

color_dict = {'Mars': 'red', 'Earth': 'blue'}
g = sns.FacetGrid(df, col='Method', height=4, aspect=1.5)
g.map_dataframe(sns.barplot,x='Oxide',y='Epistemic Fraction',hue='Dataset',palette=color_dict)
g.add_legend()
plt.savefig('figures/epistemic_frac_mars.png', bbox_inches='tight', dpi=300)

# %% Spectral plot
train_targ = get_groups(train_oxides)
targ_sel = np.random.choice(len(train_targ.keys()),5)
plt.figure(figsize=(12,12))
for i, ti in enumerate(targ_sel):
    plt.subplot(5,1,i+1)
    spec_tmp = train_spec[train_targ[ti],:]
    print(spec_tmp.shape)
    plt.plot(train_wav, np.mean(spec_tmp,0),'k', linewidth=0.7)
    plt.fill_between(train_wav, np.mean(spec_tmp,0) - 2*np.std(spec_tmp,0),
                     np.mean(spec_tmp,0) + 2*np.std(spec_tmp,0), alpha=0.7)
    plt.xlim([390,650])
    plt.yticks([])
    plt.ylabel('Intensity (a.u.)')
plt.xlabel('Wavelength (nm)')
plt.tight_layout()
plt.savefig('figures/spectra.png', bbox_inches='tight', dpi=300)

# %% Composition plot
train_ox_unique = np.unique(train_oxides,axis=0)*oxide_sd*100
ox_df = pd.DataFrame(train_ox_unique, columns=oxides)
sns.pairplot(ox_df, kind='kde')
plt.savefig('figures/oxides.png', bbox_inches='tight', dpi=300)





