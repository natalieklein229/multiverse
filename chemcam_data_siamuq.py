"""
Get ChemCam single-shot data for SIAM UQ. Split train and test and fit PLS baseline model. 

"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
import os

np.random.seed(42)

vnir_range = [492.427, 849.0]
vio_range = [382.13, 473.184]
uv_range = [246.635, 338.457]
keep_shots = ['shot%d' % i for i in range(5, 50)]
oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
data_paths = ['/home/neklein/chemcam_data/calib_2015', '/home/neklein/chemcam_data/calib_2021']

# %% Params
prop_val = 0.1

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

def load_data(data_path, targets, comp, lookup):
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
                comp_col, comp_val = lookup[targets[i]]
                try:
                    v = comp.loc[comp[comp_col]==comp_val,o].values.astype(float)[0]
                except:
                    v = np.nan
                #if targets[i] in comp['target'].values:
                #    v = comp.loc[comp['target']==targets[i],o].values.astype(float)[0]
                #else:
                #    v = np.nan
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

# %% data loading
comp = pd.read_csv('/home/neklein/chemcam_data/ccam_calibration_compositions.csv',nrows=572)
comp['Spectrum Name'] = comp['Spectrum Name'].astype(str)
#comp['target'] = comp['Spectrum Name'].apply(lambda x: x.lower().replace('-','').replace('_','').replace(' ',''))
comp['target'] = comp['Target'].astype(str).apply(lambda x: x.lower().replace('-','').replace('_','').replace(' ', ''))
comp['spectrum'] = comp['Spectrum Name'].apply(lambda x: x.lower().replace('-','').replace('_','').replace(' ',''))

all_train_data = []
all_val_data = []
all_test_data = []
nonmatch_dirs = []
for data_path in data_paths:
    print(data_path)
    targ_dirs_lookup = {}
    targ_dirs = os.listdir(data_path)
    print('found %d directories' % len(targ_dirs))
    for t in targ_dirs:
        t_rep = t.lower().replace('-','').replace('_','').replace(' ','')
        if t in comp['target'].values:
            targ_dirs_lookup[t] = ['target', t]
        elif t in comp['spectrum'].values:
            targ_dirs_lookup[t] = ['spectrum', t]
        elif t_rep in comp['target'].values:
            targ_dirs_lookup[t] = ['target', t_rep]
        elif t_rep in comp['spectrum'].values:
            targ_dirs_lookup[t] = ['spectrum', t_rep]

    print('matched %d directories' % len(targ_dirs_lookup.keys()))

    N = len(targ_dirs_lookup.keys())
    n_val = int(prop_val*N)
    n_test = n_val
    n_train = N - (n_val + n_test)

    targ_dirs = np.array(list(targ_dirs_lookup.keys()))
    targ_ind = np.random.choice(N, n_train+n_val+n_test, replace=False)
    train_targ = targ_dirs[targ_ind[:n_train]]
    val_targ = targ_dirs[targ_ind[n_train:(n_train+n_val)]]
    test_targ = targ_dirs[targ_ind[(n_train+n_val):]]

    train_data = load_data(data_path, train_targ, comp, targ_dirs_lookup)
    val_data = load_data(data_path, val_targ, comp, targ_dirs_lookup)
    test_data = load_data(data_path, test_targ, comp, targ_dirs_lookup)

    all_train_data.append(train_data)
    all_val_data.append(val_data)
    all_test_data.append(test_data)

train_data = pd.concat(all_train_data, axis=0)
val_data = pd.concat(all_val_data, axis=0)
test_data = pd.concat(all_test_data, axis=0)

# %% save pandas -- SLOW
# train_data.to_csv('data/train_data.csv', index=False)
# val_data.to_csv('data/val_data.csv', index=False)
# test_data.to_csv('data/test_data.csv', index=False)

# %%
# a little other normalization
wav_cols = [c for c in train_data.columns if 'wav' in c]

train_spec = train_data[wav_cols].values
val_spec = val_data[wav_cols].values
test_spec = test_data[wav_cols].values
spec_max = np.max(train_spec)
train_spec /= spec_max
val_spec /= spec_max
test_spec /= spec_max
train_spec[train_spec<-0.499] = -0.499
val_spec[val_spec<-0.499] = -0.499
test_spec[test_spec<-0.499] = -0.499
train_spec = np.log(train_spec + 0.5)
val_spec = np.log(val_spec + 0.5)
test_spec = np.log(test_spec + 0.5)
assert np.all(np.isfinite(train_spec))
assert np.all(np.isfinite(val_spec))
assert np.all(np.isfinite(test_spec))

train_oxides = train_data[oxides].values / 100.0
oxide_sd = np.std(train_oxides, 0, keepdims=True)
train_oxides /= oxide_sd
val_oxides = (val_data[oxides].values / 100.0) / oxide_sd
test_oxides = (test_data[oxides].values / 100.0) / oxide_sd
assert np.all(np.isfinite(train_oxides))
assert np.all(np.isfinite(val_oxides))
assert np.all(np.isfinite(test_oxides))

# %% Save data as numpy (for NN)
np.save('data/train_spec.npy', train_spec)
np.save('data/val_spec.npy', val_spec)
np.save('data/test_spec.npy', test_spec)
np.save('data/train_oxides.npy', train_oxides)
np.save('data/val_oxides.npy', val_oxides)
np.save('data/test_oxides.npy', test_oxides)

# %% PLS baseline
params = {'n_components': np.arange(5, 25, 2)}
pls = PLSRegression()
gcv = GridSearchCV(pls, params, verbose=2)
gcv.fit(train_spec, train_oxides)

# %%
cv_mean = gcv.cv_results_['mean_test_score']
cv_se = gcv.cv_results_['std_test_score']/np.sqrt(5)
args_ix = np.argmin(cv_mean > (cv_mean[np.argmin(cv_mean)]+cv_se[np.argmin(cv_mean)]))
n_comp = params['n_components'][args_ix]

plt.figure()
plt.plot(params['n_components'], cv_mean)
plt.fill_between(params['n_components'], cv_mean-cv_se, cv_mean+cv_se, alpha=0.7)
plt.axvline(n_comp)
plt.show()

# %%
pls = PLSRegression(n_components=n_comp).fit(train_spec, train_oxides)
pls_y_hat = pls.predict(test_spec)

for i in range(len(oxides)):
    plt.figure()
    plt.plot(test_oxides[:, i], pls_y_hat[:, i], 'k.')
    plt.axline([0,0], slope=1)
    plt.title(oxides[i])
    plt.show()

    pls_test_mse = np.mean(np.square(pls_y_hat[:, i]-test_oxides[:, i]))
    print('%s PLS test rmse' % (oxides[i], pls_test_mse))

# %% Save PLS and/or predictions
np.save('results/PLS_predictions.npy', pls_y_hat)

# %% look at residual variance?
resid = pls_y_hat - test_oxides
np.std(resid, 0)

# %%
