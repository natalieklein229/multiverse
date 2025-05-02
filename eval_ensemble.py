"""
Evaluate CNN ensemble.

"""
# %% 
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

np.set_printoptions(precision=3, suppress=True)

sns.set_theme(context='talk')

from models import CCamCNN
from util import rmse

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def list_subdirectories(path):
    return [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

# %% get list of subdir
ensemble_dirs = list_subdirectories('lightning_logs')

# %%
oxide_sd = np.load('/data/0/chemcam_bnn/oxide_sd.npy')
test_spec = np.load('/data/0/chemcam_bnn/test_spec.npy')
test_oxides = np.load('/data/0/chemcam_bnn/test_oxides.npy')
mars_spec = np.load('/data/0/chemcam_bnn/mars_spec.npy')

# %% load/inspect models, make predictions with/without noise
ens_pred = []
ens_pred_mars = []
ens_pred_noisy = []
ens_pred_noisy_mars = []
ens_mean_rmse = []
for i, mpath in enumerate(ensemble_dirs):
    cpath = glob.glob('%s/checkpoints/*.ckpt' % mpath)
    model = CCamCNN.load_from_checkpoint(cpath[0])#,cnn=cnn,output_dim=len(oxides))
    noise_prec = torch.exp(-model.log_var).detach().cpu().numpy()
    noise_sd = np.squeeze(np.sqrt(1/noise_prec)*oxide_sd*100)
    yhat = model(torch.Tensor(test_spec).to(device)).cpu().detach().numpy()
    yhat *= oxide_sd*100
    ens_pred.append(yhat)
    yhat_noisy_tmp = []
    for j in range(100):
        yhat_noisy_tmp.append(yhat + np.random.normal(scale=noise_sd,size=(len(yhat),len(oxides))))
    ens_pred_noisy.append(np.array(yhat_noisy_tmp))
    rm = rmse(test_oxides*oxide_sd*100,yhat)
    print('seed %d' % (i+1))
    #print('RMSE')
    #print(rm)
    print('mean RMSE')
    print(np.mean(rm))
    ens_mean_rmse.append(np.mean(rm))
    #print('Noise SD')
    #print(noise_sd)
    # Mars
    yhat = model(torch.Tensor(mars_spec).to(device)).cpu().detach().numpy()
    yhat *= oxide_sd*100
    ens_pred_mars.append(yhat)
    yhat_noisy_tmp = []
    for j in range(100):
        yhat_noisy_tmp.append(yhat + np.random.normal(scale=noise_sd,size=(len(yhat),len(oxides))))
    ens_pred_noisy_mars.append(np.array(yhat_noisy_tmp))
ens_pred = np.array(ens_pred)
ens_pred_mars = np.array(ens_pred_mars)
ens_pred_noisy = np.array(ens_pred_noisy)
ens_pred_noisy_mars = np.array(ens_pred_noisy_mars)
ens_mean_rmse = np.array(ens_mean_rmse)

res= {'rmse':ens_mean_rmse, 'ens_pred':ens_pred, 'ens_pred_mars':ens_pred_mars,
      'ens_pred_noisy':ens_pred_noisy, 'ens_pred_noisy_mars':ens_pred_noisy_mars}
with open('results/ensemble_compiled.pkl', 'wb') as f:
    pickle.dump(res,f)

# %%
# ens_pred_sub = ens_pred[ens_mean_rmse<3.0]
# ens_pred_noisy_sub = ens_pred_noisy[ens_mean_rmse<3.0]
# ens_pred_noisy_sub = ens_pred_noisy_sub.reshape([-1,ens_pred_noisy.shape[2],ens_pred_noisy.shape[3]])
# ens_mean = np.mean(ens_pred_noisy_sub,0)
# ens_sd = np.std(ens_pred_noisy_sub,0)

# for j in range(len(oxides)):
#     plt.figure(figsize=(10,6))
#     plt.subplot(121)
#     plt.axline([0,0], slope=1)
#     plt.plot(test_oxides[:, j]*oxide_sd[0,j]*100, np.mean(ens_pred_sub,0)[:,j], 'k.')
#     plt.xlabel("Reference")
#     plt.ylabel('Predicted')
#     plt.subplot(122)
#     plt.axline([0,0], slope=1)
#     #plt.plot(test_oxides[:, j]*oxide_sd[0,j]*100, ens_mean[:,j], 'k.')
#     plt.errorbar(test_oxides[:, j]*oxide_sd[0,j]*100, ens_mean[:,j], yerr=ens_sd[:, j], fmt='k.', zorder=-1)
#     plt.xlabel("Reference")
#     plt.ylabel('Predicted')
#     plt.suptitle(oxides[j])
#     plt.tight_layout()
#     plt.show()

# TODO: below probably not too useful; need to evaluate across ensemble.
# with and without data noise, I guess.


# # %% mean predictions (no data uncertainty)
# ensemble_pred = []
# for i in np.arange(len(ensemble_dirs)):
#     tmp = np.load('results/cnn%d_predictions.npy' % (i+1))
#     ensemble_pred.append(tmp)
# ensemble_pred = np.array(ensemble_pred)

# # %% initial metrics, per model
# for i in range(ensemble_pred.shape[0]):
#     rm = rmse(test_oxides*oxide_sd*100,ensemble_pred[i, :, :]*oxide_sd*100)
#     print('seed %s' % (i+1))
#     print(rm)



# # %%
# test_oxides = np.load('/data/0/chemcam_bnn/test_oxides.npy')
# for i in range(len(oxides)):

#     plt.figure(figsize=(9,5))
#     ax = plt.subplot(121)
#     plt.axline([0,0], slope=1)
#     for j in range(ensemble_pred.shape[0]):
#         plt.plot(test_oxides[:, i]*oxide_sd[0,i]*100, ensemble_pred[j, :, i]*oxide_sd[0,i]*100, 'k.')
#     plt.plot(test_oxides[:, i]*oxide_sd[0,i]*100, np.mean(ensemble_pred,0)[:, i]*oxide_sd[0,i]*100, 'r.')
#     plt.xlabel("Reference")
#     plt.ylabel('Predicted')


# # %%
