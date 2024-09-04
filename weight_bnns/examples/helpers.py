import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats
from tqdm import tqdm
import functorch

from copy import deepcopy
import pyro

import seaborn as sns

def covariance_plots(bnn_model):
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].bar(torch.arange(len(bnn_model.GGN.diag())), (1/bnn_model.GGN.diag()).sqrt(), color='black')
    # set title
    axs[0].title.set_text('Approx. posterior std. of $\\theta$')

    for label in axs[0].yaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    symm_cov = bnn_model.GGN.inverse()
    std_dev = torch.sqrt(torch.diag(symm_cov))
    inv_std_dev = 1.0 / std_dev
    correlation_matrix = symm_cov * (inv_std_dev.unsqueeze(0) * inv_std_dev.unsqueeze(1))

    # plot covariance matrix with seaborn
    plot2 = sns.heatmap(correlation_matrix, cmap='seismic', vmin=-1, vmax=1)
    axs[1] = plot2
    
    axs[1].set_title('Approx. posterior correlation matrix of $\\theta$')

    # title('Approx. posterior correlation matrix of $\\theta$')
    # disable ticsk
    plt.xticks([])
    plt.yticks([])

    plt.show()


def plot_nll(nll_hist, test_nll_hist=None, title = None, save = False, save_path = None):
    ''' Plots the negative log-likelihood history of the model during training.
    '''
    plt.figure(figsize=(10, 5))
    nll_hist = nll_hist.cpu().detach()
    plt.plot(nll_hist)
    if test_nll_hist is not None:
        test_nll_hist = test_nll_hist.cpu().detach()
        plt.plot(len(nll_hist) * np.arange(len(test_nll_hist)) / len(test_nll_hist), test_nll_hist, color='orange')
        plt.legend(['Train', 'Test'])
    else:
        plt.legend(['Train'])
    plt.xlabel('Epochs')
    plt.ylabel('Negative log-likelihood')
    if title is not None:
        plt.title(title)
    if save:
        plt.savefig(save_path)
    plt.show()


def get_test_idx(targets_idx, test_prop=0.2, seed=None):
    ''' Returns a boolean array of the same length as targets_idx, where True indicates that the corresponding
    element is in the test set. The test set is chosen randomly, with the proportion of unique elements in the test set
    equal to test_prop.
    '''
    if seed is not None:
        np.random.seed(seed)

    idx = {x:i for i,x in enumerate(targets_idx)}
    numerical_idx = [idx[i] for i in targets_idx]

    num_unique = len(np.unique(numerical_idx))
    test_prop = test_prop
    targets_test_set = np.random.choice(list(idx.values()),int(num_unique*test_prop), replace=False)

    test_idx = np.array([idx in targets_test_set for idx in numerical_idx])
    return test_idx

def plot_1d_gaussian_preds(y_predictions,test_x, train_x=None, train_y=None, precision=None, method_name=None):
    ''' Plots the predictive distribution of normally distributed data.
    '''
    y_predictions = y_predictions.squeeze().cpu().detach()
    predictive_mean = y_predictions.mean(axis=0)
    predictive_std = y_predictions.std(axis=0)

    
    plt.figure(figsize=(6,4))

    # disable ticks
    plt.xticks([])
    plt.yticks([])


    plt.plot(test_x, predictive_mean.detach(), color='darkblue', alpha=0.5)
    
    if precision is not None:
        
        aleatoric_var = precision**(-1)
        combined_var = predictive_std**2 + aleatoric_var
        
        combined_std = torch.sqrt(combined_var)
        plt.fill_between(test_x, predictive_mean-2*(predictive_std), 
                    predictive_mean-2*combined_std, 
                    color='limegreen', alpha=0.5, linewidth=0.0, label='Aleatoric')
        plt.fill_between(test_x, predictive_mean+2*(predictive_std), 
                        predictive_mean+2*combined_std, 
                        color='limegreen', alpha=0.5, linewidth=0.0)
        
        plt.fill_between(test_x, predictive_mean-2*predictive_std, 
                        predictive_mean+2*predictive_std, color='steelblue', 
                        alpha=0.8, linewidth=0.0, label='Epistemic')
    else:
        plt.fill_between(test_x, predictive_mean-2*predictive_std, 
                        predictive_mean+2*predictive_std, color='steelblue', 
                        alpha=0.8, linewidth=0.0, label='Epistemic + Aleatoric')
    plt.scatter(train_x,train_y, color='black', alpha=0.5, s=2)

    plt.ylim(-1.3,1.5)
    plt.xlim(-1.5,2.3)
    if method_name is not None:
        plt.title(method_name)
    plt.legend()
    plt.show()

def plot_predictions(y_predictions, true_y, title=None, fontsize=14):
    ''' Plot each component of the predictive distribution of y.
    '''
    y_predictions = y_predictions.cpu().detach()
    predictive_mean_ll = y_predictions.mean(axis=0).squeeze()
    predictive_CI_high_ll = y_predictions.quantile(0.975, dim=0).squeeze()
    predictive_CI_low_ll = y_predictions.quantile(0.025, dim=0).squeeze()

    # plot the results
    fig, axs = plt.subplots(1,2, figsize=(10,5),sharey=True)
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)
    axs[0].plot(true_y, label='True')
    axs[1].plot(predictive_mean_ll, label='Predictive mean')

    K = y_predictions.shape[-1]
    for i in range(K):
        axs[1].fill_between(np.arange(len(predictive_mean_ll)), predictive_CI_low_ll[:,i], predictive_CI_high_ll[:,i], alpha=0.3, label='95% CI')

def components_plot(prediction, data, prediction_std=None, title=None, fontsize=14):
    ''' Compare predicted and true values of each component of the data.
    '''
    fig, axs = plt.subplots(3,3, figsize=(7,7))
    axs = axs.flatten()
    prediction = prediction.cpu().detach()
    components_names =  ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
    # set only two ticks on x and y axes
    # for ax in axs:
    #     ax.locator_params(axis='x', nbins=2)
    #     ax.locator_params(axis='y', nbins=2)
    # disable ticks and numbers on axes
    for ax in axs:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

    for j in range(9):
        axs[j].scatter(prediction[:,j], data[:,j], color='black', s=2)
        if j in [0,3,6]:
            axs[j].set_ylabel('True')
        if j in [6,7,8]:
            axs[j].set_xlabel('Predicted')
        
        axs[j].set_title(components_names[j])
        # add 1:1 line
        lims = [
            np.min([axs[j].get_xlim(), axs[j].get_ylim()]),  # min of both axes
            np.max([axs[j].get_xlim(), axs[j].get_ylim()]),  # max of both axes
        ]
        axs[j].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        
        # add regression line
        m, b = np.polyfit(prediction[:,j], data[:,j], 1)
        axs[j].plot(prediction[:,j], m*prediction[:,j] + b, color='red')

        axs[j].set_xlim(lims)
        axs[j].set_ylim(lims)

        if prediction_std is not None:
            prediction_std = prediction_std.cpu().detach()
            # add approx. 95% confidence interval
            prediction_sorted,_ = torch.sort(prediction[:,j], dim=0)
            data_sorted,_ = torch.sort(data[:,j], dim=0)
            axs[j].fill_between(prediction_sorted,data_sorted- 2*prediction_std[:,j], data_sorted + 2*prediction_std[:,j], alpha=0.3)
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    fig.show()


def coverage(predictive, x_test, y_test, x_train, y_train, M=100):
    '''
    Proportion of test set observations covered by the 95% CI built on the test set and
    training set observations covered by the 95% CI built on the training set, averaged
    across M MC samples from the predictive distribution of y.

    Parameters
    ----------
    predictive : function
        Function that takes in an input and samples M times from the predictive of y
    x_test : torch.Tensor
        Test set covariates.
    y_test : torch.Tensor
        Test set responses.
    x_train : torch.Tensor
        Training set covariates.
    y_train : torch.Tensor
        Training set responses.
    M : int
        Number of MC samples from the predictive distribution of y.
    '''
    N_train = len(x_train)
    N_test = len(x_test)

    y_pred_train = predictive(x_train, num_samples=M).cpu().detach()
    y_pred_test = predictive(x_test, num_samples=M).cpu().detach()

    y_pred_train_CI_high = y_pred_train.quantile(0.975, dim=0)
    y_pred_train_CI_low = y_pred_train.quantile(0.025, dim=0)

    y_pred_test_CI_high = y_pred_test.quantile(0.975, dim=0)
    y_pred_test_CI_low = y_pred_test.quantile(0.025, dim=0)

    cover_test = (y_pred_test_CI_low < y_test)&(y_test < y_pred_test_CI_high)
    cover_train = (y_pred_train_CI_low < y_train)&(y_train < y_pred_train_CI_high)

    cp_test = torch.sum(cover_test, dim=0)/N_test
    cp_train = torch.sum(cover_train, dim=0)/N_train
    return cp_train, cp_test


def loss_on_2d_subspace(data_loader, model, weights, resolution = 50, u_lim = 30, l_lim = -5, MC_chains = None):
    ''' Plots the log joint distribution of the model on a 2D subspace of the parameter space.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        Data loader for the data set.
    model : inference.bnn._BNN
        Bayesian neural network object.
    weights : tuple
        Tuple of three weights from the model, from which the 2D subspace is constructed.
    resolution : int
        Number of points in each dimension of the subspace.
    u_lim : float
        Upper limit of the subspace.
    l_lim : float   
        Lower limit of the subspace.
    MC_chains : list    
        List of Markov chains from MCMC sampling of the posterior distribution of the model.
    '''
    def log_joint(w):
        torch.nn.utils.vector_to_parameters(w.to(device), model.torch_net.parameters())
        f = model.torch_net(x).cpu().detach()
        ll = model.likelihood.log_likelihood(f, y, reduction='sum').cpu().detach()
        log_prior = model.prior._distribution.log_prob(w).sum().cpu().detach()
        return ll + log_prior

    def orth_basis_2d(w1,w2,w3):
        u = w2 - w1
        v = (w3 - w1) - (u * (u.dot(w3 - w1) / u.dot(u)))
        u_norm = u / torch.norm(u)
        v_norm = v / torch.norm(v)
        return u_norm, v_norm

    def create_weights(w1,u,v,positions):
        return w1 + positions @ torch.stack([u,v], dim=1).T

    def get_xy(point, origin, vector_x, vector_y):
        return torch.stack([torch.dot(point - origin, vector_x), torch.dot(point - origin, vector_y)])

    device = model.torch_net.device
    x, y = data_loader.dataset.tensors
    x = x.to(device)
    y = y.to(device)
    w1,w2,w3 = weights
    u_norm, v_norm = orth_basis_2d(w1,w2,w3)

    e = torch.linspace(l_lim, u_lim, resolution)
    X, Y = torch.meshgrid(e, e)
    positions = torch.stack([X, Y], dim=2)   
    W = create_weights(w1,u_norm,v_norm,positions)
    
    energy = torch.zeros(resolution,resolution)
    for i in tqdm(range(resolution)):
        for j in range(resolution):
            energy[i,j] = log_joint(w=W[i,j,:])
    # set plot size
    plt.figure(figsize=(6,4))

    cs = plt.contourf(X,Y,energy.detach(),resolution*2,cmap="viridis")
    plt.contour(cs, colors='black', linewidths=.3)
    if MC_chains is not None:
        batched_get_xy = functorch.vmap(get_xy, in_dims=(0,None,None,None))

        for i, chain in enumerate(MC_chains):
            endpoint = chain[-1]
            xy = get_xy(endpoint, w1, u_norm, v_norm)
            chain_subspace = batched_get_xy(chain,w1,u_norm,v_norm)
            plt.plot(chain_subspace[:,0],chain_subspace[:,1],color="black",linewidth=1, alpha=.5, label=f"MC {i}")
            plt.scatter(xy[0],xy[1], marker="x",color="red")
    plt.xticks([])
    plt.yticks([])

    plt.legend()

    cs.changed()

        
def samples_to_tensor(samples_dict_original):
    ''' Converts a dictionary of MC samples from the posterior distribution of a model to a tensor.
    '''

    keys = list(samples_dict_original.keys())
    samples_dict = {}
    for i in range(0, len(keys) - 1, 2):
        samples_dict[keys[i + 1]] = samples_dict_original[keys[i + 1]]
        samples_dict[keys[i]] = samples_dict_original[keys[i]]
            # get shapes of all values in the dictionary
    shape_dict = {}
    for key in samples_dict.keys():
        shape_dict[key] = samples_dict[key].squeeze().shape
    # concatenate all values in the dictionary along the second dimension
    samples = torch.cat([samples_dict[key] if len(shape_dict[key])<2 
                          else samples_dict[key].squeeze() if len(shape_dict[key]) == 2 
                          else samples_dict[key].squeeze().flatten(1) 
                          for key in samples_dict.keys()],
                          dim=1)
    return samples