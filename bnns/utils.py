
import math
import pytz
from datetime import datetime
import numpy as np
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain     # ~~~ used (optionally) to define the prior distribution on network weights

from quality_of_life.my_base_utils import process_for_saving, dict_to_json
try:
    from quality_of_life.my_base_utils import buffer
except:
    from quality_of_life.my_visualization_utils import buffer   # ~~~ deprecated
    print("Please update quality_of_life")

#
# ~~~ Generate a .json filename based on the current datetime
def generate_json_filename(verbose=True):
    time = datetime.now(pytz.timezone('US/Mountain'))               # ~~~ current date and time MST
    file_name = str(time)
    file_name = file_name[:file_name.find(".")].replace(" ","_")    # ~~~ remove the number of seconds (indicated with ".") and replace blank space (between date and time) with an underscore
    file_name = process_for_saving(file_name+".json")               # ~~~ procsess_for_saving("path_that_exists.json") returns "path_that_exists (1).json"
    if verbose:
        if time.hour > 12:
            hour = time.hour - 12
            suffix = "pm"
        else:
            hour = time.hour
            suffix = "am"
        print("")
        print(f"    Generating file name {file_name} at {hour}:{time.minute}{suffix}")
        print("")
    return file_name

#
# ~~~ Compute the log pdf of a multivariate normal distribution with independent coordinates
def log_gaussian_pdf( where, mu, sigma ):
    assert mu.shape==where.shape
    marginal_log_probs = -((where-mu)/sigma)**2/2 - torch.log( math.sqrt(2*torch.pi)*sigma )   # ~~~ note: isn't (x-mu)/sigma numerically unstable, like numerical differentiation?
    return marginal_log_probs.sum()

#
# ~~~ Use Cholesky decompositions to compute the KL divergence N(mu_theta,Sigma_theta) || N(mu_0,Sigma_0) as described here https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
def gaussian_kl( mu_theta, root_of_Sigma_theta, mu_0, root_of_Sigma_0_inv ):
    mu_theta = mu_theta.flatten()
    mu_0 = mu_0.flatten()
    assert len(mu_theta)==len(mu_0)
    k = len(mu_0)
    assert root_of_Sigma_theta.shape==(k,k)==root_of_Sigma_0_inv.shape
    return ((root_of_Sigma_theta@root_of_Sigma_0_inv).norm()**2 - k + (root_of_Sigma_0_inv@(mu_0-mu_theta)).norm()**2)/2 - root_of_Sigma_0_inv.diag().log().sum() - root_of_Sigma_theta.diag().log().sum()


#
# ~~~ Define what we want the prior std. to be for each group of model parameters
def get_std(p):
    if len(p.shape)==1: # ~~~ for the biase vectors, take variance=1/length
        numb_pars = len(p)
        std = 1/math.sqrt(numb_pars)
    else:   # ~~~ for the weight matrices, mimic pytorch's `xavier normal` initialization (https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_normal_)
        fan_in, fan_out = _calculate_fan_in_and_fan_out(p)
        gain = calculate_gain("relu")
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return torch.tensor( std, device=p.device, dtype=p.dtype )

#
# ~~~ My version of the missing feature: a `dataset.to` method
def set_Dataset_attributes( dataset, device, dtype ):
    try:
        #
        # ~~~ Directly access and modify the underlying tensors
        dataset.X = dataset.X.to( device=device, dtype=dtype )
        dataset.y = dataset.y.to( device=device, dtype=dtype )
        return dataset
    except AttributeError:
        #
        # ~~~ Redefine the __getattr__ method (this is hacky; I don't know a better way; also, chat-gpt proposed this)
        class ModifiedDataset(torch.utils.data.Dataset):
            def __init__(self,original_dataset):
                self.original_dataset = original_dataset
                self.device = device
                self.dtype = dtype
            def __getitem__(self,index):
                x, y = self.original_dataset[index]
                return x.to( device=self.device, dtype=self.dtype ), y.to( device=self.device, dtype=self.dtype )
            def __len__(self):
                return len(self.original_dataset)
        return ModifiedDataset(dataset)


### ~~~
## ~~~ Plotting routines
### ~~~

#
# ~~~ Somewhat general helper routine for making plots
def univar_figure( fig, ax, grid, green_curve, x_train, y_train, model, title=None, blue_curve=None, **kwargs ):
    with torch.no_grad():
        #
        # ~~~ Green curve and green scatterplot of the data
        _, = ax.plot( grid.cpu(), green_curve.cpu(), color="green", label="Ground Truth", linestyle='--', linewidth=.5 )
        _ = ax.scatter( x_train.cpu(), y_train.cpu(),   color="green" )
        #
        # ~~~ Blue curve(s) of the model
        try:
            ax = blue_curve( model, grid, ax, **kwargs )
        except:
            ax = blue_curve( model, grid, ax ) 
        #
        # ~~~ Finish up
        _ = ax.set_ylim(buffer( y_train.cpu().tolist(), multiplier=0.35 ))
        _ = ax.legend()
        _ = ax.grid()
        _ = ax.set_title( description_of_the_experiment if title is None else title )
        _ = fig.tight_layout()
    return fig, ax

#
# ~~~ Basically just plot a plain old function
def trivial_sampler(f,grid,ax):
    _, = ax.plot( grid.cpu(), f(grid).cpu(), label="Neural Network", linestyle="-", linewidth=.5, color="blue" )
    return ax

#
# ~~~ Just plot a the model as an ordinary function
def plot_nn(
            fig,
            ax,
            grid,
            green_curve,    # ~~~ tensor with the same shape as `grid`
            x_train,
            y_train,        # ~~~ tensor with the same shape as `x_train`
            NN,             # ~~~ anything with a `__call__` method
            **kwargs
        ):
    return univar_figure(
            fig,
            ax,
            grid,
            green_curve,
            x_train,
            y_train,
            model = NN,
            title = "Conventional, Deterministic Training",
            blue_curve = trivial_sampler,
            **kwargs
        )

#
# ~~~ Graph the two standard deviations given pre-computed mean and std
def pre_computed_mean_and_std( mean, std, grid, ax, predictions_include_conditional_std, alpha=0.2, **kwargs ):
    #
    # ~~~ Graph the median as a blue curve
    _, = ax.plot( grid.cpu(), mean.cpu(), label="Predicted Posterior Mean", linestyle="-", linewidth=0.5, color="blue" )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "+/- Two Standard Deviations"
    lo, hi = mean-2*std, mean+2*std
    _ = ax.fill_between( grid.cpu(), lo.cpu(), hi.cpu(), facecolor="blue", interpolate=True, alpha=alpha, label=(tittle if predictions_include_conditional_std else tittle+" Including Measurment Noise") )
    return ax

#
# ~~~ Just plot a the model as an ordinary function
def plot_gpr(
            fig,
            ax,
            grid,
            green_curve,    # ~~~ tensor with the same shape as `grid`
            x_train,
            y_train,        # ~~~ tensor with the same shape as `x_train`
            mean,           # ~~~ tensor with the same shape as `grid`
            std,            # ~~~ tensor with the same shape as `grid`
            predictions_include_conditional_std,    # ~~~ Boolean
            **kwargs
        ):
    return univar_figure(
            fig,
            ax,
            grid,
            green_curve,
            x_train,
            y_train,
            model = "None! All we need are the vectors `mean` and `std`",
            title="Gaussian Process Regression",
            blue_curve = lambda model,grid,ax: pre_computed_mean_and_std(mean,std,grid,ax,predictions_include_conditional_std),
            **kwargs
        )

#
# ~~~ Graph the mean +/- two standard deviations
def two_standard_deviations( predictions, grid, ax, predictions_include_conditional_std, alpha=0.2, how_many_individual_predictions=6, **kwargs ):
    #
    # ~~~ Extract summary stats from `predictions` assuming that each *column* of `predictions` is a sample from the posterior predictive distribution
    mean = predictions.mean(dim=-1)
    std  =  predictions.std(dim=-1) + conditional_std
    lo, hi = mean-2*std, mean+2*std
    #
    # ~~~ Graph the median as a blue curve
    _, = ax.plot( grid.cpu(), mean.cpu(), label="Posterior Predictive Mean", linestyle="-", linewidth=( 0.7 if how_many_individual_predictions>0 else 0.5 ), color="blue" )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if how_many_individual_predictions>0:
        n_posterior_samples = predictions.shape[-1]
        which_NNs = (np.linspace( 1, n_posterior_samples, min(n_posterior_samples,how_many_individual_predictions), dtype=np.int32 ) - 1).tolist()
        for j in which_NNs:
            if j==max(which_NNs):
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), label="A Sampled Network", linestyle="-", linewidth=(1 if how_many_individual_predictions>0 else 0.5), color="blue", alpha=(alpha+1)/2 )
            else:
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), linestyle="-", linewidth=(1 if how_many_individual_predictions>0 else 0.5), color="blue", alpha=(alpha+1)/2 )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "+/- Two Standard Deviations"
    _ = ax.fill_between( grid.cpu(), lo.cpu(), hi.cpu(), facecolor="blue", interpolate=True, alpha=alpha, label=(tittle if predictions_include_conditional_std else tittle+" Including Measurment Noise") )
    return ax

#
# ~~~ Given a matrix of predictions, plot the empirical mean and +/- 2*std bars
def plot_bnn_mean_and_std(
            fig,
            ax,
            grid,
            green_curve,    # ~~~ tensor with the same shape as `grid`
            x_train,
            y_train,        # ~~~ tensor with the same shape as `x_train`
            predictions,    # ~~~ matrix with number of rows len(predictions)==len(grid)==len(x_train)
            predictions_include_conditional_std,    # ~~~ Boolean
            how_many_individual_predictions,
            title,
            **kwargs
        ):
    return univar_figure(
            fig,
            ax,
            grid,
            green_curve,
            x_train,
            y_train,
            model = "None! All we need is the matrix of predictions",
            title = title,
            blue_curve = lambda model,grid,ax: two_standard_deviations( predictions, grid, ax, predictions_include_conditional_std, how_many_individual_predictions=how_many_individual_predictions ),
            **kwargs
        )

#
# ~~~ Graph a symmetric, empirical 95% confidence interval of a model with a median point estimate
def empirical_quantile( predictions, grid, ax, predictions_include_conditional_std, alpha=0.2, how_many_individual_predictions=6, **kwargs ):
    #
    # ~~~ Extract summary stats from `predictions` assuming that each *column* of `predictions` is a sample from the posterior predictive distribution
    lo,med,hi = predictions.quantile( q=torch.Tensor([0.05,0.5,0.95]), dim=-1 )
    #
    # ~~~ Graph the median as a blue curve
    _, = ax.plot( grid.cpu(), med.cpu(), label="Posterior Predictive Median", linestyle="-", linewidth=( 0.7 if how_many_individual_predictions>0 else 0.5 ), color="blue" )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if how_many_individual_predictions>0:
        n_posterior_samples = predictions.shape[-1]
        which_NNs = (np.linspace( 1, n_posterior_samples, min(n_posterior_samples,how_many_individual_predictions), dtype=np.int32 ) - 1).tolist()
        for j in which_NNs:
            if j==max(which_NNs):
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), label="A Sampled Network", linestyle="-", linewidth=(1 if how_many_individual_predictions>0 else 0.5), color="blue", alpha=(alpha+1)/2 )
            else:
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), linestyle="-", linewidth=(1 if how_many_individual_predictions>0 else 0.5), color="blue", alpha=(alpha+1)/2 )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "95% Empirical Quantile Interval"
    _ = ax.fill_between( grid.cpu(), lo.cpu(), hi.cpu(), facecolor="blue", interpolate=True, alpha=alpha, label=(tittle if predictions_include_conditional_std else tittle+" Including Measurment Noise") )
    return ax

#
# ~~~ Given a matrix of predictions, plot the empirical median and symmetric 95% confidence bars
def plot_bnn_empirical_quantiles(
            fig,
            ax,
            grid,
            green_curve,    # ~~~ tensor with the same shape as `grid`
            x_train,
            y_train,        # ~~~ tensor with the same shape as `x_train`
            predictions,    # ~~~ matrix with number of rows len(predictions)==len(grid)==len(x_train)
            predictions_include_conditional_std,    # ~~~ Boolean
            how_many_individual_predictions,
            title,
            **kwargs
        ):
    return univar_figure(
            fig,
            ax,
            grid,
            green_curve,
            x_train,
            y_train,
            model = "None! All we need is the matrix of predictions",
            title = title,
            blue_curve = lambda model,grid,ax: empirical_quantile( predictions, grid, ax, predictions_include_conditional_std, how_many_individual_predictions=how_many_individual_predictions ),
            **kwargs
        )


    # #
    # # ~~~ Draw from the posterior predictive distribuion
    # example_output = model( grid, resample_weights=False )
    # predictions = torch.column_stack([
    #         model(grid,resample_weights=True) + conditional_std*torch.randn_like(example_output)
    #         for _ in range(n_samples)
    #     ])

