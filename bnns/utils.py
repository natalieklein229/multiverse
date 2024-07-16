
import numpy as np
import torch
try:
    from quality_of_life.my_base_utils import buffer
except:
    from quality_of_life.my_visualization_utils import buffer   # ~~~ deprecated
    print("Please update quality_of_life")


class TrainingConfig:
    DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

#
# ~~~ Regarding the training method
functional = True
Optimizer = torch.optim.Adam
batch_size = 64
lr = 0.0005
n_epochs = 2000
n_posterior_samples = 100   # ~~~ posterior distributions are approximated as empirical dist.'s of this many samples
n_MC_samples = 20           # ~~~ expectations are estimated as an average of this many Monte-Carlo samples
project = True              # ~~~ if True, use projected gradient descent; else use the weird thing from the paper
projection_tol = 1e-6       # ~~~ for numerical reasons, project onto [projection_tol,Inf), rather than onto [0,Inft)
conditional_std = 0.9       # ~~~ what Natalie was explaining to me on Tuesday

#
# ~~~ Regarding the SSGE
M = 50          # ~~~ M in SSGE
J = 10          # ~~~ J in SSGE
eta = 0.0001    # ~~~ stability term added to the SSGE's RBF kernel

#
# ~~~ Regarding Stein GD
n_Stein_particles = n_posterior_samples
n_Stein_iterations = n_epochs

#
# ~~~ Regarding visualizaing of training
make_gif = True         # ~~~ if true, aa .gif is made (even if false, the function is still plotted)
how_often = 10          # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
initial_frame_repetitions = 24  # ~~~ for how many frames should the state of initialization be rendered
final_frame_repetitions = 48    # ~~~ for how many frames should the state after training be rendered
plot_indivitual_NNs = False     # ~~~ if True, do *not* plot confidence intervals and, instead, plot only a few sampled nets
extra_std = False               # ~~~ if True, add the conditional std. when plotting the +/- 2 standard deviation bars


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
    _, = ax.plot( grid.cpu(), mean.cpu(), label="Posterior Predictive Mean", linestyle="-", linewidth=( 0.7 if plot_indivitual_NNs else 0.5 ), color="blue" )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if how_many_individual_predictions>0:
        which_NNs = (np.linspace( 1, n_posterior_samples, min(n_posterior_samples,how_many_individual_predictions), dtype=np.int32 ) - 1).tolist()
        for j in which_NNs:
            if j==max(which_NNs):
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), label="A Sampled Network", linestyle="-", linewidth=(1 if plot_indivitual_NNs else 0.5), color="blue", alpha=(alpha+1)/2 )
            else:
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), linestyle="-", linewidth=(1 if plot_indivitual_NNs else 0.5), color="blue", alpha=(alpha+1)/2 )
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
    _, = ax.plot( grid.cpu(), med.cpu(), label="Posterior Predictive Median", linestyle="-", linewidth=( 0.7 if plot_indivitual_NNs else 0.5 ), color="blue" )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if how_many_individual_predictions>0:
        which_NNs = (np.linspace( 1, n_posterior_samples, min(n_posterior_samples,how_many_individual_predictions), dtype=np.int32 ) - 1).tolist()
        for j in which_NNs:
            if j==max(which_NNs):
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), label="A Sampled Network", linestyle="-", linewidth=(1 if plot_indivitual_NNs else 0.5), color="blue", alpha=(alpha+1)/2 )
            else:
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), linestyle="-", linewidth=(1 if plot_indivitual_NNs else 0.5), color="blue", alpha=(alpha+1)/2 )
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

