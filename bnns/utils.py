
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
        _, = ax.plot(    grid.cpu(), green_curve.cpu(), color="green", label="Ground Truth", linestyle='--', linewidth=.5 )
        _ = ax.scatter( x_train.cpu(), y_train.cpu(),   color="green" )
        #
        # ~~~ Blue curve(s) of the model
        try:
            ax = blue_curve( model, grid, ax, **kwargs )
        except:
            ax = blue_curve( model, grid, ax ) 
        #
        # ~~~ Finish up
        _ = ax.set_ylim(ylim)
        _ = ax.legend()
        _ = ax.grid()
        _ = ax.set_title( description_of_the_experiment if title is None else title )
        _ = fig.tight_layout()
    return fig, ax

#
# ~~~ Basically just plot a function
def trivial_sampler(f,grid,ax):
    _, = ax.plot( grid.cpu(), f(grid).cpu(), label="Neural Network", linestyle="-", linewidth=.5, color="blue" )
    return ax


#
# ~~~ Graph the mean +/- two standard deviations
def two_standard_deviations( model, grid, ax, n_samples=100, conditional_std=0., alpha=0.2, plot_indivitual_NNs=True, how_many=6, **kwargs ):
    #
    # ~~~ Draw from the posterior predictive distribution
    predictions = torch.column_stack([
            model(grid,resample_weights=True)
            for _ in range(n_samples)
        ])
    mean = predictions.mean(dim=-1)
    std  =  predictions.std(dim=-1) + conditional_std
    lo, hi = mean-2*std, mean+2*std
    #
    # ~~~ Graph the median as a blue curve
    _, = ax.plot( grid.cpu(), mean.cpu(), label="Predicted Posterior Mean", linestyle="-", linewidth=( 0.7 if plot_indivitual_NNs else 0.5 ), color="blue" )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if plot_indivitual_NNs:
        which_NNs = (np.linspace( 1, n_posterior_samples, min(n_posterior_samples,how_many), dtype=np.int32 ) - 1).tolist()
        for j in which_NNs:
            if j==max(which_NNs):
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), label="A Sampled Network", linestyle="-", linewidth=(1 if plot_indivitual_NNs else 0.5), color="blue", alpha=(alpha+1)/2 )
            else:
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), linestyle="-", linewidth=(1 if plot_indivitual_NNs else 0.5), color="blue", alpha=(alpha+1)/2 )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "95% Empirical Quantile Interval"
    _ = ax.fill_between( grid.cpu(), lo.cpu(), hi.cpu(), facecolor="blue", interpolate=True, alpha=alpha, label=(tittle if conditional_std==0) else (tittle+" Including Measurment Noise") )
    return ax

#
# ~~~ Graph a symmetric, empirical 95% confidence interval of a model with a median point estimate
def empirical_quantile( model, grid, ax, n_samples=100, conditional_std=0., alpha=0.2, plot_indivitual_NNs=True, how_many=6, **kwargs ):
    #
    # ~~~ Draw from the posterior predictive distribuion
    example_output = model( grid, resample_weights=False )
    predictions = torch.column_stack([
            model(grid,resample_weights=True) + conditional_std*torch.randn_like(example_output)
            for _ in range(n_samples)
        ])
    lo,med,hi = predictions.quantile( q=torch.Tensor([0.05,0.5,0.95]), dim=-1 )
    #
    # ~~~ Graph the median as a blue curve
    _, = ax.plot( grid.cpu(), med.cpu(), label="Predicted Posterior Mean", linestyle="-", linewidth=( 0.7 if plot_indivitual_NNs else 0.5 ), color="blue" )
    #
    # ~~~ Optionally, also graph several of the actual sample NN's as more blue curves (label only the last one)
    if plot_indivitual_NNs:
        which_NNs = (np.linspace( 1, n_posterior_samples, min(n_posterior_samples,how_many), dtype=np.int32 ) - 1).tolist()
        for j in which_NNs:
            if j==max(which_NNs):
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), label="A Sampled Network", linestyle="-", linewidth=(1 if plot_indivitual_NNs else 0.5), color="blue", alpha=(alpha+1)/2 )
            else:
                _, = ax.plot( grid.cpu(), predictions[:,j].cpu(), linestyle="-", linewidth=(1 if plot_indivitual_NNs else 0.5), color="blue", alpha=(alpha+1)/2 )
    #
    # ~~~ Fill in a 95% confidence region
    tittle = "95% Empirical Quantile Interval"
    _ = ax.fill_between( grid.cpu(), lo.cpu(), hi.cpu(), facecolor="blue", interpolate=True, alpha=alpha, label=(tittle if conditional_std==0) else (tittle+" Including Measurment Noise") )
    return ax
