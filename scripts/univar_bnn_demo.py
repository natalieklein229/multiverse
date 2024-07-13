
### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Standard packages
import torch
from torch import nn
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

#
# ~~~ The guts of the model

from bnns.SequentialGaussianBNN import SequentialGaussianBNN
from bnns.SSGE import BaseScoreEstimator as SSGE_backend

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker, buffer, points_with_curves
from quality_of_life.my_torch_utils         import convert_Tensors_to_Dataset, nonredundant_copy_of_module_list
from quality_of_life.my_numpy_utils         import moving_average
from quality_of_life.my_base_utils          import support_for_progress_bars



### ~~~
## ~~~ Config
### ~~~

#
# ~~~ Misc.
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(2024)
torch.set_default_dtype(torch.float)    # ~~~ note: why doesn't torch.double work?


#
# ~~~ Regarding the training method
functional = True
Optimizer = torch.optim.Adam
batch_size = n_train
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



### ~~~
## ~~~ Define the network architecture
### ~~~

from models.univar_BNN_untrained import BNN
from models.univar_NN_untrained  import  NN
NN, BNN = NN.to(DEVICE), BNN.to(DEVICE)

### ~~~
## ~~~ Define the data
### ~~~

from bnns.data.univar_data.missing_middle import x_train, y_train, x_test, y_test
x_train, y_train, x_test, y_test = x_train.to(DEVICE), y_train.to(DEVICE), x_test.to(DEVICE), y_test.to(DEVICE)



### ~~~
## ~~~ Some objects and helper functions for making plots
### ~~~

grid = x_test.cpu()                     # ~~~ move to cpu in order to plot it
ground_truth = y_test.squeeze().cpu()   # ~~~ move to cpu in order to plot it
ylim = buffer( ground_truth.tolist(), multiplier=0.2 )  # ~~~ infer a good ylim
description_of_the_experiment = "Functional BNN Training" if functional else "Weight Space BNN Training (BBB)"
def populate_figure( fig, ax , point_estimate=None, std=None, title=None, extra_std=0. ):
    with torch.no_grad():
        point_estimate, std = BNN.posterior_predicted_mean_and_std( x_test, n_posterior_samples ) if (point_estimate is None and std is None) else (point_estimate,std) # on cpu
        try:
            std += extra_std
        except: # ~~~ if extra_std is on cuda
            std += extra_std.cpu()
    green_curve, = ax.plot( grid, ground_truth, label="Ground Truth", linestyle='--', linewidth=.5, color="green", )
    blue_curve, = ax.plot( grid, point_estimate, label="Predicted Posterior Mean", linestyle="-", linewidth=.5, color="blue" )
    _ = ax.scatter( x_train.cpu(), y_train.cpu(), color="green" )
    _ = ax.fill_between( grid, point_estimate-2*std, point_estimate+2*std, facecolor="blue", interpolate=True, alpha=0.3, label="95% Confidence Interval")
    _ = ax.set_ylim(ylim)
    _ = ax.legend()
    _ = ax.grid()
    _ = ax.set_title( description_of_the_experiment if title is None else title )
    _ = fig.tight_layout()
    return fig, ax




### ~~~
## ~~~ Train a conventional neural network, for reference
### ~~~

optimizer = Optimizer( NN.parameters(), lr=lr )
dataloader = torch.utils.data.DataLoader( convert_Tensors_to_Dataset(x_train,y_train), batch_size=batch_size )
loss_fn = nn.MSELoss()
if make_gif:
    gif = GifMaker()

with support_for_progress_bars():   # ~~~ this just supports green progress bars
    for e in trange( n_epochs, ascii=' >=', desc="Deterministic Training" ):
        #
        # ~~~ The actual training logic (totally conventional, hopefully familiar)
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = loss_fn(NN(X),y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #
        # ~~~ Plotting logic
        if make_gif and (e+1)%how_often==0:
            fig, ax = points_with_curves(   # ~~~ this function simply just makes a plot
                    x       =     x_train,
                    y       =     y_train,
                    curves  =     (NN,f),
                    show    =     False,
                    title   =     r"Conventional Training",
                    fig     =     (fig if "fig" in globals() else "new"),
                    ax      =     (ax if "ax" in globals() else "new")
                )
            gif.capture()   # ~~~ save a picture of the current plot (whatever plt.show() would show)
    #
    # ~~~ Afterwards, develop the .gif if applicable
    if make_gif:
        gif.develop( destination="NN", fps=24 )



# ### ~~~
# ## ~~~ Run GPR, for reference
# ### ~~~

# #
# # ~~~ Borrow from SSGE, the implementation of the sub-routines responsible for building the kernel matrix and estimating a good kernel bandwidth
# kernel_matrix = SSGE_backend().gram_matrix
# bandwidth_estimator = SSGE_backend().heuristic_sigma

# #
# # ~~~ Do GPR
# bw = 0.1 #bandwidth_estimator( x_test.unsqueeze(-1), x_train.unsqueeze(-1) )
# K_in    =   kernel_matrix( x_train.unsqueeze(-1), x_train.unsqueeze(-1), bw )
# K_out   =   kernel_matrix( x_test.unsqueeze(-1),  x_test.unsqueeze(-1),  bw )
# K_btwn  =   kernel_matrix( x_test.unsqueeze(-1),  x_train.unsqueeze(-1), bw )
# with torch.no_grad():
#     sigma2 = ((NN(x_train)-y_train)**2).mean() if conditional_std=="auto" else torch.tensor(conditional_std)**2

# K_inv = torch.linalg.inv( K_in + sigma2*torch.eye(n_train,device=DEVICE) )
# posterior_mean  =  (K_btwn@K_inv@y_train).squeeze()
# posterior_std  =  ( K_out - K_btwn@K_inv@K_btwn.T ).diag().sqrt()

# #
# # ~~~ Plot the result
# fig,ax = plt.subplots(figsize=(12,6))
# fig,ax = populate_figure( fig, ax, point_estimate=posterior_mean.cpu(), std=posterior_std.cpu(), title="Gaussian Process Regression" )
# plt.show()



### ~~~
## ~~~ Do Bayesian training
### ~~~

#
# ~~~ The optimizer and dataloader
dataloader = torch.utils.data.DataLoader( convert_Tensors_to_Dataset(x_train,y_train), batch_size=batch_size )
mean_optimizer = Optimizer( BNN.model_mean.parameters(), lr=lr )
std_optimizer  =  Optimizer( BNN.model_std.parameters(), lr=lr )

#
# ~~~ Specify, now, the assumed conditional variance for the likelihood function (i.e., for the theoretical data-generating proces)
with torch.no_grad():
    BNN.conditional_std = torch.sqrt(((NN(x_train)-y_train)**2).mean()) if conditional_std=="auto" else torch.tensor(conditional_std)

#
# ~~~ Plot the state of the posterior predictive distribution upon its initialization
if make_gif:
    gif = GifMaker()      # ~~~ essentially just a list of images
    fig,ax = plt.subplots(figsize=(12,6))
    fig,ax = populate_figure( fig, ax, extra_std=BNN.conditional_std if extra_std else 0. )    # ~~~ plot the current state of the model
    for j in range(initial_frame_repetitions):
        gif.capture( clear_frame_upon_capture=(j+1==initial_frame_repetitions) )

#
# ~~~ Do Bayesian training
metrics = ( "ELBO", "post", "prior", "like" )
history = {}
for metric in metrics:
    history[metric] = []

#
# ~~~ Define how to project onto the constraint set
if project:
    BNN.rho = lambda x:x
    def projection_step(BNN):
        with torch.no_grad():
            for p in BNN.model_std.parameters():
                p.data = torch.clamp( p.data, min=projection_tol )
    projection_step(BNN)

#
# ~~~ Define the measurement set for functional training
BNN.measurement_set = x_train


# torch.autograd.set_detect_anomaly(True)
with support_for_progress_bars():   # ~~~ this just supports green progress bars
    pbar = tqdm( desc=description_of_the_experiment, total=n_epochs*len(dataloader), ascii=' >=' )
    for e in range(n_epochs):
        #
        # ~~~ Training logic
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            for j in range(n_MC_samples):
                #
                # ~~~ Compute the gradient of the loss function
                if functional:
                    log_posterior_density, log_prior_density = BNN.functional_kl(resample_measurement_set=False)
                else:
                    BNN.sample_from_standard_normal()   # ~~~ draw a new Monte-Carlo sample for estimating the integrals as an MC average
                    log_posterior_density   =   BNN.log_posterior_density()
                    log_prior_density       =   BNN.log_prior_density()
            #
            # ~~~ Add the the likelihood term and differentiate
            log_likelihood_density = BNN.log_likelihood_density(X,y)
            negative_ELBO = ( log_posterior_density - log_prior_density - log_likelihood_density )/n_MC_samples
            negative_ELBO.backward()
            #
            # ~~~ This would be training based only on the data:
            # loss = -BNN.log_likelihood_density(X,y)
            # loss.backward()
            #
            # ~~~ Do the gradient-based update
            for optimizer in (mean_optimizer,std_optimizer):
                optimizer.step()
                optimizer.zero_grad()
            #
            # ~~~ Do the projection
            if project:
                projection_step(BNN)
            #
            # ~~~ Record some diagnostics
            history["ELBO"].append( -negative_ELBO.item())
            history["post"].append( log_posterior_density.item())
            history["prior"].append(log_prior_density.item())
            history["like"].append( log_likelihood_density.item())
            to_print = {
                "ELBO" : f"{-negative_ELBO.item():<4.2f}",
                "post" : f"{log_posterior_density.item():<4.2f}",
                "prior": f"{log_prior_density.item():<4.2f}",
                "like" : f"{log_likelihood_density.item():<4.2f}"
            }
            pbar.set_postfix(to_print)
            _ = pbar.update()
        #
        # ~~~ Plotting logic
        if make_gif and n_posterior_samples>0 and (e+1)%how_often==0:
            fig,ax = populate_figure(fig,ax)
            gif.capture()
            # print("captured")

#
# ~~~ Plot the state of the posterior predictive distribution at the end of training
if make_gif:
    for j in range(final_frame_repetitions):
        gif.frames.append( gif.frames[-1] )
    gif.develop( destination=os.path.join("scaled_by_12",("fBNN" if functional else "BBB")+f" Cv={BNN.conditional_std*scale}, e={n_epochs}, lr={lr}"), fps=24 )
    plt.close()
else:
    fig,ax = plt.subplots(figsize=(12,6))
    fig,ax = populate_figure( fig, ax, extra_std=BNN.conditional_std if extra_std else 0. )
    plt.show()

pbar.close()



# fig, (ax_gpr,ax_bbb) = plt.subplots(1,2,figsize=(12,6))
# posterior_mean  =  (K_btwn@K_inv@y_train).squeeze()
# posterior_std  =  ( K_out - K_btwn@K_inv@K_btwn.T ).diag().sqrt()
# fig,ax_gpr = populate_figure( fig, ax_gpr, point_estimate=posterior_mean.cpu(), std=posterior_std.cpu(), title="Gaussian Process Regression" )
# fig,ax_bbb = populate_figure( fig, ax_bbb )
# plt.show()



### ~~~
## ~~~ Do a Stein neural network ensemble
### ~~~

#
# ~~~ Instantiate an ensemble
with torch.no_grad():
    conditional_std = torch.sqrt(((NN(x_train)-y_train)**2).mean()) if conditional_std=="auto" else torch.tensor(conditional_std)

ensemble = Ensemble(
        architecture = nonredundant_copy_of_module_list(NN),
        n_copies = n_Stein_particles,
        Optimizer = lambda params: Optimizer( params, lr=lr ),
        conditional_std = conditional_std
    )

#
# ~~~ The dataloader
dataloader = torch.utils.data.DataLoader( convert_Tensors_to_Dataset(x_train,y_train), batch_size=batch_size )

#
# ~~~
description_of_the_experiment = "Stein Neural Network Ensemble"
if PLOT_INDIVIDUAL_NNs:
    def ensemble_figure( fig, ax , point_estimate=None, std=None, title=None, how_many=18 ):
        with torch.no_grad():
            preds = ensemble(x_test)
        green_curve, = ax.plot( grid, ground_truth, label="Ground Truth", linestyle='--', linewidth=.5, color="green", )
        for j in range(how_many):
            j+= 80
            blue_curve, = ax.plot( grid, preds[:,j].cpu(), label=f"Network {j}", linestyle="-", linewidth=.5, color="blue" )
        _ = ax.scatter( x_train.cpu(), y_train.cpu(), color="green" )
        _ = ax.set_ylim(ylim)
        _ = ax.legend()
        _ = ax.grid()
        _ = ax.set_title( description_of_the_experiment if title is None else title )
        _ = fig.tight_layout()
        return fig, ax
else:
    def ensemble_figure( fig, ax, extra_std=ensemble.conditional_std if extra_std else 0. ):
        with torch.no_grad():
            preds = ensemble(x_test)
            return populate_figure( fig, ax, point_estimate=preds.mean(dim=-1).cpu(), std=preds.std(dim=-1).cpu(), title="Stein Neural Network Ensemble", extra_std=extra_std )

#
# ~~~ Plot the state of the posterior predictive distribution upon its initialization
if make_gif:
    gif = GifMaker()      # ~~~ essentially just a list of images
    fig,ax = plt.subplots(figsize=(12,6))
    fig,ax = ensemble_figure( fig, ax, extra_std=ensemble.conditional_std if extra_std else 0. )
    for j in range(initial_frame_repetitions):
        gif.capture( clear_frame_upon_capture=(j+1==initial_frame_repetitions) )

#
# ~~~ Do the actual training loop
K_history, grads_of_K_history = [], []
with support_for_progress_bars():   # ~~~ this just supports green progress bars
    for e in trange( n_epochs, ascii=' >=', desc="Stein Enemble" ):
        #
        # ~~~ Training logic
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            # ensemble.train_step(X,y)
            K, grads_of_K = ensemble.train_step(X,y)
            K_history.append( (torch.eye( *K.shape, device=K.device ) - K).abs().mean().item() )
            grads_of_K_history.append( grads_of_K.abs().mean().item() )
        #
        # ~~~ Plotting logic
        if make_gif and n_posterior_samples>0 and (e+1)%how_often==0:
            fig,ax = ensemble_figure( fig, ax, extra_std=ensemble.conditional_std if extra_std else 0. )
            gif.capture()
            # print("captured")

#
# ~~~ Plot the state of the posterior predictive distribution at the end of training
if not make_gif:    # ~~~ make a plot now
    fig,ax = plt.subplots(figsize=(12,6))

fig,ax = ensemble_figure( fig, ax, extra_std=ensemble.conditional_std if extra_std else 0. )

if make_gif:
    for j in range(final_frame_repetitions):
        gif.capture( clear_frame_upon_capture=(j+1==final_frame_repetitions) )
    gif.develop( destination=os.path.join("scaled_by_12",f"Stein Ensemble, Cv={ensemble.conditional_std*scale}, bw={ensemble.bw}, e={n_Stein_iterations}, lr={lr}"), fps=24 )
else:
    plt.show()



### ~~~
## ~~~ Diagnostics
### ~~~

def plot( metric, window_size=n_epochs/50 ):
    plt.plot( moving_average(history[metric],int(window_size)) )
    plt.grid()
    plt.tight_layout()
    plt.show()

#