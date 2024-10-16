
### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Standard packages
import os
import torch
from torch import nn
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from importlib import import_module

#
# ~~~ The guts of the model

from bnns.SequentialGaussianBNN import SequentialGaussianBNN
from bnns.Stein_GD import SequentialSteinEnsemble as Ensemble
from bnns.SSGE import BaseScoreEstimator as SSGE_backend

#
# ~~~ Package-specific utils
from bnns.utils import plot_nn, plot_gpr, plot_bnn_mean_and_std, plot_bnn_empirical_quantiles

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
from quality_of_life.my_torch_utils         import nonredundant_copy_of_module_list
from quality_of_life.my_numpy_utils         import moving_average
from quality_of_life.my_base_utils          import support_for_progress_bars, dict_to_json, json_to_dict



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
functional = False
Optimizer = torch.optim.Adam
batch_size = 64
lr = 0.0005
n_epochs = 200
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
visualize_bnn_using_quantiles = False
how_many_individual_predictions = 6

#
# ~~~ Regarding the data
data = "univar_missing_middle"



### ~~~
## ~~~ Define the network architecture
### ~~~

from bnns.models.univar_BNN import BNN
from bnns.models.univar_NN  import  NN
NN, BNN = NN.to(DEVICE), BNN.to(DEVICE)



### ~~~
## ~~~ Define the data
### ~~~

try:
    data = import_module("bnns.data."+data)
except:
    data = import_module(data)

x_train, y_train, x_test, y_test = data.x_train.to(DEVICE), data.y_train.to(DEVICE), data.x_test.to(DEVICE), data.y_test.to(DEVICE)



### ~~~
## ~~~ Define some objects used for plotting
### ~~~

grid = x_test
green_curve =  y_test.cpu().squeeze()
x_train_cpu = x_train.cpu()
y_train_cpu = y_train.cpu().squeeze()
plot_predictions = plot_bnn_empirical_quantiles if visualize_bnn_using_quantiles else plot_bnn_mean_and_std




### ~~~
## ~~~ Train a conventional neural network, for reference
### ~~~

optimizer = Optimizer( NN.parameters(), lr=lr )
dataloader = torch.utils.data.DataLoader( torch.utils.data.TensorDataset(x_train,y_train), batch_size=batch_size )
loss_fn = nn.MSELoss()

#
# ~~~ Some plotting stuff
fig,ax = plt.subplots(figsize=(12,6))
if make_gif:
    gif = GifMaker()

with support_for_progress_bars():   # ~~~ this just supports green progress bars
    for e in trange( n_epochs, ascii=' >=', desc="Conventional, Deterministic Training" ):
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
            fig, ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN=NN )
            gif.capture()   # ~~~ save a picture of the current plot (whatever plt.show() would show)

#
# ~~~ Afterwards, develop the .gif if applicable
if make_gif:
    gif.develop( destination="NN", fps=24 )
else:
    fig, ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN=NN )
    plt.show()



### ~~~
## ~~~ Run GPR, for reference
### ~~~

#
# ~~~ Borrow from SSGE, the implementation of the sub-routines responsible for building the kernel matrix and estimating a good kernel bandwidth
kernel_matrix = SSGE_backend().gram_matrix
bandwidth_estimator = SSGE_backend().heuristic_sigma

#
# ~~~ Do GPR
bw = 0.1 #bandwidth_estimator( x_test.unsqueeze(-1), x_train.unsqueeze(-1) )
K_in    =   kernel_matrix( x_train.unsqueeze(-1), x_train.unsqueeze(-1), bw )
K_out   =   kernel_matrix( x_test.unsqueeze(-1),  x_test.unsqueeze(-1),  bw )
K_btwn  =   kernel_matrix( x_test.unsqueeze(-1),  x_train.unsqueeze(-1), bw )
with torch.no_grad():
    sigma2 = ((NN(x_train)-y_train)**2).mean() if conditional_std=="auto" else torch.tensor(conditional_std)**2

K_inv = torch.linalg.inv( K_in + sigma2*torch.eye(len(x_train),device=DEVICE) )
posterior_mean  =  (K_btwn@K_inv@y_train).squeeze()
posterior_std  =  ( K_out - K_btwn@K_inv@K_btwn.T ).diag().sqrt()

#
# ~~~ Plot the result
fig,ax = plt.subplots(figsize=(12,6))
fig,ax = plot_gpr( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, mean = (posterior_mean+sigma2 if extra_std else posterior_mean), std = posterior_std, predictions_include_conditional_std = extra_std )
plt.show()



### ~~~
## ~~~ Do Bayesian training
### ~~~

#
# ~~~ The optimizer and dataloader
dataloader = torch.utils.data.DataLoader( torch.utils.data.TensorDataset(x_train,y_train), batch_size=batch_size )
mean_optimizer = Optimizer( BNN.model_mean.parameters(), lr=lr )
std_optimizer  =  Optimizer( BNN.model_std.parameters(), lr=lr )

#
# ~~~ Specify, now, the assumed conditional variance for the likelihood function (i.e., for the theoretical data-generating proces)
with torch.no_grad():
    BNN.conditional_std = torch.sqrt(((NN(x_train)-y_train)**2).mean()) if conditional_std=="auto" else torch.tensor(conditional_std)

#
# ~~~ Some plotting stuff
description_of_the_experiment = "fBNN" if functional else "BBB"
def plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, bnn, predictions_include_conditional_std=extra_std, how_many_individual_predictions=how_many_individual_predictions, n_posterior_samples=n_posterior_samples, title=description_of_the_experiment, prior=False ):
    #
    # ~~~ Draw from the posterior predictive distribuion
    with torch.no_grad():
        forward = bnn.prior_forward if prior else bnn
        predictions = torch.column_stack([ forward(grid,resample_weights=True) for _ in range(n_posterior_samples) ])
        if predictions_include_conditional_std:
            predictions += bnn.conditional_std * torch.randn_like(predictions)
    return plot_predictions( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, predictions, predictions_include_conditional_std, how_many_individual_predictions, title )

#
# ~~~ Plot the state of the posterior predictive distribution upon its initialization
if make_gif:
    gif = GifMaker()      # ~~~ essentially just a list of images
    fig,ax = plt.subplots(figsize=(12,6))
    fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN, prior=True )
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
            fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN )
            gif.capture()
            # print("captured")

#
# ~~~ Plot the state of the posterior predictive distribution at the end of training
if make_gif:
    for j in range(final_frame_repetitions):
        gif.frames.append( gif.frames[-1] )
    gif.develop( destination=description_of_the_experiment, fps=24 )
    plt.close()
else:
    fig,ax = plt.subplots(figsize=(12,6))
    fig,ax = plot_bnn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, BNN )
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
dataloader = torch.utils.data.DataLoader( torch.utils.data.TensorDataset(x_train,y_train), batch_size=batch_size )

#
# ~~~ Some plotting stuff
description_of_the_experiment = "Stein Neural Network Ensemble"
def plot_esnsemble( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble, predictions_include_conditional_std=extra_std, how_many_individual_predictions=how_many_individual_predictions, title=description_of_the_experiment ):
    #
    # ~~~ Draw from the posterior predictive distribuion
    with torch.no_grad():
        predictions = ensemble(grid)
        if predictions_include_conditional_std:
            predictions += ensemble.conditional_std * torch.randn_like(predictions)
    return plot_predictions( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, predictions, predictions_include_conditional_std, how_many_individual_predictions, title )


#
# ~~~ Plot the state of the posterior predictive distribution upon its initialization
if make_gif:
    gif = GifMaker()      # ~~~ essentially just a list of images
    fig,ax = plt.subplots(figsize=(12,6))
    fig,ax = plot_esnsemble( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble )
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
            fig,ax = plot_esnsemble( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble )
            gif.capture()
            # print("captured")

#
# ~~~ Plot the state of the posterior predictive distribution at the end of training
if not make_gif:    # ~~~ make a plot now
    fig,ax = plt.subplots(figsize=(12,6))

fig,ax = plot_esnsemble( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, ensemble )

if make_gif:
    for j in range(final_frame_repetitions):
        gif.capture( clear_frame_upon_capture=(j+1==final_frame_repetitions) )
    gif.develop( destination=description_of_the_experiment, fps=24 )
else:
    plt.show()



# ### ~~~
# ## ~~~ Diagnostics
# ### ~~~

# def plot( metric, window_size=n_epochs/50 ):
#     plt.plot( moving_average(history[metric],int(window_size)) )
#     plt.grid()
#     plt.tight_layout()
#     plt.show()

# #
