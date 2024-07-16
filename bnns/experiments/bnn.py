
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
#
# ~~~ Package-specific utils
from bnns.utils import plot_bnn_mean_and_std, plot_bnn_empirical_quantiles

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
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

#
# ~~~ Regarding the SSGE
M = 50          # ~~~ M in SSGE
J = 10          # ~~~ J in SSGE
eta = 0.0001    # ~~~ stability term added to the SSGE's RBF kernel

#
# ~~~ Regarding visualizaing of training
make_gif = True         # ~~~ if true, aa .gif is made (even if false, the function is still plotted)
how_often = 10          # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
initial_frame_repetitions = 24  # ~~~ for how many frames should the state of initialization be rendered
final_frame_repetitions = 48    # ~~~ for how many frames should the state after training be rendered
plot_indivitual_NNs = False     # ~~~ if True, do *not* plot confidence intervals and, instead, plot only a few sampled nets
visualize_bnn_using_quantiles = False
how_many_individual_predictions = 6

#
# ~~~ Regarding the likelihood model
conditional_std = 0.19

#
# ~~~ Regarding the predictions
extra_std = False               # ~~~ if True, add the conditional std. when plotting the +/- 2 standard deviation bars

#
# ~~~ Regarding the data
data = "univar_missing_middle"

#
# ~~~ Regarding the model
model = "univar_BNN"



### ~~~
## ~~~ Load the network architecture
### ~~~

#
# ~~~ `import bnns.models.<model> as model`
try:
    model = import_module(f"bnns.models.{model}")
except:
    model = import_module(model)

BNN = model.BNN.to(DEVICE)



### ~~~
## ~~~ Load the data
### ~~~

#
# ~~~ `import bnns.data.<data> as data`
try:
    data = import_module(f"bnns.data.{data}")
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
## ~~~ Do Bayesian training
### ~~~

#
# ~~~ The optimizer and dataloader
dataloader = torch.utils.data.DataLoader( torch.utils.data.TensorDataset(x_train,y_train), batch_size=batch_size )
mean_optimizer = Optimizer( BNN.model_mean.parameters(), lr=lr )
std_optimizer  =  Optimizer( BNN.model_std.parameters(), lr=lr )

#
# ~~~ Specify, now, the assumed conditional variance for the likelihood function (i.e., for the theoretical data-generating proces)
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



# ### ~~~
# ## ~~~ Diagnostics
# ### ~~~

# def plot( metric, window_size=n_epochs/50 ):
#     plt.plot( moving_average(history[metric],int(window_size)) )
#     plt.grid()
#     plt.tight_layout()
#     plt.show()

# #
