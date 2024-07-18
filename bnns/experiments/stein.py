
### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Standard packages
import torch
from torch import nn, optim
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from importlib import import_module
import argparse

#
# ~~~ The guts of the model
from bnns.Stein_GD import SequentialSteinEnsemble as Ensemble

#
# ~~~ Package-specific utils
from bnns.utils import plot_bnn_mean_and_std, plot_bnn_empirical_quantiles, set_Dataset_attributes, generate_json_filename

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
from quality_of_life.my_torch_utils         import nonredundant_copy_of_module_list
from quality_of_life.my_numpy_utils         import moving_average
from quality_of_life.my_base_utils          import support_for_progress_bars, dict_to_json, json_to_dict, my_warn



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
Optimizer = torch.optim.Adam
batch_size = 64
lr = 0.0005
n_epochs = 200

#
# ~~~ Regarding Stein GD
n_Stein_particles = 100
n_Stein_iterations = 200

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
model = "univar_NN"



### ~~~
## ~~~ Load the network architecture
### ~~~

#
# ~~~ `import bnns.models.<model> as model`
try:
    model = import_module(f"bnns.models.{model}")
except:
    model = import_module(model)

NN = model.NN.to(DEVICE)



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
        if make_gif and (e+1)%how_often==0:
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
