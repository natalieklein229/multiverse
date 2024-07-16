
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
# ~~~ Package-specific utils
from bnns.utils import plot_nn, generate_json_filename

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
from quality_of_life.my_base_utils          import support_for_progress_bars, dict_to_json, json_to_dict



### ~~~
## ~~~ Config
### ~~~

hyperparameters = json_to_dict("new_trial.json")

#
# ~~~ Misc.
DEVICE = hyperparameters["DEVICE"]
torch.manual_seed(hyperparameters["seed"])
torch.set_default_dtype(torch.float)    # ~~~ note: why doesn't torch.double work?

#
# ~~~ Regarding the training method
Optimizer = hyperparameters["Optimizer"]
batch_size = hyperparameters["batch_size"]
lr = hyperparameters["lr"]
n_epochs = hyperparameters["lr"]

#
# ~~~ Regarding visualizaing of training
make_gif = hyperparameters["make_gif"]         # ~~~ if true, aa .gif is made (even if false, the function is still plotted)
how_often = hyperparameters["how_often"]          # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
initial_frame_repetitions = hyperparameters["initial_frame_repetitions"]  # ~~~ for how many frames should the state of initialization be rendered
final_frame_repetitions = hyperparameters["final_frame_repetitions"]     # ~~~ for how many frames should the state after training be rendered

#
# ~~~ Regarding the data
data = hyperparameters["data"]

#
# ~~~ Regarding the model
model = hyperparameters["model"]



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
## ~~~ Save the results
### ~~~

hyperparameters["metric"] = "here, we will record metrics"
file_name = generate_json_filename()
dict_to_json(file_name)
