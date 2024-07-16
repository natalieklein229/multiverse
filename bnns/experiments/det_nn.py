
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
from bnns.utils import plot_nn

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
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
Optimizer = torch.optim.Adam
batch_size = 64
lr = 0.0005
n_epochs = 200

#
# ~~~ Regarding visualizaing of training
make_gif = True         # ~~~ if true, aa .gif is made (even if false, the function is still plotted)
how_often = 10          # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
initial_frame_repetitions = 24  # ~~~ for how many frames should the state of initialization be rendered
final_frame_repetitions = 48    # ~~~ for how many frames should the state after training be rendered

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

