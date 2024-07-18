
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
# ~~~ Package-specific utils
from bnns.utils import plot_nn, generate_json_filename, set_Dataset_attributes

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
from quality_of_life.my_visualization_utils import GifMaker
from quality_of_life.my_base_utils          import support_for_progress_bars, dict_to_json, json_to_dict, my_warn



### ~~~
## ~~~ Config/setup
### ~~~

#
# ~~~ Template for what the dictionary of hyperparmeters should look like
hyperparameter_template = {
    #
    # ~~~ MIsc.
    "DEVICE" : "cpu",
    "dtype" : "float",
    "seed" : 2024,
    #
    # ~~~ Which problem
    "data" : "univar_missing_middle",
    "model" : "univar_NN",
    #
    # ~~~ For training
    "Optimizer" : "Adam",
    "lr" : 0.0005,
    "batch_size" : 64,
    "n_epochs" : 200,
    #
    # ~~~ For visualization
    "make_gif" : True,
    "how_often" : 10,                   # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
    "initial_frame_repetitions" : 24,   # ~~~ for how many frames should the state of initialization be rendered
    "final_frame_repetitions" : 48,     # ~~~ for how many frames should the state after training be rendered
}

#
# ~~~ Use argparse to extract the file name from `python det_nn.py --json "my_hyperparmeters.json"` (https://stackoverflow.com/a/67731094)
parser = argparse.ArgumentParser()
try:
    parser.add_argument( '--json', type=str, required=True )
    input_json_filename = parser.parse_args().json
    input_json_filename = input_json_filename if input_json_filename.endswith(".json") else input_json_filename+".json"
except:
    print("")
    print("    Hint: try `python det_nn.py --json demo_det_nn.json`")
    print("")
    raise

#
# ~~~ Load the .json file into a dictionary
hyperparameters = json_to_dict(input_json_filename)

#
# ~~~ Load the dictionary's key/value pairs into the global namespace
globals().update(hyperparameters)       # ~~~ e.g., if hyperparameters=={ "a":1, "B":2 }, then this defines a=1 and B=2

#
# ~~~ Might as well fix a seed, e.g., for randomly shuffling the order of batches during training
torch.manual_seed(seed)

#
# ~~~ Handle the dtypes not writeable in .json format (e.g., if your dictionary includes the value `torch.optim.Adam` you save it as .json)
dtype = getattr(torch,dtype)            # ~~~ e.g., "float" (str) -> torch.float (torch.dtype) 
torch.set_default_dtype(dtype)
Optimizer = getattr(optim,Optimizer)    # ~~~ e.g., "Adam" (str) -> optim.Adam

#
# ~~~ Load the network architecture
try:
    model = import_module(f"bnns.models.{model}")   # ~~~ this is equivalent to `import bnns.models.<model> as model`
except:
    model = import_module(model)

NN = model.NN.to( device=DEVICE, dtype=dtype )

#
# ~~~ Load the data
try:
    data = import_module(f"bnns.data.{data}")   # ~~~ this is equivalent to `import bnns.data.<data> as data`
except:
    data = import_module(data)

D_train = set_Dataset_attributes( data.D_train, device=DEVICE, dtype=dtype )
D_test  =  set_Dataset_attributes( data.D_val, device=DEVICE, dtype=dtype )
data_is_univariate = (D_train[0][0].numel()==1)



### ~~~
## ~~~ Train a conventional neural network, for reference
### ~~~

optimizer = Optimizer( NN.parameters(), lr=lr )
dataloader = torch.utils.data.DataLoader( D_train, batch_size=batch_size )
loss_fn = nn.MSELoss()

#
# ~~~ Some plotting stuff
if data_is_univariate:
    grid = data.x_test.to( device=DEVICE, dtype=dtype )
    green_curve =  data.y_test.cpu().squeeze()
    x_train_cpu = data.x_train.cpu()
    y_train_cpu = data.y_train.cpu().squeeze()
    if make_gif:
        fig,ax = plt.subplots(figsize=(12,6))
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
        if data_is_univariate and make_gif and (e+1)%how_often==0:
            fig, ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN=NN )
            gif.capture()   # ~~~ save a picture of the current plot (whatever plt.show() would show)

#
# ~~~ Afterwards, develop the .gif if applicable
if data_is_univariate:
    if make_gif:
        gif.develop( destination="NN", fps=24 )
    else:
        fig,ax = plt.subplots(figsize=(12,6))
        fig, ax = plot_nn( fig, ax, grid, green_curve, x_train_cpu, y_train_cpu, NN=NN )
        plt.show()



### ~~~
## ~~~ Evaluate the trained model
### ~~~

hyperparameters["metric"] = "here, we will record metrics"



### ~~~
## ~~~ Save the results
### ~~~

if input_json_filename.startswith("demo"):
    my_warn(f'Results are not saved when the hyperparameter json filename starts with "demo" (in this case `{input_json_filename}`)')
else:
    output_json_filename = generate_json_filename()
    dict_to_json( hyperparameters, output_json_filename )

#