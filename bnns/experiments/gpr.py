
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
from bnns.SSGE import BaseScoreEstimator as SSGE_backend

#
# ~~~ Package-specific utils
from bnns.utils import plot_gpr

#
# ~~~ My Personal Helper Functions (https://github.com/ThomasLastName/quality_of_life)
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
# ~~~ Regarding the likelihood model
conditional_std = 0.19

#
# ~~~ Regarding the predictions
extra_std = False               # ~~~ if True, add the conditional std. when plotting the +/- 2 standard deviation bars

#
# ~~~ Regarding the data
data = "univar_missing_middle"



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


