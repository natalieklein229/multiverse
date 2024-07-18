
import os
import torch
import pyreadr                  # ~~~ from https://stackoverflow.com/a/61699417
import numpy as np

#
# ~~~ Set path to the .rda file
PATH = 'slosh_dat_nj.rda'
if __name__ == "__main__":
    ans = input("    Is the path 'slosh_dat_nj.rda' correct?\n    Enter 'y' for yes, any other key for no.\n")
    if not ans.lower()=="y":
        PATH = input(".   Please type the path without quotes and press enter:\n") # ~~~ e.g., /Users/winckelman/Downloads/slosh_dat_nj.rda

#
# ~~~ Extract the data as numpy arrays
DATA = pyreadr.read_r(PATH)     # ~~~ from https://stackoverflow.com/a/61699417
coords_np  =  DATA["coords"].to_numpy()
inputs_np  =  DATA["inputs"].to_numpy()
out_np     =     DATA["out"].to_numpy()

#
# ~~~ Compute indices for a train/val/test split
np.random.seed(2024)
n_train = 2600
n_test = 700
n_val = 700
n = len(inputs_np)
assert len(inputs_np) == 4000 == len(out_np)
assert n_train + n_test + n_val == n
idx = np.random.permutation(n)
idx_train, idx_test, idx_val = np.split( idx, [n_train,n_train+n_test] )

#
# ~~~ Use indices for a train/val/test split
x_train, x_test, x_val = inputs_np[idx_train], 
#
# ~~~ Compute indices for a train/val/test split
np.random.seed(2024)
n_train = 2800
n_test = 600
n_val = 600
n = len(inputs_np)
assert len(inputs_np) == 4000 == len(out_np)
assert n_train + n_test + n_val == n
idx = np.random.permutation(n)
idx_train, idx_test, idx_val = np.split( idx, [n_train,n_train+n_test] )

#
# ~~~ Use indices for a train/val/test split
x_train = torch.from_numpy(inputs_np[idx_train])
x_test = torch.from_numpy(inputs_np[idx_test])
x_val = torch.from_numpy(inputs_np[idx_val])
y_train = torch.from_numpy(out_np[idx_train])
y_test = torch.from_numpy(out_np[idx_test])
y_val = torch.from_numpy(out_np[idx_val])

