
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
# ~~~ Train/val/test split
np.random.seed(2024)
assert len(inputs_np)==4000==len(out_np)
n = len(inputs_np)
idx = np.random.permutation(n)
percent_train = 0.5
percent_test  = 0.25
n_train = int(n*percent_train)
n_test = int(n*percent_test)
n_val = n - n_train - n_test

