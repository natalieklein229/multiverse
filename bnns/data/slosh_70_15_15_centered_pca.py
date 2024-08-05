
import numpy as np
import torch

try:
    U = np.load("slosh_centered_U.npy")
    s = np.load("slosh_centered_s.npy")
    V = np.load("slosh_centered_V.npy")
except:
    from bnns.data.slosh_70_15_15 import coords_np, inputs_np, out_np, idx_train, idx_test, idx_val
    U, s, V = np.linalg.svd( out_np - np.mean(out_np,axis=0) )
    np.save( "slosh_centered_U.npy", U )
    np.save( "slosh_centered_s.npy", s )
    np.save( "slosh_centered_V.npy", V )

#
# ~~~ Compute indices for a train/val/test split (same code as in slosh_70_15_15.py)
np.random.seed(2024)
n_train = 2600
n_test = 700
n_val = 700
n = len(inputs_np)
assert len(inputs_np) == 4000 == len(out_np)
assert n_train + n_test + n_val == n
idx = np.random.permutation(n)
idx_train, idx_test, idx_val = np.split( idx, [n_train,n_train+n_test] )
