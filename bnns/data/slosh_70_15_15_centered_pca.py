
import numpy as np
import torch

try:
    #
    # ~~~ Load the processed data
    U = torch.load("slosh_centered_U.pt")
    s = torch.load("slosh_centered_s.pt")
    V = torch.load("slosh_centered_V.pt")
except:
    #
    # ~~~ Load the unprocessed data
    from bnns.data.slosh_70_15_15 import coords_np, inputs_np, out_np
    data_matrix = torch.from_numpy( out_np - np.mean(out_np,axis=0) )
    #
    # ~~~ Process the data (do SVD)
    evals, evecs = torch.linalg.eigh(data_matrix@data_matrix.T)
    s_squared = evals.flip(dims=(0,)) # ~~~ the squared singular values of `data_matrix`
    percentage_of_variance_explained = s_squared.cumsum(dim=0)/s_squared.sum()
    r = (percentage_of_variance_explained<.99).int().argmin().item()    # ~~~ the first index at which percentage_of_variance_explained>=.99
    torch.manual_seed(2024)     # ~~~ torch.svd_lowrank is stochastic
    U, s, V = torch.linalg.svd( data_matrix, full_matrices=False )
    #
    # ~~~ Save the processed data
    torch.save( U, "slosh_centered_U.pt" )
    torch.save( s, "slosh_centered_s.pt" )
    torch.save( V, "slosh_centered_V.pt" )

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

#
# ~~~ Use indices for a train/val/test split
x_train = torch.from_numpy(inputs_np[idx_train])
x_test = torch.from_numpy(inputs_np[idx_test])
x_val = torch.from_numpy(inputs_np[idx_val])
y_train = U[idx_train]
y_test = U[idx_test]
y_val = U[idx_val]

#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)
