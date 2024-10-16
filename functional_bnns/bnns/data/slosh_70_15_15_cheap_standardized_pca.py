
import numpy as np
import torch
import os
from quality_of_life.my_torch_utils import convert_Tensors_to_Dataset
from quality_of_life.my_base_utils import find_root_dir_of_repo
from bnns.data.slosh_70_15_15 import coords_np, inputs_np, out_np, idx_train, idx_test, idx_val

#
# ~~~ Establish the path to the folder `bnns/data`
root = find_root_dir_of_repo()
PATH = os.path.join( root, "bnns", "data" )

#
# ~~~ Generate U, s, V, and r
try:
    #
    # ~~~ Load the processed data
    U_truncated = torch.load(os.path.join( PATH, "slosh_cheap_standardized_U_truncated.pt"))
    s_truncated = torch.load(os.path.join( PATH, "slosh_cheap_standardized_s_truncated.pt"))
    V_truncated = torch.load(os.path.join( PATH, "slosh_cheap_standardized_V_truncated.pt"))
    r = len(s_truncated)    # ~~~ 17
except:
    #
    # ~~~ Load the unprocessed data
    data_matrix = torch.from_numpy( (out_np - np.mean(out_np,axis=0)) / (np.std(out_np,axis=0)+1e-10) )
    #
    # ~~~ Process the data (do SVD)
    evals, evecs = torch.linalg.eigh(data_matrix@data_matrix.T)
    s_squared = evals.flip(dims=(0,)) # ~~~ the squared singular values of `data_matrix`
    percentage_of_variance_explained = s_squared.cumsum(dim=0)/s_squared.sum()
    r = (percentage_of_variance_explained<.99).int().argmin().item()    # ~~~ the first index at which percentage_of_variance_explained>=.99
    torch.manual_seed(2024)     # ~~~ torch.svd_lowrank is stochastic
    U_truncated, s_truncated, V_truncated = torch.svd_lowrank( data_matrix, r )
    #
    # ~~~ Save the processed data
    torch.save( U_truncated, os.path.join( PATH, "slosh_cheap_standardized_U_truncated.pt" ))
    torch.save( s_truncated, os.path.join( PATH, "slosh_cheap_standardized_s_truncated.pt" ))
    torch.save( V_truncated, os.path.join( PATH, "slosh_cheap_standardized_V_truncated.pt" ))

#
# ~~~ Use indices for a train/val/test split
x_train = torch.from_numpy(inputs_np[idx_train])
x_test = torch.from_numpy(inputs_np[idx_test])
x_val = torch.from_numpy(inputs_np[idx_val])
y_train = U_truncated[idx_train]
y_test = U_truncated[idx_test]
y_val = U_truncated[idx_val]

#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)
