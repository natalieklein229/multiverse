
import torch
from torch.func import jacrev, functional_call
from bnns.models.slosh_NN import NN
from bnns.utils import manual_Jacobian
from tqdm import trange

#
# ~~~ Make up some data
torch.manual_seed(2024)
x = torch.randn(50,5)
batch_size = x.shape[0]
number_of_output_features = NN[-1].out_features # ~~~ for slosh nets, 49719
V = x
for j in range(len(NN)-1):
    V = NN[j](V)

final_J = jacrev(functional_call, argnums=1)( NN[-1], dict(NN[-1].named_parameters()), (V,) )
final_J = final_J["weight"].reshape(number_of_output_features,-1)