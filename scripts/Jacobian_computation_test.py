
imoprt torch
from torch.func import jacrev, functional_call
from bnns.models.slosh_NN import NN
from bnns.utils import manual_Jacobian


#
# ~~~ Make up some data
torch.manual_seed(2024)
x = torch.randn(50,5)
batch_size = x.shape[0]


final_J = jacrev(functional_call, argnums=1)( NN[-1], dict(NN[-1].named_parameters()), (v,) )