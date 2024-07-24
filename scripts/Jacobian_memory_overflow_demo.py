
from torch.func import jacrev, functional_call
from bnns.models.slosh_NN import NN
from bnns.data.slosh_70_15_15 import D_train
x_train = D_train.X.float()
x = x_train[:100]

#
# ~~~ Compute the full Jacobian using torch.func
jacobians = jacrev(functional_call, argnums=1)(NN, dict(NN.named_parameters()), (x,)) # ~~~ a dictionary with the same keys as NN.named_parameters()
