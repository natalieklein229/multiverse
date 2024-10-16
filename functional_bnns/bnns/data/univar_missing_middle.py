
import torch
torch.manual_seed(1234)
from quality_of_life.my_torch_utils import convert_Tensors_to_Dataset

#
# ~~~ Data settings
n_train = 50
n_test = 500
n_val = 30
noise = 0.2  # ~~~ pollute y_train wth Gaussian noise of variance noise**2
f = lambda x: 2*torch.cos(torch.pi*(x+0.2)) + torch.exp(2.5*(x+0.2))/2.5 - 2.25 # ~~~ the ground truth (subtract a term so that the response is centered around 0)

#
# ~~~ Synthetic (noisy) training data
x_train = 2*torch.rand( size=(n_train,) )**2 - 1    # ~~~ uniformly random points in [-1,1]
x_train = x_train.sign() * x_train.abs()**(1/6)     # ~~~ push it away from zero
y_train = f(x_train) + noise*torch.randn( size=(n_train,) )

#
# ~~~ Synthetic (noise-less) test data
x_test = torch.linspace( -1.5, 1.5, n_test )
y_test = f(x_test)

#
# ~~~ Synthetic (noisy) validation data
x_val = 2*torch.rand( size=(n_val,) )**2 - 1        # ~~~ uniformly random points in [-1,1]
x_val = x_val.sign() * x_val.abs()**(1/6)         # ~~~ push it away from zero
y_val = f(x_val) + noise*torch.randn( size=(n_val,) )

#
# ~~~ Reshape y data in order to be consistent with the shape returned by a model with final layer nn.Linear(m,1)
y_train = y_train.reshape(-1,1)    
y_test  =  y_test.reshape(-1,1)
y_val   =   y_val.reshape(-1,1)

#
# ~~~ Rename the function according to how it will be imported
ground_truth = f

#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)