
import torch
torch.manual_seed(1234)

#
# ~~~ Data settings
n_train = 50
noise = 0.2  # ~~~ pollute y_train wth Gaussian noise of variance noise**2
n_test = 500
f = lambda x: 2*torch.cos(torch.pi*(x+0.2)) + torch.exp(2.5*(x+0.2))/2.5 - 2.25 # ~~~ the ground truth (subtract a term so that the response is centered around 0)

#
# ~~~ Synthetic (noisy) training data
x_train = 2*torch.rand( size=(n_train,) )**2 - 1            # ~~~ uniformly random points in [-1,1]
x_train = x_train.sign() * x_train.abs()**(1/6)             # ~~~ push it away from zero
y_train = f(x_train) + noise*torch.randn( size=(n_train,) )

#
# ~~~ Synthetic (noise-less) test data
x_test = torch.linspace( -1.5, 1.5, n_test )
y_test = f(x_test)
