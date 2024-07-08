
import math
import torch
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain # ~~~ used (optionally) to define the prior distribution on network weights
from SSGE import SpectralSteinEstimator as SSGE



### ~~~
## ~~~ Define a BNN with the necessary methods
### ~~~

#
# ~~~ Helper function which creates a new instance of the supplied sequential architeture (ChatGPT wrote this one)
def nonredundant_copy_of_module_list(module_list):
    architecture = [ (type(layer),layer) for layer in module_list ]
    layers = []
    for layer_type, layer in architecture:
        if layer_type == torch.nn.Linear:
            #
            # ~~~ For linear layers, create a brand new linear layer of the same size independent of the original
            layers.append(torch.nn.Linear( layer.in_features, layer.out_features ))
        else:
            #
            # ~~~ For other layers (activations, Flatten, softmax, etc.) just copy it
            layers.append(layer)
    return nn.ModuleList(layers)

#
# ~~~ Helper function that just computes the log pdf of a multivariate normal distribution with independent coordinates
def log_gaussian_pdf( where, mu, sigma ):
    assert mu.shape==where.shape
    marginal_log_probs = -((where-mu)/sigma)**2/2 - torch.log( math.sqrt(2*torch.pi)*sigma )   # ~~~ note: isn't (x-mu)/sigma numerically unstable, like numerical differentiation?
    return marginal_log_probs.sum()

#
# ~~~ Main class: intended to mimic nn.Sequential
class SequentialGaussianBNN(nn.Module):
    def __init__(self,*args):
        #
        # ~~~ Means and standard deviations for each network parameter
        super().__init__()
        self.model_mean = nn.ModuleList(args)
        self.model_std  = nonredundant_copy_of_module_list(self.model_mean)
        self.n_layers   = len(self.model_mean)
        #
        # ~~~ Define the prior means: first copy the architecture (maybe inefficient?), then set requires_grad=False and assign the desired mean values (==zero, for now)
        self.prior_mean = nonredundant_copy_of_module_list(self.model_mean)
        for p in self.prior_mean.parameters():
            p.requires_grad = False
            p = torch.zeros_like(p) # ~~~ assign the desired prior mean values
        #
        # ~~~ Define the prior std. dev.'s: first copy the architecture (maybe inefficient?), then set requires_grad=False and assign the desired std values
        self.prior_std = nonredundant_copy_of_module_list(self.model_mean)
        for p in self.prior_std.parameters():
            p.requires_grad = False
            if len(p.shape)==1: # ~~~ for the bias vectors, simply take variance=1/length
                numb_pars = len(p)
                std = 1/math.sqrt(numb_pars)
            else:   # ~~~ for the weight matrices, mimic pytorch's `xavier normal` initialization (https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_normal_)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(p)
                gain = calculate_gain("relu")
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            p = p.fill_(std)
        #
        # ~~~ Define a "standard normal distribution in the shape of our neural network"
        self.realized_standard_normal = nonredundant_copy_of_module_list(self.model_mean)
        for p in self.realized_standard_normal.parameters():
            p.requires_grad = False
            nn.init.normal_(p)
        #
        # ~~~ Define a reparameterization (-Inf,Inf) -> (0,Inf)
        self.rho = lambda sigma: torch.log(1+torch.exp(sigma))
        #
        # ~~~ Define the assumed level of noise in the training data: when this is set to smaller values, the model "pays more attention" to the data, and fits it more aggresively (can also be a vector)
        self.conditional_std = torch.tensor(0.001)
        #
        # ~~~ NEW attributes for SSGE and functional training
        self.J_prior   = 10
        self.J_post    = 10
        self.eta_prior = 10
        self.eta_post  = 10
        self.M_prior   = 50
        self.M_post    = 50
        self.measurement_set = None
        self.prior_SSGE      = None
        self.use_eigh        = True
    #
    # ~~~ Sample according to a "standard normal distribution in the shape of our neural network"
    def sample_from_standard_normal(self):
        for p in self.realized_standard_normal.parameters():
            nn.init.normal_(p)
    #
    # ~~~ Sample the distribution of Y|X=x,W=w
    def forward(self,x,resample_weights=True):
        #
        # ~~~ The realized sample of the distribution of Y|X=x,W=w is entirely determined by self.realized_standard_normal
        if resample_weights:
            self.sample_from_standard_normal()      # ~~~ this methods re-generates the values of weights and biases in `self.realized_standard_normal` (IID standard normal)
        #
        # ~~~ Basically, `x=layer(x)` for each layer in model, but with a twist on the weights
        for j in range(self.n_layers):
            z = self.realized_standard_normal[j]    # ~~~ the network's j'th layer, but with IID standard normal weights and biases
            #
            # ~~~ If this layer is just like relu or something, then there aren't any weights; just apply the layer and be done
            if not isinstance( z, torch.nn.modules.linear.Linear ):
                x = z(x)                            # ~~~ x = layer(x)
            #
            # ~~~ Aforementioned twist is that we apply F_\theta to the weights before doing x = layer(x)
            else:
                mean_layer = self.model_mean[j]     # ~~~ the trainable (posterior) means of this layer's parameters
                std_layer  =  self.model_std[j]     # ~~~ the trainable (posterior) standard deviations of this layer's parameters
                A = mean_layer.weight + self.rho(std_layer.weight) * z.weight   # ~~~ A = F_\theta(z.weight) is normal with the trainable (posterior) mean and std
                b = mean_layer.bias   +   self.rho(std_layer.bias) * z.bias     # ~~~ b = F_\theta(z.bias)   is normal with the trainable (posterior) mean and std
                x = x@A.T + b                                                   # ~~~ apply the appropriately distributed weights to this layer's input
        return x
    #
    # ~~~ What the forward pass would be if we distributed weights according to the prior distribution
    def prior_forward(self,x,resample_weights=True):
        #
        # ~~~ The realized sample of the distribution of Y|X=x,W=w is entirely determined by self.realized_standard_normal
        if resample_weights:
            self.sample_from_standard_normal()      # ~~~ this methods re-generates the values of weights and biases in `self.realized_standard_normal` (IID standard normal)
        #
        # ~~~ Basically, `x=layer(x)` for each layer in model, but with a twist on the weights
        for j in range(self.n_layers):
            z = self.realized_standard_normal[j]    # ~~~ the network's j'th layer, but with IID standard normal weights and biases
            #
            # ~~~ If this layer is just like relu or something, then there aren't anny weights; just apply the layer and be done
            if not isinstance( z, nn.modules.linear.Linear ):
                x = z(x)                            # ~~~ x = layer(x)
            #
            # ~~~ Aforementioned twist is that we apply F_\theta to the weights before doing x = layer(x)
            else:
                mean_layer = self.prior_mean[j]     # ~~~ the user-specified prior means of this layer's parameters
                std_layer  =  self.prior_std[j]     # ~~~ the user-specified prior standard deviations of this layer's parameters
                A = mean_layer.weight + std_layer.weight * z.weight # ~~~ A = F_\theta(z.weight) is normal with the user-specified prior mean and std
                b = mean_layer.bias   +   std_layer.bias * z.bias   # ~~~ b = F_\theta(z.bias)   is normal with the user-specified prior mean and std
                x = x@A.T + b                       # ~~~ apply the appropriately distributed weights to this layer's input
        return x
    #
    # ~~~ Compute ln( f_{Y \mid X,W}(F_\theta(z),x_train,y_train) ) at a point z sampled from the standard MVN distribution ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def log_likelihood_density( self, x_train, y_train ):
        return log_gaussian_pdf( where=y_train, mu=self(x_train,resample_weights=False), sigma=self.conditional_std )  # ~~~ Y|X,W is assumed to be normal with mean self(X) and variance self.conditional_std (the latter being a tunable hyper-parameter)
    #
    # ~~~ Compute \ln( f_W(F_\theta(z)) ) at a point w sampled from the standard MVN distribution, where f_W is the prior PDF of the network parameters ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def log_prior_density(self):
        #
        # ~~~ Because the weights and biases are mutually independent, the log prior pdf can be decomposed as a summation \sum_j
        log_prior = 0.
        for j in range(self.n_layers):
            z = self.realized_standard_normal[j]        # ~~~ the network's j'th layer, but with IID standard normal weights and biases
            if isinstance( z, torch.nn.modules.linear.Linear ):
                post_mean      =    self.model_mean[j]  # ~~~ the trainable (posterior) means of this layer's parameters
                post_std       =    self.model_std[j]   # ~~~ the trainable (posterior) standard deviations of this layer's parameters
                prior_mean     =    self.prior_mean[j]  # ~~~ the prior means of this layer's parameters
                prior_std      =    self.prior_std[j]   # ~~~ the prior standard deviations of this layer's parameters
                F_theta_of_z   =    post_mean.weight + self.rho(post_std.weight)*z.weight
                log_prior     +=    log_gaussian_pdf( where=F_theta_of_z, mu=prior_mean.weight, sigma=prior_std.weight )
                F_theta_of_z   =    post_mean.bias   +  self.rho(post_std.bias) * z.bias
                log_prior     +=    log_gaussian_pdf( where=F_theta_of_z,  mu=prior_mean.bias,  sigma=prior_std.bias   )
                # F_theta_of_z   =    prior_mean.weight + prior_std.weight*z.weight
                # log_prior     +=    log_gaussian_pdf( where=F_theta_of_z, mu=prior_mean.weight, sigma=prior_std.weight )
                # F_theta_of_z   =    prior_mean.bias   +  prior_std.bias * z.bias
                # log_prior     +=    log_gaussian_pdf( where=F_theta_of_z,  mu=prior_mean.bias,  sigma=prior_std.bias   )
        return log_prior
    #
    # ~~~ Compute \ln( q_\theta(F_\theta(z)) ) at a point z sampled from the standard MVN distribution, where q_\theta is the posterior PDF of the network parameters ( F_\theta(z)=\mu+\sigma*z are the appropriately distributed network weights; \theta=(\mu,\sigma) )
    def log_posterior_density(self):
        #
        # ~~~ Because the weights and biases are mutually independent, the log_prior_pdf can be decomposed as a summation \sum_j
        log_posterior = 0.
        for j in range(self.n_layers):
            z = self.realized_standard_normal[j]        # ~~~ the network's j'th layer, but with IID standard normal weights and biases
            if isinstance( z, torch.nn.modules.linear.Linear ):
                mean_layer      =    self.model_mean[j] # ~~~ the trainable (posterior) means of this layer's parameters
                std_layer       =    self.model_std[j]  # ~~~ the trainable (posterior) standard deviations of this layer's parameters
                sigma_weight    =    self.rho(std_layer.weight)
                sigma_bias      =    self.rho(std_layer.bias)
                F_theta_of_z    =    mean_layer.weight + sigma_weight*z.weight
                log_posterior  +=    log_gaussian_pdf( where=F_theta_of_z, mu=mean_layer.weight, sigma=sigma_weight )
                F_theta_of_z    =    mean_layer.bias   +  sigma_bias * z.bias
                log_posterior  +=    log_gaussian_pdf( where=F_theta_of_z,  mu=mean_layer.bias,   sigma=sigma_bias  )
        return log_posterior
    #
    # ~~~ NEW
    def setup_prior_SSGE(self):
        with torch.no_grad():
            prior_samples = torch.column_stack([ self.prior_forward( self.measurement_set ) for _ in range(self.M_prior) ]).T
            try:
                self.prior_SSGE = SSGE( samples=prior_samples, eta=self.eta_prior, J=self.J_prior, h=self.use_eigh )
            except RuntimeError:
                self.use_eigh = False
                my_warn("Due to a bug in the pytorch source code, BNN.use_eigh has been set to False")
                self.prior_SSGE = SSGE( samples=prior_samples, eta=self.eta_prior, J=self.J_prior, h=self.use_eigh )
    #
    # ~~~ NEW
    def sample_measurement_set(self):
        raise NotImplementedError("In order to resample a measurement set, you must assign `self.sample_measurement_set=my_method` where `my_method` accepts the argument `self` and assigns a value to `self.meaurementset=___`.")
        # self.measurement_set = ???
    #
    # ~~~ NEW
    def functional_kl( self, resample_measurement_set=True ):
        #
        # ~~~ if `resample_measurement_set==True` then generate a new meausrement set
        if resample_measurement_set:
            self.sample_new_measurement_set()
        #
        # ~~~ Prepare for using SSGE to estimate some of the gradient terms
        with torch.no_grad():
            if self.prior_SSGE is None:
                self.setup_prior_SSGE()
            posterior_samples = torch.column_stack([ self( self.measurement_set ) for _ in range(self.M_post) ]).T
            posterior_SSGE = SSGE( samples=posterior_samples, eta=self.eta_prior, J=self.J_prior, h=self.use_eigh )
        #
        # ~~~ By the chain rule, at these points we must (use SSGE to) compute the scores          
        yhat = self( self.measurement_set )
        #
        # ~~~ Use SSGE to compute "the intractible parts of the chain rule"
        with torch.no_grad():
            posterior_score_at_yhat =  posterior_SSGE( yhat.reshape(1,-1) )
            prior_score_at_yhat     = self.prior_SSGE( yhat.reshape(1,-1) )
        #
        # ~~~ Combine all the ingridents as per the chain rule 
        log_posterior_density  =  ( posterior_score_at_yhat @ yhat).squeeze() # ~~~ the inner product from the chain rule
        log_prior_density      =  (prior_score_at_yhat @ yhat).squeeze()      # ~~~ the inner product from the chain rule            
        return log_posterior_density, log_prior_density
    #
    # ~~~ A helper function that samples a bunch from the predicted posterior distribution
    def posterior_predicted_mean_and_std( self, x_test, n_samples ):
        with torch.no_grad():
            predictions = torch.column_stack([ self(x_test) for _ in range(n_samples) ])
            std = predictions.std(dim=-1).cpu()             # ~~~ transfer to cpu in order to be able to plot them
            point_estimate = predictions.mean(dim=-1).cpu() # ~~~ transfer to cpu in order to be able to plot them
        return point_estimate, std


#
# ~~~ This was the first thing I tried, but it resulted in memory overflow, which is why I opted to not even both with torch.distributions
# from torch.distributions.multivariate_normal import MultivariateNormal
# priors = []
# for p in BNN.parameters():
#     print("go")
#     if len(p.shape)==1: # ~~~ for the biases
#         numb_pars = len(p)
#         std = 1/math.sqrt(numb_pars)
#     else:               # ~~~ for the weight matrices, pytorch's `xavier normal` initialization (https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_normal_)
#         fan_in, fan_out = _calculate_fan_in_and_fan_out(p)
#         gain = calculate_gain("relu")
#         std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
#         numb_pars = math.prod(p.shape)
#     priors.append( MultivariateNormal(
#             mean=torch.zeros(numb_pars),
#             covariance_matrix = std*torch.eye(numb_pars)
#         ) )
