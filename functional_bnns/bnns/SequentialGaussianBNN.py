
import math
import torch
from torch import nn
from torch.func import jacrev, functional_call
from bnns.SSGE import SpectralSteinEstimator as SSGE
from bnns.SSGE import BaseScoreEstimator as SSGE_backend
from bnns.utils import log_gaussian_pdf, get_std, gaussian_kl, manual_Jacobian
from quality_of_life.my_torch_utils import nonredundant_copy_of_module_list
kernel_matrix = SSGE_backend().gram_matrix



### ~~~
## ~~~ Define a BNN with the necessary methods
### ~~~

#
# ~~~ Main class: intended to mimic nn.Sequential
class SequentialGaussianBNN(nn.Module):
    def __init__(self,*args):
        #
        # ~~~ Means and standard deviations for each network parameter
        super().__init__()
        self.model_mean = nn.Sequential(*nn.ModuleList(args))
        self.model_std  = nonredundant_copy_of_module_list(self.model_mean)
        self.n_layers   = len(self.model_mean)
        with torch.no_grad():
            #
            # ~~~ Define the prior means: first copy the architecture (maybe inefficient?), then set requires_grad=False and assign the desired mean values (==zero, for now)
            self.prior_mean = nonredundant_copy_of_module_list(self.model_mean)
            for p in self.prior_mean.parameters():
                p.requires_grad = False # ~~~ don't train the prior
                p = torch.zeros_like(p) # ~~~ assign the desired prior mean values
            #
            # ~~~ Define the prior std. dev.'s: first copy the architecture (maybe inefficient?), then set requires_grad=False and assign the desired std values
            self.prior_std = nonredundant_copy_of_module_list(self.model_mean)
            for p in self.prior_std.parameters():
                p.requires_grad = False # ~~~ don't train the prior
                p = p.fill_(get_std(p)) # ~~~ assign the desired prior standard deviation values
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
        self.prior_J   = "please specify"
        self.post_J    = "please specify"
        self.prior_eta = "please specify"
        self.post_eta  = "please specify"
        self.prior_M   = "please specify"
        self.post_M    = "please specify"
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
            prior_samples = torch.row_stack([ self.prior_forward(self.measurement_set).flatten() for _ in range(self.prior_M) ])
            try:
                self.prior_SSGE = SSGE( samples=prior_samples, eta=self.prior_eta, J=self.prior_J, h=self.use_eigh )
            except RuntimeError:
                self.use_eigh = False
                my_warn("Due to a bug in the pytorch source code, BNN.use_eigh has been set to False")
                self.prior_SSGE = SSGE( samples=prior_samples, eta=self.prior_eta, J=self.prior_J, h=self.use_eigh )
    #
    # ~~~ NEW
    def sample_new_measurement_set(self):
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
            posterior_samples = torch.row_stack([ self( self.measurement_set ).flatten() for _ in range(self.post_M) ])
            posterior_SSGE = SSGE( samples=posterior_samples, eta=self.post_eta, J=self.post_J, h=self.use_eigh )
        #
        # ~~~ By the chain rule, at these points we must (use SSGE to) compute the scores          
        yhat = self( self.measurement_set ).flatten()
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
    # ~~~ A callable GP prior
    def GP_prior(self,x):
        # if not hasattr(self,"K0inv"):
        #     bw = 0.1
        #     K_first_feature  = kernel_matrix( X[:,0].unsqueeze(-1), X[:,0].unsqueeze(-1) , bw ) + 0.0001*torch.ones_like(X[:,0]).diag()
        #     K_second_feature = kernel_matrix( X[:,1].unsqueeze(-1), X[:,1].unsqueeze(-1) , bw ) + 0.0001*torch.ones_like(X[:,1]).diag()
        #     self.K0inv = torch.block_diag( torch.linalg.inv(K_first_feature), torch.linalg.inv(K_second_feature) ) + 0.0001*torch.ones_like(X).diag()
        return torch.zeros_like(x.flatten()), torch.ones_like(x.flatten()).diag()
    #
    # ~~~ Compute the mean and standard deviation of a normal distribution approximating q_theta
    def simple_gaussian_approximation( self, resample_measurement_set=True ):
        #
        # ~~~ if `resample_measurement_set==True` then generate a new meausrement set
        if resample_measurement_set:
            self.sample_new_measurement_set()
        #
        # ~~~ First, compute the Jacobian at m of the model output with respect to the final layer's parameters (and treating the other layers' parameters as fixed)
        if not isinstance( self.model_mean[-1] , torch.nn.modules.linear.Linear ):
            raise NotImplementedError('Currently, the only case implemented is the the case from the paper where `\beta` is "the set of parameters in the final neural network layer" (bottom of pg. 4 of the paper).')
        #
        # ~~~ In this case, the Jacbian is easy to compute exactly: e.g., the Jacobian of A@whatever w.r.t. A is, simply `whatever`; let's first compute the `whatever` provided it is shaped correctly
        whatever = self.measurement_set
        for j in range(self.n_layers-1):            # ~~~ stop before feeding it into the final layer
            whatever = self.model_mean[j](whatever) # ~~~ BTW since the Jacobian is computed at the mean, it does not depend on self.realized_standard_normal
        J_beta = manual_Jacobian( whatever, number_of_output_features=self.model_mean[-1].out_features )    # ~~~ basically just shape it correctly
        #
        # ~~~ Deviate slightly from the paper by not actually computing J_alpha, and instead only approximating the requried sample
        self.sample_from_standard_normal()          # ~~~ essentially, resample network weights from the current distribution
        z = self.realized_standard_normal[-1]
        S_beta = self.rho(self.model_std[-1].weight).flatten()  # ~~~ the vector along the main diagonal of the covariance matrix, which would be the diagonal matrix `S_beta.diag()`
        Theta_beta_minus_mu_beta = S_beta * z.weight.flatten()  # ~~~ Theta_beta = mu_theta + Sigma_beta*z is sampled as Theta_sampled = mu_theta + Sigam_theta*z_sampled (a flat 1d vector)
        mu_theta = self( self.measurement_set, resample_weights=False ).flatten() - J_beta @ Theta_beta_minus_mu_beta   # ~~~ solving for the mean of the approximating normal distribution when using f on the LHS of the paper's equation (12)
        Sigma_theta = (S_beta*J_beta) @ (S_beta*J_beta).T
        return mu_theta, Sigma_theta
        # #
        # # ~~~ In this case, the Jacbian is easy to compute exactly: e.g., the Jacobian of A@whatever w.r.t. A is, simply `whatever`; let's first compute the `whatever`
        # J_beta = self.measurement_set
        # for j in range(self.n_layers-1):        # ~~~ stop before feeding it into the final layer
        #     J_beta = self.model_mean[j](J_beta) # ~~~ since the Jacobian is computed at the mean, it does not depend on self.realized_standard_normal
        # #
        # # ~~~ The Jacobian of A@whatever+b w.r.t. (A,b) is, simply `column_stack(whatever,1)`
        # if self.model_mean[-1].bias is not None:    # ~~~ only stack with 1's if there is a bias term
        #     J_beta = torch.column_stack([
        #             J_beta,
        #             torch.ones( J_beta.shape[0], 1, device=self.measurement_set.device, dtype=self.measurement_set.dtype )
        #         ])
        # S_diag = torch.column_stack([
        #         self.rho(self.model_std[-1].weight),
        #         self.rho(self.model_std[-1].bias)
        #     ]) if self.model_std[-1].bias else self.rho(self.model_std[-1].weight)
        # Theta_beta_minus_mu_beta = S_diag * torch.column_stack([z.weight,z.bias])   # ~~~ Theta_beta = mu+sigma*z is sampled as Theta_sampled = mu+sigma*z_sampled
        # mu_theta = self( self.measurement_set, resample_weights=False ) - J_beta @ Theta_beta_minus_mu_beta.T # ~~~ solving for the mean of the approximating normal distribution when using f on the LHS of the paper's equation (12)
        # Sigma_theta = J_beta @ S_diag.squeeze().diag() @ J_beta.T
        # return mu_theta, Sigma_theta
    #
    # ~~~ Compute the mean and standard deviation of a normal distribution approximating q_theta
    def gaussian_kl( self, mu_theta=None, Sigma_theta=None, resample_measurement_set=True, add_stabilizing_noise=False ):
        mu_0, Sigma_0_inv = self.GP_prior(self.measurement_set)
        if mu_theta is None and Sigma_theta is None:
            mu_theta, Sigma_theta = self.simple_gaussian_approximation( resample_measurement_set=resample_measurement_set )
        if add_stabilizing_noise:
            Sigma_theta += torch.diag( self.conditional_std*torch.ones_like(Sigma_theta.diag()) )
        return gaussian_kl( mu_theta, torch.linalg.cholesky(Sigma_theta), mu_0, torch.linalg.cholesky(Sigma_0_inv) )
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
