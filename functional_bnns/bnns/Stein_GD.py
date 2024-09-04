
import math
import torch
from torch import nn
from bnns.SSGE import BaseScoreEstimator as SSGE_backend
from bnns.utils import log_gaussian_pdf, get_std

from quality_of_life.my_torch_utils import get_flat_grads, set_flat_grads, nonredundant_copy_of_module_list


kernel_stuff = SSGE_backend().grad_gram
bandwidth_estimator = SSGE_backend().heuristic_sigma


#
# ~~~ Compute the mean-zero prior log gaussian density over the model weights
def log_prior_density(model):
    log_prior = 0.
    for p in model.parameters():
        log_prior += log_gaussian_pdf( where=p, mu=torch.zeros_like(p), sigma=get_std(p) )
    return log_prior

#
# ~~~ Take a list of model; flatten each of their parameters into a single vector; stack those vectors into a matrix
def flatten_parameters(list_of_models):
    return torch.stack([
            torch.cat([ p.view(-1) for p in model.parameters() ])
            for model in list_of_models
        ])  # ~~~ has shape (len(list_of_models),n_parameters_per_model)


loss_fn = nn.MSELoss()
class SteinEnsemble:
    #
    # ~~~ 
    def __init__( self, list_of_NNs, Optimizer, conditional_std, bw=None ):
        self.models = list_of_NNs    # ~~~ each "particle" is (the parameters of) a neural network
        self.conditional_std = conditional_std
        self.bw = bw
        self.optimizers = [ Optimizer(model.parameters()) for model in self.models ]
    #
    def kernel_stuff( self, list_of_models, other_list_of_models ):
        x = flatten_parameters(list_of_models)
        y = flatten_parameters(other_list_of_models)
        if self.bw is None:
            self.bw = bandwidth_estimator(x,y)
        K, dK = kernel_stuff(x,y,self.bw)
        return K, dK    # ~~~ K has shape (len(list_of_models),len(other_list_of_models)); dK has shape (len(list_of_models),len(other_list_of_models),n_parameters_per_model)
    #
    def train_step(self,X,y,stein=True):
        #
        # ~~~ Compute \grad \ln p(particle) for each particle (particles are NN's)
        for model in self.models:
            if stein:
                log_likelihood = log_gaussian_pdf( where=y, mu=model(X), sigma=self.conditional_std )
                log_prior = log_prior_density(model)
                un_normalized_log_posterior = log_likelihood + log_prior
                un_normalized_log_posterior.backward()
            else:
                loss = loss_fn(model(X),y) # -log_gaussian_pdf( where=y, mu=model(X), sigma=self.conditional_std )loss_fn( model(X), y )
                loss.backward()
        #
        # ~~~ Replace the gradients by \widehat{\phi}^*(particle) for each particle (particles are NN's)
        if stein:
            with torch.no_grad():
                log_posterior_grads = torch.stack([ get_flat_grads(model) for model in self.models ]) # ~~~ has shape (n_models,n_params_in_each_model)
                K, grads_of_K = self.kernel_stuff( self.models, self.models )
                # K, grads_of_K = torch.eye( len(self.models), device=K.device, dtype=K.dtype ), torch.zeros_like(grads_of_K)
                stein_grads = -( K@log_posterior_grads + grads_of_K.sum(axis=0) ) / len(self.models)        # ~~~ take the negative so that pytorch's optimizer's *maximize* the intended objective
                for i, model in enumerate(self.models):
                    set_flat_grads( model, stein_grads[i] )
        #
        # ~~~ Do the update
        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad()
        #
        # ~~~ Return for diagnostics
        if stein:
            return K, grads_of_K
    #
    # ~~~ forward for the full ensemble
    def __call__(self,x):
        return torch.stack([ model(x) for model in self.models ])


class SequentialSteinEnsemble(SteinEnsemble):
    def __init__( self, architecture, n_copies, device="cpu", *args, **kwargs ):
        self.device = device
        super().__init__(
                list_of_NNs = [
                    nonredundant_copy_of_module_list( architecture, sequential=True ).to(device)
                    for _ in range(n_copies)
                ],
                *args,
                **kwargs
            )

# class SteinEnsembleDebug:
#     #
#     # ~~~ 
#     def __init__( self, list_of_NNs, Optimizer, conditional_std, bw=None ):
#         self.models = list_of_NNs    # ~~~ each "particle" is (the parameters of) a neural network
#         self.conditional_std = conditional_std
#         self.bw = bw
#         self.optimizers = [ Optimizer(model.parameters()) for model in self.models ]
#     #
#     def kernel_stuff( self, list_of_models, other_list_of_models ):
#         x = flatten_parameters(list_of_models)
#         y = flatten_parameters(other_list_of_models)
#         if self.bw is None:
#             self.bw = bandwidth_estimator(x,y)
#         K, dK = kernel_stuff(x,y,self.bw)
#         return K, dK    # ~~~ K has shape (len(list_of_models),len(other_list_of_models)); dK has shape (len(list_of_models),len(other_list_of_models),n_parameters_per_model)
#     #
#     def train_step(self,X,y):
#         #
#         # ~~~ Compute \grad \ln p(particle) for each particle (particles are NN's)
#         for model in self.models:
#             log_likelihood = log_gaussian_pdf( where=y, mu=model(X), sigma=self.conditional_std )
#             log_prior = 0. #log_prior_density(model)
#             negative_un_normalized_log_posterior = -(log_likelihood + log_prior)
#             negative_un_normalized_log_posterior.backward()
#         #
#         # ~~~ Apply the affine transformation
#         with torch.no_grad():
#             log_posterior_grads = torch.stack([ get_flat_grads(model) for model in self.models ]) # ~~~ has shape (n_models,n_params_in_each_model)
#             K, grads_of_K = self.kernel_stuff( self.models, self.models )
#             # K, grads_of_K = torch.eye( len(self.models), device=K.device, dtype=K.dtype ), torch.zeros_like(grads_of_K)
#             stein_grads = ( K@log_posterior_grads + grads_of_K.sum(axis=0) ) / len(self.models)
#             for i, model in enumerate(self.models):
#                 set_flat_grads( model, stein_grads[i] )
#         #
#         # ~~~
#         for optimizer in self.optimizers:
#             optimizer.step()
#             optimizer.zero_grad()
#         # return K, grads_of_K
#     #
#     # ~~~ View the full ensemble
#     def __call__(self,x):
#         return torch.column_stack([ model(x) for model in self.models ])




# class SequentialSteinEnsembleDebug(SteinEnsembleDebug):
#     def __init__( self, architecture, n_copies, *args, **kwargs ):
#         some_device = "cuda" if torch.cuda.is_available() else "cpu"
#         super().__init__(
#             list_of_NNs = [
#                 nonredundant_copy_of_module_list( architecture, sequential=True ).to(some_device)
#                 for _ in range(n_copies)
#             ],
#             *args,
#             **kwargs
#         )