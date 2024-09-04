'''
Code in this file (bnn.py) is based on TyXe: https://github.com/TyXe-BDL/TyXe/blob/master/tyxe/bnn.py

Original code license:
MIT License

Copyright (c) 2021 Hippolyt Ritter, Theofanis Karaletsos

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Multiverse repository license:
BSD 3-Clause License

Copyright (c) 2023, Los Alamos National Laboratory

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

from collections import defaultdict
import itertools
from operator import itemgetter
from functools import partial

import torch
import numpy as np

import pyro
import pyro.nn as pynn
import pyro.infer.autoguide as autoguide
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC

from tqdm import tqdm

from .util import *

__all__ = ["PytorchBNN", "VariationalBNN", "LaplaceBNN", "MCMC_BNN"]


def _empty_guide(*args, **kwargs):
    return {}


def _as_tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return x,


def _to(x, device):
    return map(lambda t: t.to(device) if device is not None else t, _as_tuple(x))


class _BNN(pynn.PyroModule):
    """BNN base class that takes an nn.Module, turns it into a PyroModule and applies a prior to it, i.e. replaces
    nn.Parameter attributes by PyroSamples according to the specification in the prior. The forward method wraps the
    forward pass of the net and samples weights from the prior distributions.

    :param nn.Module net: pytorch neural network to be turned into a BNN.
    :param prior tyxe.priors.Prior: prior object that specifies over which parameters we want uncertainty.
    :param str name: base name for the BNN PyroModule."""

    def __init__(self, net, prior, name=""):
        super().__init__(name)
        self.torch_net = net
        self.device = net.device
        self.net = to_pyro_module(net)
        self.prior = prior
        self.prior.apply_(self.net)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def update_prior(self, new_prior):
        """Uppdates the prior of the network, i.e. calls its update_ method on the net.

        :param tyxe.priors.Prior new_prior: Prior for replacing the previous prior, i.e. substituting the PyroSample
            attributes of the net."""
        self.prior = new_prior
        self.prior.update_(self)


class GuidedBNN(_BNN):
    """Guided BNN class that in addition to the network and prior also has a guide for doing approximate inference
    over the neural network weights. The guide_builder argument is called on the net after it has been transformed to
    a PyroModule and returns the pyro guide function that sample from the approximate posterior.

    :param callable guide_builder: callable that takes a probabilistic pyro function with sample statements and returns
        an object that helps with inference, i.e. a callable guide function that samples from an approximate posterior
        for variational BNNs or an MCMC kernel for MCMC-based BNNs. May be None for maximum likelihood inference if
        the prior leaves all parameters of the net as such."""

    def __init__(self, net, prior, guide_builder=None, name=""):
        super().__init__(net, prior, name=name)
        self.net_guide = guide_builder(self.net) if guide_builder is not None else _empty_guide

    def guided_forward(self, *args, guide_tr=None, **kwargs):
        if guide_tr is None:
            guide_tr = poutine.trace(self.net_guide).get_trace(*args, **kwargs)
        return poutine.replay(self.net, trace=guide_tr)(*args, **kwargs)


class PytorchBNN(GuidedBNN):
    """Low-level variational BNN class that can serve as a drop-in replacement for an nn.Module.

    :param bool closed_form_kl: whether to use TraceMeanField_ELBO or Trace_ELBO as the loss, i.e. calculate KL
        divergences in closed form or via a Monte Carlo approximate of the difference of log densities between
        variational posterior and prior."""

    def __init__(self, net, prior, guide_builder=None, name="", closed_form_kl=True):
        super().__init__(net, prior, guide_builder=guide_builder, name=name)
        self.cached_output = None
        self.cached_kl_loss = None
        self._loss = TraceMeanField_ELBO() if closed_form_kl else Trace_ELBO()

    def named_pytorch_parameters(self, *input_data):
        """Equivalent of the named_parameters method of an nn.Module. Ensures that prior and guide are run once to
        initialize all pyro parameters. Those are then collected and returned via the trace poutine."""
        model_trace = poutine.trace(self.net, param_only=True).get_trace(*input_data)
        guide_trace = poutine.trace(self.net_guide, param_only=True).get_trace(*input_data)
        for name, msg in itertools.chain(model_trace.nodes.items(), guide_trace.nodes.items()):
            yield name, msg["value"].unconstrained()

    def pytorch_parameters(self, input_data_or_fwd_fn):
        yield from map(itemgetter(1), self.named_pytorch_parameters(input_data_or_fwd_fn))

    def cached_forward(self, *args, **kwargs):
        # cache the output of forward to make it effectful, so that we can access the output when running forward with
        # posterior rather than prior samples
        self.cached_output = super().forward(*args, **kwargs)
        return self.cached_output

    def forward(self, *args, **kwargs):
        self.cached_kl_loss = self._loss.differentiable_loss(self.cached_forward, self.net_guide, *args, **kwargs)
        return self.cached_output


class _SupervisedBNN(GuidedBNN):
    """Base class for supervised BNNs that defines the interface of the predict method and implements
    evaluate. Agnostic to the kind of inference performed.

    :param tyxe.likelihoods.Likelihood likelihood: Likelihood object that implements a forward method including
        a pyro.sample statement for labelled data given neural network predictions and implements logic for aggregating
        multiple predictions and evaluating them."""

    def __init__(self, net, prior, likelihood, net_guide_builder=None, name=""):
        super().__init__(net, prior, net_guide_builder, name=name)
        self.likelihood = likelihood

    def model(self, x, obs=None):
        predictions = self(*_as_tuple(x))
        self.likelihood(predictions, obs)
        return predictions

    def evaluate(self, input_data, y, num_predictions=1, reduction="sum"):
        """"Utility method for evaluation. Calculates a likelihood-dependent errors measure, e.g. squared errors or
        mis-classifications and

        :param input_data: Inputs to the neural net. Must be a tuple of more than one.
        :param y: observations, e.g. class labels.
        :param int num_predictions: number of forward passes.
        :param bool aggregate: whether to aggregate the outputs of the forward passes before evaluating.
        :param str reduction: "sum", "mean" or "none". How to process the tensor of errors. "sum" adds them up,
            "mean" averages them and "none" simply returns the tensor."""
        predictions = self.predict(*_as_tuple(input_data), num_predictions=num_predictions)
        error = self.likelihood.error(predictions, y, reduction=reduction)
        ll = self.likelihood.log_likelihood(predictions, y, reduction=reduction)
        return error, ll

    def predict(self, *input_data, num_predictions=1, aggregate=False):
        """Makes predictions on the input data

        :param input_data: inputs to the neural net, e.g. torch.Tensors
        :param int num_predictions: number of forward passes through the net
        :param bool aggregate: whether to aggregate the predictions depending on the likelihood, e.g. averaging them."""
        raise NotImplementedError


class VariationalBNN(_SupervisedBNN):
    """Variational BNN class for supervised problems. Requires a likelihood that describes the data noise and an
    optional guide builder for it should it contain any variables that need to be inferred. Provides high-level utility
    method such as fit, predict and

    :param callable net_guide_builder: pyro.infer.autoguide.AutoCallable style class that given a pyro function
        constructs a variational posterior that sample the same unobserved sites from distributions with learnable
        parameters.
    :param callable likelihood_guide_builder: optional callable that constructs a guide for the likelihood if it
        contains any unknown variable, such as the precision/scale of a Gaussian."""
    def __init__(self, net, prior, likelihood, net_guide_builder=None, likelihood_guide_builder=None, name=""):
        super().__init__(net, prior, likelihood, net_guide_builder, name=name)
        weight_sample_sites = list(pyro_sample_sites(self.net))
        if likelihood_guide_builder is not None:
            self.likelihood_guide = likelihood_guide_builder(poutine.block(
                self.model, hide=weight_sample_sites + [self.likelihood.data_name]))
        else:
            self.likelihood_guide = _empty_guide

    def guide(self, x, obs=None):
        result = self.net_guide(*_as_tuple(x)) or {}
        result.update(self.likelihood_guide(*_as_tuple(x), obs) or {})
        return result

    def fit(self, data_loader, optim, num_epochs, callback=None, num_particles=1, closed_form_kl=True, device=None,
            hist=False, test_loader=None, test_eval_interval=10):
        """Optimizes the variational parameters on data from data_loader using optim for num_epochs.

        :param Iterable data_loader: iterable over batches of data, e.g. a torch.utils.data.DataLoader. Assumes that
            each element consists of a length two tuple of list, with the first element either containing a single
            object or a list of objects, e.g. torch.Tensors, that are the inputs to the neural network. The second
            element is a single torch.Tensor e.g. of class labels.
        :param optim: pyro optimizer to be used for constructing an SVI object, e.g. pyro.optim.Adam({"lr": 1e-3}).
        :param int num_epochs: number of passes over data_loader.
        :param callable callback: optional function to invoke after every training epoch. Receives the BNN object,
            the epoch number and the average value of the ELBO over the epoch. May return True to terminate
            optimization before num_epochs, e.g. if it finds that a validation log likelihood saturates.
        :param int num_particles: number of MC samples for estimating the ELBO.
        :param bool closed_form_kl: whether to use TraceMeanField_ELBO or Trace_ELBO, i.e. calculate KL divergence
            between approximate posterior and prior in closed form or via a Monte Carlo estimate.
        :param torch.device device: optional device to send the data to.
        """
        if hist:
            nll_hist = torch.zeros(num_epochs)
            if test_loader is not None:
                nll_hist_test = torch.zeros(num_epochs//test_eval_interval)

        loss = TraceMeanField_ELBO(num_particles) if closed_form_kl else Trace_ELBO(num_particles)
        svi = SVI(self.model, self.guide, optim, loss=loss)

        for i in tqdm(range(num_epochs)):
            elbo = 0.
            num_batch = 1
            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                svi_step = svi.step(tuple(_to(input_data, device)), tuple(_to(observation_data, device))[0])
                elbo += svi_step
                if hist:
                    guide_tr = pyro.poutine.trace(self.net_guide).get_trace(input_data.to(self.device))
                    batch_nll = -self.likelihood.log_likelihood(pyro.poutine.replay(self.net, guide_tr)(input_data.to(self.device)), 
                                                                observation_data.to(self.device), reduction="mean")
                    nll_hist[i] += batch_nll
            if hist:
                nll_hist[i] = nll_hist[i] / num_batch

            if test_loader is not None and hist and i % test_eval_interval == 0:
                for num_batch, (input_data, observation_data) in enumerate(iter(test_loader), 1):
                    guide_tr = pyro.poutine.trace(self.net_guide).get_trace(input_data.to(self.device))
                    test_batch_nll = -self.likelihood.log_likelihood(pyro.poutine.replay(self.net, guide_tr)(input_data.to(self.device)), 
                                                                     observation_data.to(self.device), reduction="mean")
                    nll_hist_test[i] += test_batch_nll
                nll_hist_test[i] = nll_hist_test[i] / num_batch

            # the callback can stop training by returning True
            if callback is not None and callback(self, i, elbo / num_batch):
                break

        if hist:
            return nll_hist
        return svi

    def predict(self, *input_data, num_predictions=1, aggregate=False, guide_traces=None):
        if guide_traces is None:
            guide_traces = [None] * num_predictions

        preds = []
        with torch.autograd.no_grad():
            for trace in guide_traces:
                preds.append(self.guided_forward(*input_data, guide_tr=trace))
        predictions = torch.stack(preds)
        return self.likelihood.aggregate_predictions(predictions) if aggregate else predictions


class LaplaceBNN(_SupervisedBNN):
    """Laplace-GGN approximation to BNN posterior for supervised problems. Requires a likelihood that describes the data noise and an
    optional guide builder for it should it contain any variables that need to be inferred. Provides high-level utility
    method such as fit, predict, energy and log_marginal_likelihood.

    :param callable likelihood_guide_builder: optional callable that constructs a guide for the likelihood if it
        contains any unknown variable, such as the precision/scale of a Gaussian.
    :param str approximation: "full", "diag" or "subnet". Whether to use the full Hessian, a diagonal approximation
        or a subset of the parameters with the largest diagonal entries of the Hessian.
    :param float S_perc: percentage of parameters to use for the subnet approximation.
    """   
    
    def __init__(self, net, prior, likelihood, likelihood_guide_builder=None, name="", approximation="full", S_perc=1.):
        super().__init__(net, prior, likelihood, net_guide_builder=autoguide.AutoLaplaceApproximation, name=name)
        self.approximation = approximation
        self.S_perc = torch.tensor(S_perc, device=self.device)
        weight_sample_sites = list(pyro_sample_sites(self.net))
        if likelihood_guide_builder is not None:
            self.likelihood_guide = likelihood_guide_builder(poutine.block(
                self.model, hide=weight_sample_sites + [self.likelihood.data_name]))
        else:
            self.likelihood_guide = _empty_guide

    def guide(self, x, obs=None):
        result = self.net_guide(*_as_tuple(x)) or {}
        result.update(self.likelihood_guide(*_as_tuple(x), obs) or {})
        return result

    def fit(self, data_loader, optim, num_epochs, callback=None, num_particles=1, closed_form_kl=True, hist=False,
            test_loader=None, test_eval_interval=10):
        """Optimizes the parameters to the MAP on data from data_loader using optim for num_epochs. Then calculates a
        Generalized Gauss-Newton (GGN) approx. to the Hessian of the log joint at the MAP and uses it to construct a 
        Gaussian approximation to the posterior.

        :param Iterable data_loader: iterable over batches of data, e.g. a torch.utils.data.DataLoader. Assumes that
            each element consists of a length two tuple of list, with the first element either containing a single
            object or a list of objects, e.g. torch.Tensors, that are the inputs to the neural network. The second
            element is a single torch.Tensor e.g. of class labels.
        :param optim: pyro optimizer to be used for constructing an SVI object, e.g. pyro.optim.Adam({"lr": 1e-3}).
        :param int num_epochs: number of passes over data_loader.
        :param callable callback: optional function to invoke after every training epoch. Receives the BNN object,
            the epoch number and the average value of the ELBO over the epoch. May return True to terminate
            optimization before num_epochs, e.g. if it finds that a validation log likelihood saturates.
        :param int num_particles: number of MC samples for estimating the ELBO.
        :param bool closed_form_kl: whether to use TraceMeanField_ELBO or Trace_ELBO, i.e. calculate KL divergence
            between approximate posterior and prior in closed form or via a Monte Carlo estimate.
        :param bool hist: whether to return a history of the ELBO values over the epochs.
        """
        
        def full_GGN(self, J, Lambda, jitter=0):
            C = torch.linalg.cholesky(Lambda)
            if self.net.output_size == 1:
                JtC = J.T.double() @ C
            else:
                JtC = J.transpose(1,2).double().matmul(C).transpose(0,1).reshape(num_par,-1)
            GGN = P.double() + JtC.matmul(JtC.T)
            return GGN
        
        def diag_GGN(self, J, Lambda):
            C = torch.linalg.cholesky(Lambda)
            if self.net.output_size == 1:
                JtC = J.T.double() @ C
            else:
                JtC = J.transpose(1,2).double().matmul(C).transpose(0,1).reshape(num_par,-1)
            GGN = P.double().squeeze() + (JtC**2).sum(axis=1).squeeze()

            return GGN
        
        if hist:
            nll_hist = torch.zeros(num_epochs)
            if test_loader is not None:
                nll_hist_test = torch.zeros(num_epochs//test_eval_interval)
        
        loss = TraceMeanField_ELBO(num_particles) if closed_form_kl else Trace_ELBO(num_particles)
        svi = SVI(self.model, self.guide, optim, loss=loss)

        for i in tqdm(range(num_epochs)):
            elbo = 0.
            num_batch = 1
            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                svi_step = svi.step(tuple(_to(input_data, self.device)), tuple(_to(observation_data, self.device))[0])
                elbo += svi_step
                if hist:
                    guide_tr = pyro.poutine.trace(self.net_guide).get_trace(input_data.to(self.device))
                    batch_nll = -self.likelihood.log_likelihood(pyro.poutine.replay(self.net, guide_tr)(input_data.to(self.device)), 
                                                                observation_data.to(self.device), reduction="mean")
                    nll_hist[i] += batch_nll
            if hist:
                nll_hist[i] = nll_hist[i] / num_batch

            if test_loader is not None and hist and i % test_eval_interval == 0:
                for num_batch, (input_data, observation_data) in enumerate(iter(test_loader), 1):
                    guide_tr = pyro.poutine.trace(self.net_guide).get_trace(input_data.to(self.device))
                    test_batch_nll = -self.likelihood.log_likelihood(pyro.poutine.replay(self.net, guide_tr)(input_data.to(self.device)), 
                                                                     observation_data.to(self.device), reduction="mean")
                    nll_hist_test[i] += test_batch_nll
                nll_hist_test[i] = nll_hist_test[i] / num_batch

            # the callback can stop training by returning True
            if callback is not None and callback(self, i, elbo / num_batch):
                break

        input_data = data_loader.dataset.tensors[0].to(self.device)
        
        guide_tr = pyro.poutine.trace(self.net_guide).get_trace(input_data)
        net_tr = pyro.poutine.trace(pyro.poutine.replay(self.net, guide_tr)).get_trace(input_data)

        self.map_params = [guide_tr.nodes[name]["value"].unconstrained() for name in guide_tr.param_nodes][0]
        num_par = self.map_params.shape[0]

        if isinstance(self.prior._distribution, torch.distributions.Normal):
            prior_scale = self.prior._distribution.scale
            P = (1/self.S_perc)*(prior_scale**(-2)) * torch.eye(num_par, device = self.device)
        else:
            P = - (1/self.S_perc)*hessian(net_tr.log_prob_sum(), self.net_guide.loc).to(self.device)

        torch.nn.utils.vector_to_parameters(self.map_params, self.torch_net.parameters())
        self.jacobian = partial(backpack_jacobian, self.torch_net)
        J,f = self.jacobian(input_data)
        Lambda = self.likelihood.information_matrix(f).double().to(self.device)
        
        if self.approximation == "full":
            GGN = full_GGN(self, J, Lambda)
            self.GGN = GGN.float()
        else:
            P = torch.diag(P)
            GGN = diag_GGN(self, J, Lambda)
            if self.approximation == "subnet":
                self.top_S_perc = GGN.sort().indices[:int(self.S_perc*num_par)]
                num_par = len(self.top_S_perc)
                P = torch.diag(P[self.top_S_perc])
                GGN = full_GGN(self, J[...,self.top_S_perc], Lambda)
                self.GGN = GGN.float()
            else:
                self.GGN = torch.diag(GGN.float())
        
        # TODO: KFAC approx.

        # Initalize a new multivariate normal guide with the new covariance matrix:
        with torch.no_grad():

            if self.approximation == "subnet":
                loc = self.net_guide.loc[self.top_S_perc]
            else:
                loc = self.net_guide.loc

            if self.approximation == "diag":
                scale_tril = torch.diag(torch.sqrt(1/GGN))  
            else:
                cov = GGN.inverse() # possibly ad-hoc efficient method 
                cov = (cov + cov.T) / 2 # ensure symmetry
                try:
                    scale_tril = torch.linalg.cholesky(cov)
                except:     
                    print("Covariance matrix of the Laplace approx. not PSD - trying to add jitter to the diagonal")
                    GGN = GGN + 1e-2*torch.diag(GGN.diagonal())
                    chol = torch.linalg.cholesky(GGN)
                    cov = torch.cholesky_inverse(chol) # possibly ad-hoc efficient method 
                    cov = (cov + cov.T) / 2 # ensure symmetry
                    scale_tril = torch.linalg.cholesky(cov)
        self.gaussian_approx = torch.distributions.MultivariateNormal(loc,scale_tril=scale_tril.float())

        if hist:
            if test_loader is not None:
                return nll_hist, nll_hist_test
            return nll_hist
    

    def predict(self, input_data, num_predictions=1, aggregate=False):
        """ Samples from the linearized Laplace predictive distribution on the output of the neural network.
        :param input_data: inputs to the neural net, e.g. torch.Tensors.
        :param int num_predictions: number of samples from the predictive distribution.
        :param bool aggregate: whether to aggregate the predictions depending on the likelihood, e.g. averaging them.
        """
        input_data = input_data.to(self.device)
        J_map, f_map = self.jacobian(input_data)
        theta_samples = self.gaussian_approx.sample((num_predictions,))
        
        if self.approximation == "subnet":
            J_map = J_map[...,self.top_S_perc]
            map_params = self.map_params[...,self.top_S_perc]
        else:
            map_params = self.map_params
        if self.net.output_size == 1:
            predictions = f_map + (theta_samples - map_params) @ J_map.T
        else:
            predictions = (f_map.T + (J_map @ (theta_samples - map_params).T).T).mT
        return self.likelihood.aggregate_predictions(predictions) if aggregate else predictions
    
    def energy(self, data_loader):
        """ Computes the negative log joint probability of the data under the MAP. """
        x, y = data_loader.dataset.tensors
        x = x.to(self.device)
        y = y.to(self.device)
        f_map = self.torch_net(x)
        log_lik = self.likelihood.log_likelihood(f_map, y, reduction="sum")
        guide_tr = pyro.poutine.trace(self.net_guide).get_trace(x)
        net_tr = pyro.poutine.trace(pyro.poutine.replay(self.net, guide_tr)).get_trace(x)
        log_prior = net_tr.log_prob_sum()
        log_joint = log_lik + log_prior
        return - log_joint
    
    def log_marginal_likelihood(self, data_loader):
        """Computes the marginal likelihood of the data under the Laplace-GGN approximation to the BNN posterior."""
        log_joint = - self.energy(data_loader)
        log_det_H = torch.logdet((1/(2*np.pi)) * self.GGN)
        # if nan try numpy
        if torch.isnan(log_det_H):
            npGGN = self.GGN.cpu().detach().numpy()
            log_det_H = torch.tensor(np.linalg.slogdet((1/(2*np.pi)) * npGGN)).prod().to(self.device)
        log_ml = log_joint - 0.5 * log_det_H
        return log_ml


class MCMC_BNN(_BNN):
    """Supervised BNN class with an interface to pyro's MCMC that is unified with the VariationalBNN class.
        Currently not supporting multiple chains.

    :param callable kernel_builder: function or class that returns an object that will accepted as kernel by
        pyro.infer.mcmc.MCMC, e.g. pyro.infer.mcmc.HMC or NUTS. Will be called with the entire model, i.e. also
        infer variables in the likelihood."""

    def __init__(self, net, prior, likelihood, kernel_builder, name=""):
        super().__init__(net, prior, name=name)
        self.likelihood = likelihood
        self.kernel = kernel_builder(self.model)
        self._mcmc = None

    def model(self, x, obs=None):
        predictions = self(*_as_tuple(x))
        self.likelihood(predictions, obs)
        return predictions

    def fit(self, data_loader, num_samples, device=None, batch_data=False, **mcmc_kwargs):
        """Runs MCMC on the data from data loader using the kernel that was used to instantiate the class.

        :param data_loader: iterable or list of batched inputs to the net. If iterable treated like the data_loader
            of VariationalBNN and all network inputs are concatenated via torch.cat. Otherwise must be a tuple of
            a single or list of network inputs and a tensor for the targets.
        :param int num_samples: number of MCMC samples to draw.
        :param torch.device device: optional device to send the data to.
        :param batch_data: whether to treat data_loader as a full batch of data or an iterable over mini-batches.
        :param dict mcmc_kwargs: keyword arguments for initializing the pyro.infer.mcmc.MCMC object."""
        if batch_data:
            input_data, observation_data = data_loader.dataset.tensors  
            input_data = input_data.to(device)
            observation_data = observation_data.to(device)
        else:
            input_data_lists = defaultdict(list)
            observation_data_list = []
            for in_data, obs_data in iter(data_loader):
                for i, data in enumerate(_as_tuple(in_data)):
                    input_data_lists[i].append(data.to(device))
                observation_data_list.append(obs_data.to(device))
            input_data = tuple(torch.cat(input_data_lists[i]) for i in range(len(input_data_lists)))
            observation_data = torch.cat(observation_data_list)
        self._mcmc = MCMC(self.kernel, num_samples, **mcmc_kwargs)
        self._mcmc.run(input_data, observation_data)

        return self._mcmc

    def predict(self, *input_data, num_predictions=1, aggregate=False):
        if self._mcmc is None:
            raise RuntimeError("Call .fit to run MCMC and obtain samples from the posterior first.")

        preds = []
        weight_samples = self._mcmc.get_samples(num_samples=num_predictions)
        with torch.no_grad():
            for i in range(num_predictions):
                weights = {name: sample[i] for name, sample in weight_samples.items()}
                preds.append(poutine.condition(self, weights)(*input_data))
        predictions = torch.stack(preds)
        return self.likelihood.aggregate_predictions(predictions) if aggregate else predictions
