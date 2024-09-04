'''
Several functions from `TyXe` library: https://github.com/TyXe-BDL/TyXe/blob/master/tyxe/util.py
`TyXe` library license:
MIT License

Copyright (c) 2021 Hippolyt Ritter, Theofanis Karaletsos

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

`backpack_jacobian` function from `laplace` library: https://github.com/aleximmer/Laplace/blob/main/laplace/curvature/backpack.py
`laplace` library license:
MIT License

Copyright (c) 2021 Alex Immer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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

from collections import OrderedDict
import copy
from functools import reduce
from operator import mul, itemgetter
from warnings import warn

import torch

import pyro.util
import pyro.infer.autoguide.guides
import pyro.nn.module as pyromodule

from backpack import backpack, extend, memory_cleanup
from backpack.extensions import BatchGrad
from backpack.context import CTX

def make_psd(mat, threshold=1e-3, jitter=1e-3, noeval = False):
    if noeval:
        return (mat + mat.T)/2 + torch.eye(mat.shape[0])*jitter
    # check how far the matrix is from being symmetric
    diff = torch.norm(mat - mat.T)
    min_eigval = torch.min(torch.linalg.eigvals(mat).real)
    print(f"Matrix was {diff} from being symmetric, min eigenvalue was {min_eigval}")
    if diff < threshold:
        # Matrix is symmetric (or nearly symmetric)
        newmat = (mat + mat.T)/2
        if min_eigval < 0:
            # Matrix is not positive definite, add jitter
            print(f"Matrix is not positive definite, adding jitter {jitter}")
            newmat = newmat + torch.eye(mat.shape[0])*jitter
        print(f"Matrix is now {torch.norm(newmat - newmat.T)} from being symmetric, min eigenvalue is {torch.min(torch.linalg.eigvals(newmat).real)}" )
        return newmat
    else:
        # Matrix is not symmetric, handle the case accordingly
        print("Matrix is not nearly symmetric - returning original matrix")
        return mat
    
def is_psd(mat):
    diff = torch.norm(mat - mat.T)
    min_eigval = torch.min(torch.linalg.eigvals(mat).real)
    print(f"Matrix is {diff} from being symmetric, min eigenvalue is {min_eigval}")
    if bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all()):
        return True
    else:
        return False

def normalize(X, low_lim = None, up_lim = None):
    # constraint X to [0,1]
    if low_lim==None or up_lim==None:
        X = (X - torch.min(X,0)[0]) / (torch.max(X,0)[0] - torch.min(X,0)[0])
    else:
        X = (X - low_lim) / (up_lim - low_lim)
    return X

def unnormalize(X_normalized, X):
    # unnormalize X to original scale
    X_unnormalized = X_normalized * (torch.max(X,0)[0] - torch.min(X,0)[0]) + torch.min(X,0)[0]
    return X_unnormalized

def deep_hasattr(obj, name):
    warn('deep_hasattr is deprecated.', DeprecationWarning, stacklevel=2)
    try:
        pyro.util.deep_getattr(obj, name)
        return True
    except AttributeError:
        return False


def deep_setattr(obj, key, val):
    warn('deep_setattr is deprecated.', DeprecationWarning, stacklevel=2)
    return pyro.infer.autoguide.guides.deep_setattr(obj, key, val)

def deep_getattr(obj, name):
    warn('deep_getattr is deprecated.', DeprecationWarning, stacklevel=2)
    return pyro.util.deep_getattr(obj, name)


def to_pyro_module_(m, name="", recurse=True):
    """
    Same as `pyro.nn.modules.to_pyro_module_` except that it also accepts a name argument and returns the modified
    module following the convention in pytorch for inplace functions.
    """
    if not isinstance(m, torch.nn.Module):
        raise TypeError("Expected an nn.Module instance but got a {}".format(type(m)))

    if isinstance(m, pyromodule.PyroModule):
        if recurse:
            for name, value in list(m._modules.items()):
                to_pyro_module_(value)
                setattr(m, name, value)
        return

    # Change m's type in-place.
    m.__class__ = pyromodule.PyroModule[m.__class__]
    m._pyro_name = name
    m._pyro_context = pyromodule._Context()
    m._pyro_params = OrderedDict()
    m._pyro_samples = OrderedDict()

    # Reregister parameters and submodules.
    for name, value in list(m._parameters.items()):
        setattr(m, name, value)
    for name, value in list(m._modules.items()):
        if recurse:
            to_pyro_module_(value)
        setattr(m, name, value)

    return m


def to_pyro_module(m, name="", recurse=True):
    return to_pyro_module_(copy.deepcopy(m), name, recurse)


def named_pyro_samples(pyro_module, prefix='', recurse=True):
    yield from pyro_module._named_members(lambda module: module._pyro_samples.items(), prefix=prefix, recurse=recurse)


def pyro_sample_sites(pyro_module, prefix='', recurse=True):
    yield from map(itemgetter(0), named_pyro_samples(pyro_module, prefix=prefix, recurse=recurse))


def prod(iterable, initial_value=1):
    return reduce(mul, iterable, initial_value)


def fan_in_fan_out(weight):
    # this holds for linear and conv layers, but check e.g. transposed conv
    fan_in = prod(weight.shape[1:])
    fan_out = weight.shape[0]
    return fan_in, fan_out


def calculate_prior_std(method, weight, gain=1., mode="fan_in"):
    fan_in, fan_out = fan_in_fan_out(weight)
    if method == "radford":
        std = fan_in ** -0.5
    elif method == "xavier":
        std = gain * (2 / (fan_in + fan_out)) ** 0.5
    elif method == "kaiming":
        fan = fan_in if mode == "fan_in" else fan_out
        std = gain * fan ** -0.5
    else:
        raise ValueError(f"Invalid method: '{method}'. Must be one of ('radford', 'xavier', 'kaiming'.")
    return torch.tensor(std, device=weight.device)


def backpack_jacobian(model, x, enable_backprop=False):
    """
    From Laplace (https://github.com/aleximmer/Laplace/blob/main/laplace/curvature/backpack.py)
    Paper: Daxberger et al. (2021), Laplace Redux - Effortless Bayesian Deep Learning (https://arxiv.org/abs/2106.14806)
    
    Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
    using backpack's BatchGrad per output dimension.

    Parameters
    ----------
    x : torch.Tensor
        input data `(batch, input_shape)` on compatible device with model.
    enable_backprop : bool, default = False
        whether to enable backprop through the Js and f w.r.t. x

    Returns
    -------
    Js : torch.Tensor
        Jacobians `(batch, parameters, outputs)`
    f : torch.Tensor
        output function `(batch, outputs)`
    """
    model = extend(model)
    to_stack = []
    for i in range(model.output_size):
        model.zero_grad()
        out = model(x)
        with backpack(BatchGrad()):
            if model.output_size > 1:
                out[:, i].sum().backward(
                    create_graph=enable_backprop, 
                    retain_graph=enable_backprop
                )
            else:
                out.sum().backward(
                    create_graph=enable_backprop, 
                    retain_graph=enable_backprop
                )
            to_cat = []
            for param in model.parameters():
                to_cat.append(param.grad_batch.reshape(x.shape[0], -1))
                delattr(param, 'grad_batch')
            Jk = torch.cat(to_cat, dim=1)
        to_stack.append(Jk)
        if i == 0:
            f = out

    model.zero_grad()
    CTX.remove_hooks()
    _cleanup(model)
    if model.output_size > 1:
        return torch.stack(to_stack, dim=2).transpose(1, 2).squeeze(), f.squeeze()
    else:
        return Jk.unsqueeze(-1).transpose(1, 2).squeeze(), f.squeeze()
    

def _cleanup(module):
    for child in module.children():
        _cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)


def hessian(y, xs):
    # TODO: it would be nice to have a diagonal version, the problem is that computational graphs
    # are not retained when considering a view of the inputs (e.g. xs[0]), so we would need to
    # retain the graph for each element of xs, which is impossible if the prior probabilities
    # are computed jointly.

    dys = torch.autograd.grad(y, xs, create_graph=True)
    flat_dy = torch.cat([dy.reshape(-1) for dy in dys])
    H = []
    for dyi in flat_dy:
        Hi = torch.cat(
            [Hij.reshape(-1) for Hij in torch.autograd.grad(dyi, xs, retain_graph=True)]
        )
        H.append(Hi)
    H = torch.stack(H)        
    return H
