# multiverse
User-friendly Bayesian Neural Networks (BNNs) using PyTorch and Pyro, built on top of [`TyXe`](https://github.com/TyXe-BDL/TyXe/tree/master) and extending it to the [linearized Laplace approximation](https://arxiv.org/abs/2008.08400). The implementation extends parts of the functionality of [`laplace`](https://github.com/AlexImmer/Laplace) to general likelihoods and priors.

**Inference methods:**
* Stochastic Variational inference (SVI): customize the variational posterior approximation as a Pyro guide, or use an Autoguide from `pyro.infer.autoguide.guides`.
* Linearized Laplace approximation (linLA): approximate the posterior with a Multivariate Normal, with covariance matrix built by inverting a generalized Gauss-Newton (GGN) approximation to the Hessian of the log-joint of data and parameters. Predicts by using the GLM predictive detailed in [Immer et al. (2021)](https://arxiv.org/abs/2008.08400) 
* MCMC: specify a Markov transition kernel to sample from the posterior of the parameters.

# Training, prediction, and evaluation
We begin by specifying an optimizer for the MAP, a neural architecture, a prior for the parameters, and a likelihood function (and, implicitly, a link function) for the response.
Simple neural architectures are provided in `BNNmultiverse.neural_nets`.

<<<<<<< HEAD
<<<<<<< HEAD
```
optim = pyro.optim.Adam({"lr": 1e-3})
net = multiverse.neural_nets.MLP(in_dim=1, out_dim=1, width=10, depth=2, activation="tanh")
=======
test

>>>>>>> 5403cd7 (test commit adding .gif to gitignore)

wp = .1 # prior precision for the parameters of the BNN
prior = multiverse.priors.IIDPrior((dist.Normal(0., wp**-2)))
I believe, the complete list of required dependencies, excluding the standard library (e.g., `os`) is:
=======

# Setup

## Setup steps using anaconda on MacOS

(on windows, I think the only difference is that you need to use `copy` instead of `cp`?)

0. Open the terminal and say `conda env list` to confirm that the code is not present already.

1. (_create an env with standard / easy-to-install packages_) `conda create --name bnns python=3.10 tqdm matplotlib numpy plotly scipy pip` (if desired, you can swap `bnns` for your preferred name).

2. (_activate the env for further installs_) `conda activate bnns`.

3. (_install pytorch_) This may depend on whether you want cuda, and on your conda channels. The simplest approach is: first try `conda install pytorch`. If that doesn't work (probably because channels) then try instead `pip install torch`.

4. (_install quality-of-life to the active env_) Now, `pip install git+https://github.com/ThomasLastName/quality-of-life.git` should suffice.

5. (_install pyreadr_) `pip install pyreadr` because I honestly don't know what the correct anaconda channel is... If for some reason this doesn't work, you just won't be able to access the SLOSH data, but the majority of the codebase which doesn't use the SLOSH data should in theory still function correctly.

6. (_install this code_) Navigate to wherever you want (e.g., the Documents folder), and clone this repo there.

7. (_verify installation_) Try running one of the python files, e.g., `python SSGE_multivar_demo.py`, which should create a .gif of some histograms.


## Dependencies

Well, you need pytorch and matplotlib and such.
Perhaps non-trivially you need tqdm.
**Most notably,** you need my helper utils https://github.com/ThomasLastName/quality_of_life which you just need clone to anywhere on the path for your python environment (I got the impression from Natalie that y'all are allowed clone repos off the internet to your lanl devices? You need this repo)

I believe, the complete list of required dependencies, excluding the standard library (e.g., `typing`) is:
>>>>>>> 1892f43 (Minute tweaks to README)
- [ ] pytorch
- [ ] matplotlib
- [ ] tqdm
- [ ] numpy
- [ ] scipy
- [ ] plotly
- [ ] pyreadr
- [ ] https://github.com/ThomasLastName/quality-of-life (this repo has its own dependencies, but I believe it is sufficient to run this repo with only the above packages installed; I believe "the required parts" of this repo depend only on the same 5 packages as above and the standard python library).

If desired, the dependencies on `plotly` and `quality_of_life` could be removed.

## Running the code

nprec = .1**-2 # noise precision for the likelihood function
likelihood = multiverse.likelihoods.HomoskedasticGaussian(n, precision=nprec)
```

For SVI and MCMC, see [TyXe](https://github.com/TyXe-BDL/TyXe/blob/master/README.md). For linLA, we can specify an approximation to the GGN approx. of the Hessian:
* `full` computes the full GGN
* `diag` computes a diagonal approx. of the GGN
* `subnet` considers `S_perc`% of the parameters having the highest posterior variance, as detailed in [Daxberger et al. (2021)](http://proceedings.mlr.press/v139/daxberger21a.html), to build a full GGN, fixing the other parameters at the MAP
For example:
```
bayesian_mlp = multiverse.LaplaceBNN(net, prior, likelihood, approximation='subnet', S_perc=0.5)
```

We can then train the model by calling:
```
num_epochs = 100
bayesian_mlp.fit(train_loader, optim, num_epochs)
```

Samples from the posterior in function space can be used to get samples from the posterior predictive:
```
f_samples = bayesian_mlp.predict(input_data, num_predictions=100)
y_samples = bayesian_mlp.likelihood.sample(f_predictions)
```


 - The code for SSGE was taken from the repo https://github.com/AntixK/Spectral-Stein-Gradient



# TODO

See the Issues tab.