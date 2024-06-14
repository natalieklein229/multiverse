# multiverse
User-friendly Bayesian Neural Networks (BNNs) using PyTorch and Pyro, built on top of [`TyXe`](https://github.com/TyXe-BDL/TyXe/tree/master) and extending it to the [linearized Laplace approximation](https://arxiv.org/abs/2008.08400). The implementation extends parts of the functionality of [`laplace`](https://github.com/AlexImmer/Laplace) to general likelihoods and priors.

**Inference methods:**
* Stochastic Variational inference (SVI): customize the variational posterior approximation as a Pyro guide, or use an Autoguide from `pyro.infer.autoguide.guides`.
* Linearized Laplace approximation (linLA): approximate the posterior with a Multivariate Normal, with covariance matrix built by inverting a generalized Gauss-Newton (GGN) approximation to the Hessian of the log-joint of data and parameters. Predicts by using the GLM predictive detailed in [Immer et al. (2021)](https://arxiv.org/abs/2008.08400) 
* MCMC: specify a Markov transition kernel to sample from the posterior of the parameters.

# Training, prediction, and evaluation
We begin by specifying an optimizer for the MAP, a neural architecture, a prior for the parameters, and a likelihood function (and, implicitly, a link function) for the response.
Simple neural architectures are provided in `BNNmultiverse.neural_nets`.

```
optim = pyro.optim.Adam({"lr": 1e-3})
net = multiverse.neural_nets.MLP(in_dim=1, out_dim=1, width=10, depth=2, activation="tanh")

wp = .1 # prior precision for the parameters of the BNN
prior = multiverse.priors.IIDPrior((dist.Normal(0., wp**-2)))
I believe, the complete list of required dependencies, excluding the standard library (e.g., `os`) is:
- [ ] pytorch
- [ ] matplotlib
- [ ] tqdm
- [ ] numpy
- [ ] https://github.com/ThomasLastName/quality_of_life


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


The code for SSGE was taken from the repo 



# TODO

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Ski_trail_rating_symbol-green_circle.svg/800px-Ski_trail_rating_symbol-green_circle.svg.png" width="16" height="16"> Improve the setup guidlines for this repo (venv?) and overall bring this repo into alignment with industry best practices.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Ski_trail_rating_symbol-green_circle.svg/800px-Ski_trail_rating_symbol-green_circle.svg.png" width="16" height="16"> Try out different `x_train` and `y_train` in `univar_bnn_demo.py`.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Ski_trail_rating_symbol-blue_square.svg/768px-Ski_trail_rating_symbol-blue_square.svg.png" width="16" height="16"> Do some multivariable toy problems _and_ visualize them using Michal's projection trick.


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Ski_trail_rating_symbol-green_circle.svg/800px-Ski_trail_rating_symbol-green_circle.svg.png" width="16" height="16"> Experiment more with the effect that the SSGE hyperparameters J and M have on the fBNN.


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Ski_trail_rating_symbol-green_circle.svg/800px-Ski_trail_rating_symbol-green_circle.svg.png" width="16" height="16"> Better **measurement sets** in fBNN.


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Ski_trail_rating_symbol-blue_square.svg/768px-Ski_trail_rating_symbol-blue_square.svg.png" width="16" height="16"> Better functional priors in fBNN (look into https://arxiv.org/abs/2011.12829).


<img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/Ski_trail_rating_symbol-black_diamond.svg" width="16" height="16"> Implement _Tractable Function-Space Variational Inference in Bayesian Neural Networks_ by Rudner et al https://arxiv.org/abs/2312.17199.



