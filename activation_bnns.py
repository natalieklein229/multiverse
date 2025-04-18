import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal, kl_divergence


class BayesianAdditiveActivationNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, prior_sd=1.0, init_post_sd=1.0):
        """
        Bayesian NN with activation-space prior/posterior, "additive" version.
        Hard-coded to 3 layers with independent Gaussian posteriors on first layer 
        and correlated posteriors on second layer (just trying stuff out).

        Parameters:
            input_dim: dimension of input
            hidden_dim: dimension of hidden layers
            output_dim: output dim
            prior_sd: standard deviation used to scale priors
            init_post_sd: used to scale initial posterior standard deviation
        """
        super(BayesianAdditiveActivationNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Variational parameters for activations (mean and log std)
        self.activation_mean0 = nn.Parameter(torch.zeros(hidden_dim))
        self.activation_log_std0 = nn.Parameter(np.log(init_post_sd)+torch.zeros(hidden_dim))
        self.activation_mean1 = nn.Parameter(torch.zeros(hidden_dim))
        #self.activation_log_std1 = nn.Parameter(np.log(init_post_sd)*torch.zeros(hidden_dim))
        self.activation_log_diag_cov = nn.Parameter(np.log(init_post_sd)+torch.zeros(hidden_dim))  # Log of diagonal covariance entries
        self.activation_lower_triangular = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))  # Lower-triangular covariance
        
        # Define prior distribution over activations
        self.prior0 = Normal(torch.zeros(hidden_dim), prior_sd*torch.ones(hidden_dim))
        #self.prior1 = Normal(torch.zeros(hidden_dim), prior_sd*torch.ones(hidden_dim))
        self.prior1 = MultivariateNormal(torch.zeros(hidden_dim), prior_sd*torch.eye(hidden_dim))


    def forward(self, x, random_activ=True):
        """
        Forward method that samples from activations.
        
        Parameters:
            x: the input
            random_activ (bool): if True, samples from activations; 
                                 if False, does not sample (deterministic)
        
        Returns:
            network output
        """
        # Standard forward for the first layer
        x = self.fc1(x)
        
        ### first set of activations
        # Sample activations from the variational posterior
        if random_activ:
            activation_std0 = torch.exp(self.activation_log_std0)
            activation_dist0 = Normal(self.activation_mean0, activation_std0)
            activations0 = activation_dist0.rsample()
        
            # Calculate the KL divergence between posterior and prior for activations
            kl_div0 = kl_divergence(activation_dist0, self.prior0).sum()

            # Apply activations and proceed to next layer
            x = F.relu(x + activations0)  # Add sampled activations
        else:
            x = F.relu(x)
            kl_div0 = 0

        x = self.fc2(x)
        ### second set of activations
        if random_activ:
            # Construct the covariance matrix from the diagonal and lower-triangular part
            diagonal_cov = torch.diag(torch.exp(self.activation_log_diag_cov))
            lower_triangular_cov = torch.tril(self.activation_lower_triangular, -1)
            cov_matrix = diagonal_cov + lower_triangular_cov
            # Sample activations from the variational posterior
            #activation_std1 = torch.exp(self.activation_log_std1)
            #activation_dist1 = Normal(self.activation_mean1, activation_std1)
            activation_dist1 = MultivariateNormal(self.activation_mean1, scale_tril=cov_matrix)
            activations1 = activation_dist1.rsample()
            
            # Calculate the KL divergence between posterior and prior for activations
            kl_div1 = kl_divergence(activation_dist1, self.prior1).sum()

            # Apply activations and proceed to next layer
            x = F.relu(x + activations1)  # Add sampled activations
        else:
            x = F.relu(x)
            kl_div1 = 0

        x = self.fc3(x)
        
        return x, kl_div0+kl_div1


class BayesianAdditivePostActNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, prior_sd=1.0, init_post_sd=1.0):
        """
        Bayesian NN with activation-space prior/posterior, "additive" version but applied 
        after nonlinear activation function.
        Hard-coded to 3 layers with independent Gaussian posteriors on first layer 
        and correlated posteriors on second layer (just trying stuff out).

        Parameters:
            input_dim: dimension of input
            hidden_dim: dimension of hidden layers
            output_dim: output dim
            prior_sd: standard deviation used to scale priors
            init_post_sd: used to scale initial posterior standard deviation
        """
        super(BayesianAdditivePostActNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Variational parameters for activations (mean and log std)
        self.activation_mean0 = nn.Parameter(torch.zeros(hidden_dim))
        self.activation_log_std0 = nn.Parameter(np.log(init_post_sd)+torch.zeros(hidden_dim))
        self.activation_mean1 = nn.Parameter(torch.zeros(hidden_dim))
        #self.activation_log_std1 = nn.Parameter(np.log(init_post_sd)*torch.zeros(hidden_dim))
        self.activation_log_diag_cov = nn.Parameter(np.log(init_post_sd)+torch.zeros(hidden_dim))  # Log of diagonal covariance entries
        self.activation_lower_triangular = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))  # Lower-triangular covariance
        
        # Define prior distribution over activations
        self.prior0 = Normal(torch.zeros(hidden_dim), prior_sd*torch.ones(hidden_dim))
        #self.prior1 = Normal(torch.zeros(hidden_dim), prior_sd*torch.ones(hidden_dim))
        self.prior1 = MultivariateNormal(torch.zeros(hidden_dim), prior_sd*torch.eye(hidden_dim))


    def forward(self, x, random_activ=True):
        """
        Forward method that samples from activations.
        
        Parameters:
            x: the input
            random_activ (bool): if True, samples from activations; 
                                 if False, does not sample (deterministic)
        
        Returns:
            network output
        """
        # Standard forward for the first layer
        x = self.fc1(x)
        
        ### first set of activations
        # Sample activations from the variational posterior
        if random_activ:
            activation_std0 = torch.exp(self.activation_log_std0)
            activation_dist0 = Normal(self.activation_mean0, activation_std0)
            activations0 = activation_dist0.rsample()
        
            # Calculate the KL divergence between posterior and prior for activations
            kl_div0 = kl_divergence(activation_dist0, self.prior0).sum()

            # Apply activations and proceed to next layer
            x = F.relu(x) + activations0  # Add sampled activations
        else:
            x = F.relu(x)
            kl_div0 = 0

        x = self.fc2(x)
        ### second set of activations
        if random_activ:
            # Construct the covariance matrix from the diagonal and lower-triangular part
            diagonal_cov = torch.diag(torch.exp(self.activation_log_diag_cov))
            lower_triangular_cov = torch.tril(self.activation_lower_triangular, -1)
            cov_matrix = diagonal_cov + lower_triangular_cov
            # Sample activations from the variational posterior
            #activation_std1 = torch.exp(self.activation_log_std1)
            #activation_dist1 = Normal(self.activation_mean1, activation_std1)
            activation_dist1 = MultivariateNormal(self.activation_mean1, scale_tril=cov_matrix)
            activations1 = activation_dist1.rsample()
            
            # Calculate the KL divergence between posterior and prior for activations
            kl_div1 = kl_divergence(activation_dist1, self.prior1).sum()

            # Apply activations and proceed to next layer
            x = F.relu(x) + activations1 # Add sampled activations
        else:
            x = F.relu(x)
            kl_div1 = 0

        x = self.fc3(x)
        
        return x, kl_div0+kl_div1
        

class BayesianMultActivationNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, prior_sd=1.0, init_post_sd=1.0):
        """
        Bayesian NN with activation-space prior/posterior, "multiplicative" version.
        Hard-coded to 3 layers with independent Gaussian posteriors on first layer 
        and correlated posteriors on second layer (just trying stuff out).

        Parameters:
            input_dim: dimension of input
            hidden_dim: dimension of hidden layers
            output_dim: output dim
            prior_sd: standard deviation used to scale priors
            init_post_sd: used to scale initial posterior standard deviation
        """
        super(BayesianMultActivationNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Variational parameters for activations (mean and log std)
        self.activation_mean0 = nn.Parameter(torch.zeros(hidden_dim))
        self.activation_log_std0 = nn.Parameter(np.log(init_post_sd)+torch.zeros(hidden_dim))
        self.activation_mean1 = nn.Parameter(torch.zeros(hidden_dim))
        #self.activation_log_std1 = nn.Parameter(np.log(init_post_sd)*torch.zeros(hidden_dim))
        self.activation_log_diag_cov = nn.Parameter(np.log(init_post_sd)+torch.zeros(hidden_dim))  # Log of diagonal covariance entries
        self.activation_lower_triangular = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))  # Lower-triangular covariance
        
        # Define prior distribution over activations
        self.prior0 = Normal(torch.zeros(hidden_dim), prior_sd*torch.ones(hidden_dim))
        #self.prior1 = Normal(torch.zeros(hidden_dim), prior_sd*torch.ones(hidden_dim))
        self.prior1 = MultivariateNormal(torch.zeros(hidden_dim), prior_sd*torch.eye(hidden_dim))


    def forward(self, x, random_activ=True):
        """
        Forward method that samples from activations.
        
        Parameters:
            x: the input
            random_activ (bool): if True, samples from activations; 
                                 if False, does not sample (deterministic)
        
        Returns:
            network output
        """
        # Standard forward for the first layer
        x = self.fc1(x)
        
        ### first set of activations
        # Sample activations from the variational posterior
        if random_activ:
            activation_std0 = torch.exp(self.activation_log_std0)
            activation_dist0 = Normal(self.activation_mean0, activation_std0)
            activations0 = activation_dist0.rsample()
        
            # Calculate the KL divergence between posterior and prior for activations
            kl_div0 = kl_divergence(activation_dist0, self.prior0).sum()

            # Apply activations and proceed to next layer
            x = F.relu(x * activations0)  
        else:
            x = F.relu(x)
            kl_div0 = 0

        x = self.fc2(x)
        ### second set of activations
        if random_activ:
            # Construct the covariance matrix from the diagonal and lower-triangular part
            diagonal_cov = torch.diag(torch.exp(self.activation_log_diag_cov))
            lower_triangular_cov = torch.tril(self.activation_lower_triangular, -1)
            cov_matrix = diagonal_cov + lower_triangular_cov
            # Sample activations from the variational posterior
            #activation_std1 = torch.exp(self.activation_log_std1)
            #activation_dist1 = Normal(self.activation_mean1, activation_std1)
            activation_dist1 = MultivariateNormal(self.activation_mean1, scale_tril=cov_matrix)
            activations1 = activation_dist1.rsample()
            
            # Calculate the KL divergence between posterior and prior for activations
            kl_div1 = kl_divergence(activation_dist1, self.prior1).sum()

            # Apply activations and proceed to next layer
            x = F.relu(x * activations1)  
        else:
            x = F.relu(x)
            kl_div1 = 0

        x = self.fc3(x)
        
        return x, kl_div0+kl_div1