"""
Trying activation space BNN (chatGPT)
"""

# %%
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from activation_bnns import BayesianAdditiveActivationNN, \
                            BayesianAdditivePostActNN, BayesianMultActivationNN

# %% Settings
n = 500
input_dim = 1
hidden_dim = 500
output_dim = 1
prior_sd = 10
init_post_sd = 0.001
epochs = 2000
model_type = 'additive_post' # or 'additive' or 'additive_post'

# Loss function for Bayesian Neural Network
def elbo_loss(output, target, kl_div, beta=1.0):
    # Likelihood term (e.g., mean squared error for regression)
    likelihood = F.mse_loss(output, target, reduction='sum')
    
    # ELBO combines likelihood and KL divergence
    return likelihood + beta * kl_div

def generate_sine_wave(n_samples=100, noise_std=0.1):
    X = np.concatenate([np.linspace(0, 1.5 * np.pi, n_samples//2),np.linspace(2.5 * np.pi, 4 * np.pi, n_samples//2)])
    #X = np.linspace(0, 4 * np.pi, n_samples).reshape(-1, 1)
    X = X.reshape(-1, 1)
    y = np.sin(X) + noise_std * np.random.randn(n_samples, 1)  # Add some noise
    X = X/10
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Training function
def train_bnn(model, data_loader, epochs=10, learning_rate=0.01, beta=1.0, pretrain_weights=True):

    if pretrain_weights:
        # Variational parameters for activations (mean and log std)
        model.activation_mean0.requires_grad = False
        model.activation_log_std0.requires_grad = False
        model.activation_mean1.requires_grad = False
        model.activation_log_diag_cov.requires_grad = False
        model.activation_lower_triangular.requires_grad = False
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            for inputs, targets in data_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs, kl_div = model.forward(inputs,random_activ=False)
                
                # Compute ELBO loss
                loss = F.mse_loss(outputs, targets, reduction='sum')
                loss.backward()
                optimizer.step()

        model.activation_mean0.requires_grad = True
        model.activation_log_std0.requires_grad = True
        model.activation_mean1.requires_grad = True
        model.activation_log_diag_cov.requires_grad = True
        model.activation_lower_triangular.requires_grad = True
  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_hist = []
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs, kl_div = model.forward(inputs, random_activ=True)
            
            # Compute ELBO loss
            loss = elbo_loss(outputs, targets, kl_div, beta=beta)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
        loss_hist.append(total_loss)
    
    return loss_hist

# %%
# Create the dataset and dataloader
X, y = generate_sine_wave(n_samples=n, noise_std=0.1)
data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=32, shuffle=True)

plt.figure()
plt.plot(X, y, 'k.')
plt.show()

# %%
# Initialize and train Bayesian Neural Network
if model_type == 'additive':
    bnn = BayesianAdditiveActivationNN(input_dim, hidden_dim, output_dim, prior_sd=prior_sd, init_post_sd=init_post_sd)
elif model_type == 'multiplicative':
    bnn = BayesianMultActivationNN(input_dim, hidden_dim, output_dim, prior_sd=prior_sd, init_post_sd=init_post_sd)
elif model_type == 'additive_post':
    bnn = BayesianAdditivePostActNN(input_dim, hidden_dim, output_dim, prior_sd=prior_sd, init_post_sd=init_post_sd)

# Train the model
loss_hist = train_bnn(bnn, data_loader, epochs=epochs, learning_rate=0.001, beta=1)#beta=1/prior_sd)

plt.figure()
plt.plot(loss_hist)
plt.show()

# %% Sample posterior
n_samp = 100
predictions = []
for i in range(n_samp):
    with torch.no_grad():
        outputs, kl = bnn.forward(torch.Tensor(X).float())
    predictions.append(outputs.cpu().numpy().squeeze())
predictions = np.array(predictions)

plt.figure()
plt.plot(X.squeeze(), np.mean(predictions,0), 'r-')
plt.fill_between(X.squeeze(), np.quantile(predictions, 0.1, 0), np.quantile(predictions, 0.9, 0), alpha=0.6)
plt.plot(X.squeeze(), y, 'k.')
# %%

# %%
