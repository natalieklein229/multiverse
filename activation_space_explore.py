"""
Exploring activation space stuff -- code based on chatGPT...

"""

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to initialize and store multiple models
def initialize_ensemble(input_dim, hidden_dim, output_dim, ensemble_size):
    ensemble = []
    for _ in range(ensemble_size):
        model = SimpleNN(input_dim, hidden_dim, output_dim)
        #model.apply(init_weights)  # Custom weight initialization if needed
        ensemble.append(model)
    return ensemble

# Custom initialization function (optional)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

# Function to log activations for the entire dataset
def log_activations(ensemble, data_loader, layer=0):
    activation_history = {i: [] for i in range(len(ensemble))}

    # Disable gradients for activation logging
    with torch.no_grad():
        for i, model in enumerate(ensemble):
            batch_activations = []
            for inputs, _ in data_loader:
                # Hook to capture activations
                def forward_hook(module, input, output):
                    batch_activations.append(output.cpu())

                # Register hook on the first layer
                if layer == 0:
                    hook = model.fc1.register_forward_hook(forward_hook)
                else:
                    hook = model.fc2.register_forward_hook(forward_hook)
                
                # Forward pass through the model
                model(inputs)
                
                # Remove the hook
                hook.remove()

            # Store activations for the whole dataset (concatenated from all batches)
            activation_history[i].append(torch.cat(batch_activations, dim=0))

    return activation_history

# Training function for the ensemble with end-of-epoch activation logging
def train_ensemble(ensemble, data_loader, data_loader_noshuf, criterion, epochs=10, learning_rate=0.001):
    # To store weights and activations
    weight_history = {i: [] for i in range(len(ensemble))}
    epoch_activation_history_0 = {i: [] for i in range(len(ensemble))}
    epoch_activation_history_1 = {i: [] for i in range(len(ensemble))}
    loss_history = {i: [] for i in range(len(ensemble))}

    # Optimizers for each model in the ensemble
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in ensemble]

    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            for i, model in enumerate(ensemble):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()

                # Log the weights
                weight_history[i].append({
                    'fc1_weight': copy.deepcopy(model.fc1.weight.data.cpu()),
                    'fc1_bias': copy.deepcopy(model.fc1.bias.data.cpu()),
                    'fc2_weight': copy.deepcopy(model.fc2.weight.data.cpu()),
                    'fc2_bias': copy.deepcopy(model.fc2.bias.data.cpu())
                })

                loss_history[i].append(loss.item())

        # Log activations at the end of each epoch for the entire dataset
        epoch_activation_history_epoch = log_activations(ensemble, data_loader_noshuf, layer=0)
        for i in range(len(ensemble)):
            epoch_activation_history_0[i].append(epoch_activation_history_epoch[i][0].numpy())
        epoch_activation_history_epoch = log_activations(ensemble, data_loader_noshuf, layer=1)
        for i in range(len(ensemble)):
            epoch_activation_history_1[i].append(epoch_activation_history_epoch[i][0].numpy())

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    return weight_history, epoch_activation_history_0, epoch_activation_history_1, loss_history

# %%
# Data preparation
# Replace this with your dataset; here we use random data as an example
input_dim = 1
hidden_dim = 500
output_dim = 1
ensemble_size = 30  # Number of models in the ensemble
epochs = 20
n = 500

# Generate some synthetic data
# X = torch.randn(100, input_dim)
# y = torch.randn(100, output_dim)
# dataset = TensorDataset(X, y)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Generate a sine wave dataset
def generate_sine_wave(n_samples=100, noise_std=0.1):
    X = np.linspace(0, 4 * np.pi, n_samples).reshape(-1, 1)
    y = np.sin(X) + noise_std * np.random.randn(n_samples, 1)  # Add some noise
    X = X/10
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Generate a linear dataset with optional noise
def generate_linear_data(n_samples=100, slope=2.0, intercept=1.0, noise_std=0.1):
    X = 10 * torch.rand(n_samples, 1) - 5  # X values between -5 and 5
    y = slope * X + intercept + noise_std * torch.randn(n_samples, 1)  # Linear relation with noise
    return X, y

# Create the dataset and dataloader
X, y = generate_sine_wave(n_samples=n, noise_std=0.1)
#X, y = generate_linear_data(n_samples=n, slope=3.0, intercept=2.0, noise_std=0.2)

dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
data_loader_noshuf = DataLoader(dataset, batch_size=32, shuffle=False)

# Initialize ensemble and define loss
ensemble = initialize_ensemble(input_dim, hidden_dim, output_dim, ensemble_size)
criterion = nn.MSELoss()

# Train the ensemble
weight_history, activation_history0, activation_history1, loss_history = train_ensemble(ensemble, data_loader, data_loader_noshuf, criterion, 
                                                    epochs=epochs, learning_rate=3e-4)

# activation_history is a dict
# activation_history[i][j] is size (n, hidden_dim) where i = ensemble index, j = epoch
# np.array(activation_history[0]) is (n_epochs, n, hidden_dim)

# %%
plt.figure()
for i in range(ensemble_size):
    plt.plot(loss_history[i])
plt.show()

# %%
predictions = []
for i in range(ensemble_size):
    model = ensemble[i]
    with torch.no_grad():
        outputs = model(torch.Tensor(X).float())
    predictions.append(outputs.cpu().numpy().squeeze())
predictions = np.array(predictions)

plt.figure()
for i in range(ensemble_size):
    plt.plot(X, predictions[i, :], '.')
plt.plot(X, y, 'k.')
plt.show()

# %% change into arrays
act_hist0 = np.zeros((epochs, ensemble_size, n, hidden_dim))
act_hist1 = np.zeros((epochs, ensemble_size, n, hidden_dim))
for i in range(ensemble_size):
    act_hist0[:, i, :, :] = activation_history0[i]
    act_hist1[:, i, :, :] = activation_history1[i]

# %%
from sklearn.decomposition import PCA
n_pc = 10
pc_projs = []
for i in range(len(X)):
    pc = PCA(n_components=n_pc).fit_transform(act_hist1[-1, :, i, :])
    pc_projs.append(pc)
pc_projs = np.array(pc_projs) # (len(X), ensemble_size, n_pc)

# %%
pc_i = 1
plt.figure()
for i in range(ensemble_size):
    plt.plot(X, np.abs(pc_projs[:, i, pc_i])) #sign flip issue
plt.show()

# %%
plt.figure()
plt.plot(pc_projs[0, :, 0], pc_projs[0, :, 1], 'k.')

# %%
# Convert the 3D array to a DataFrame in long form
# Step 1: Reshape the array into long form
array_reshaped = act_hist1[-1, :, :, :].reshape(-1)  # Flatten the array

# Step 2: Create MultiIndex based on the original dimensions
index = pd.MultiIndex.from_product(
    [range(ensemble_size), range(n), range(hidden_dim)],
    names=["ens_index", "n_index", "act_index"]
)

# Step 3: Create the DataFrame
df_long = pd.DataFrame({"Value": array_reshaped}, index=index).reset_index()
df_long["x"] = df_long["n_index"].map(dict(enumerate(X.numpy().squeeze())))

plt.figure()
sns.scatterplot(data=df_long.query('act_index == 0'),x='x',y='Value')

# %%
plt.figure()
sns.kdeplot(data=df_long, x='Value', hue='act_index')

# %%
epo_i = epochs-1
act_i = 0
plt.figure()
for i in range(ensemble_size):
    res = np.array(activation_history1[i])
    plt.plot(X, res[epo_i, :, act_i])
plt.xlabel('x')
plt.ylabel('act_i')

# %%
act_i = 20
x_i = 1
plt.figure()
for i in range(ensemble_size):
    res = np.array(activation_history1[i])
    plt.plot(np.arange(epochs), res[:, 1, act_i])
plt.xlabel('epoch')
plt.ylabel('activation for one x')

# %%
# act_i = 20
# x_i = 1
# ens_i = 0
# plt.figure()
# res = np.array(activation_history[ens_i])
# plt.scatter(np.tile(np.arange(epochs),len(X)), res[:, :, act_i].flatten(), c=np.repeat(X,epochs))
# plt.xlabel('epoch')
# plt.ylabel('activation for one x')
# %%
