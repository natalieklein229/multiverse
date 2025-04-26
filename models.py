import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

class CCamCNN(L.LightningModule):
    def __init__(self, cnn, output_dim, weight_prior_precision=1.0, 
                 logvar_prior_mean=0.0, logvar_prior_precision=1.0):
        super().__init__()
        self.save_hyperparameters() 
        self.cnn = cnn
        self.log_var = nn.Parameter(torch.zeros(output_dim))
        # Prior parameters
        self.weight_prior_precision = weight_prior_precision
        self.logvar_prior_mean = logvar_prior_mean
        self.logvar_prior_precision = logvar_prior_precision

    def forward(self, x):
        return self.cnn(x)
    
    def negative_log_likelihood(self, x, y):
        y_pred = self.forward(x)
        precision = torch.exp(-self.log_var)
        nll = 0.5 * torch.sum(precision * (y - y_pred)**2 + self.log_var)
        return nll

    def negative_log_prior(self):
        log_prior = 0.0
        for param in self.parameters():
            if param is not self.log_var:
                log_prior += 0.5 * torch.sum(param**2)
        return self.weight_prior_precision * log_prior

    def log_var_prior_loss(self):
        # Gaussian prior on log_var
        diff = self.log_var - self.logvar_prior_mean
        return 0.5 * self.logvar_prior_precision * torch.sum(diff**2)

    def training_step(self, batch):
        x, y = batch
        #nll = self.negative_log_likelihood(x.unsqueeze(1), y.to(device))
        nll = self.negative_log_likelihood(x.unsqueeze(1), y)
        log_prior = self.negative_log_prior()
        logvar_prior = self.log_var_prior_loss()
        loss = nll + log_prior + logvar_prior
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        #nll = self.negative_log_likelihood(x.unsqueeze(1), y.to(device))
        nll = self.negative_log_likelihood(x.unsqueeze(1), y)
        log_prior = self.negative_log_prior()
        logvar_prior = self.log_var_prior_loss()
        #loss = nn.functional.mse_loss(y_hat, y.to(device))
        loss = nll + log_prior + logvar_prior
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self, lr=3e-4):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def predict_step(self, batch):
        x, y = batch
        return self(x)
