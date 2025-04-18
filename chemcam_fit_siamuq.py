"""
Fit CNN MAP model to ChemCam. Prepare model for fitting BNNs. 

"""
# %%
import random
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from neural_nets.CNN import CNN

torch.set_float32_matmul_precision('medium')

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% functions
class CCamCNN(L.LightningModule):
    def __init__(self, cnn, output_dim, weight_prior_precision=1.0, 
                 logvar_prior_mean=0.0, logvar_prior_precision=1.0):
        super().__init__()
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
        nll = self.negative_log_likelihood(x.unsqueeze(1), y.to(device))
        log_prior = self.negative_log_prior()
        logvar_prior = self.log_var_prior_loss()
        loss = nll + log_prior + logvar_prior
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        nll = self.negative_log_likelihood(x.unsqueeze(1), y.to(device))
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

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # data loading
    train_spec = np.load('/data/0/chemcam_bnn/train_spec.npy')
    val_spec = np.load('/data/0/chemcam_bnn/val_spec.npy')
    test_spec = np.load('/data/0/chemcam_bnn/test_spec.npy')
    train_oxides = np.load('/data/0/chemcam_bnn/train_oxides.npy')
    val_oxides = np.load('/data/0/chemcam_bnn/val_oxides.npy')
    test_oxides = np.load('/data/0/chemcam_bnn/test_oxides.npy')

    # PLS results
    #try:
    #    pls_y_hat = np.load('results/PLS_predictions.npy')
    #except:
    #    pls_y_hat = np.zeros_like(test_oxides)

    # CNN
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_spec).float(), 
                                            torch.from_numpy(train_oxides).float()), 
                            batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_spec).float(), 
                                          torch.from_numpy(val_oxides).float()), 
                            batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_spec).float(), 
                                           torch.from_numpy(test_oxides).float()),
                            batch_size=64, shuffle=False)

    cnn = CNN(in_dim=train_spec.shape[1], out_dim=len(oxides), ch_sizes=[32,128,1],
            krnl_sizes=[11,5,1], stride=[3,3,3], lin_l_sizes = [20, 20],
            activation='relu', device=device)

    model = CCamCNN(cnn, len(oxides), args.weight_prior_precision, args.logvar_prior_mean, args.logvar_prior_precision)
    trainer = L.Trainer(max_epochs=args.n_epo, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    #
    cnn_y_hat = trainer.predict(dataloaders=test_loader)
    cnn_y_hat = torch.cat(cnn_y_hat, 0)

    for i in range(len(oxides)):
        plt.figure()
        plt.plot(test_oxides[:, i], cnn_y_hat[:, i], 'ko')
        #plt.plot(test_oxides[:, i], pls_y_hat[:, i], 'r.', alpha=0.7)
        plt.axline([0,0], slope=1)
        plt.title(oxides[i])
        plt.savefig('figures/cnn_preds_%s.png' % oxides[i])
        plt.show()
        #pls_test_mse = np.mean(np.square(pls_y_hat[:, i]-test_oxides[:, i]))
        cnn_test_mse = np.mean(np.square(cnn_y_hat[:, i].detach().numpy()-test_oxides[:, i]))
        #print('PLS test mse: %0.3g, CNN test mse: %0.3g' % (pls_test_mse, cnn_test_mse))
        print('CNN test mse %s: %0.3g' % (oxides[i], cnn_test_mse))

    noise_prec = torch.exp(-model.log_var).detach().cpu().numpy()
    print('mean oxide values')
    print(np.mean(train_oxides,0))
    print('SD oxide values')
    print(np.std(train_oxides,0))
    print('learned noise var')
    print(1./noise_prec)

    np.save('results/cnn%d_predictions.npy' % args.seed, cnn_y_hat)
    with open('results/cnn%d_args.pkl' % args.seed, 'wb') as f:
        pickle.dump(args, f)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='fit CNN to ChemCam; change seed for ensemble')
    
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--n_epo', default=200, type=int, help='Number training epochs')
    parser.add_argument('--weight_prior_precision', default=1.0, type=float, help='Weight prior precision')
    parser.add_argument('--logvar_prior_mean', default=0.0, type=float, help='Log var prior mean')
    parser.add_argument('--logvar_prior_precision', default=1.0, type=float, help='Log var prior precision')

    args = parser.parse_args()
    
    print(args)
    
    main(args)