"""
Fit CNN model to ChemCam and compare to PLS. Prepare model for fitting BNNs. 

"""
# %%
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

vnir_range = [492.427, 849.0]
vio_range = [382.13, 473.184]
uv_range = [246.635, 338.457]
keep_shots = ['shot%d' % i for i in range(5, 50)]
oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% functions
class CCamCNN(L.LightningModule):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.cnn(x.unsqueeze(1))
        loss = nn.functional.mse_loss(y_hat, y.to(device))
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.cnn(x.unsqueeze(1))
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def predict_step(self, batch):
        x, y = batch
        return self(x)

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data loading
    train_spec = np.load('data/train_spec.npy')
    val_spec = np.load('data/val_spec.npy')
    test_spec = np.load('data/test_spec.npy')
    train_oxides = np.load('data/train_oxides.npy')
    val_oxides = np.load('data/val_oxides.npy')
    test_oxides = np.load('data/test_oxides.npy')

    # PLS results
    try:
        pls_y_hat = np.load('results/PLS_predictions.npy')
    except:
        pls_y_hat = np.zeros_like(test_oxides)

    # CNN
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_spec).float(), torch.from_numpy(train_oxides).float()), 
                            batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_spec).float(), torch.from_numpy(val_oxides).float()), 
                            batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_spec).float(), torch.from_numpy(test_oxides).float()),
                            batch_size=64, shuffle=False)

    cnn = CNN(in_dim=train_spec.shape[1], out_dim=len(oxides), ch_sizes=[32,128,1],
            krnl_sizes=[11,5,1], stride=[3,3,3], lin_l_sizes = [20, 20],
            activation='relu', device=device)

    model = CCamCNN(cnn)
    trainer = L.Trainer(max_epochs=args.n_epo, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    #
    cnn_y_hat = trainer.predict(dataloaders=test_loader)
    cnn_y_hat = torch.cat(cnn_y_hat, 0)

    # for i in range(len(oxides)):
    #     plt.figure()
    #     plt.plot(test_oxides[:, i], cnn_y_hat[:, i], 'ko')
    #     plt.plot(test_oxides[:, i], pls_y_hat[:, i], 'r.', alpha=0.7)
    #     plt.axline([0,0], slope=1)
    #     plt.title(oxides[i])
    #     plt.show()
    #     pls_test_mse = np.mean(np.square(pls_y_hat[:, i]-test_oxides[:, i]))
    #     cnn_test_mse = np.mean(np.square(cnn_y_hat[:, i].detach().numpy()-test_oxides[:, i]))
    #     print('PLS test mse: %0.3g, CNN test mse: %0.3g' % (pls_test_mse, cnn_test_mse))

    np.save('results/cnn%d_predictions.npy' % args.seed, cnn_y_hat)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='fit CNN to ChemCam; change seed for ensemble')
    
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--n_epo', default=200, type=int, help='Number training epohcs')

    args = parser.parse_args()
    
    print(args)
    
    main(args)