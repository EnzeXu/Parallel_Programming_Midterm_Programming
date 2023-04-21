import torch
from torch import nn
from torch.distributions.uniform import Uniform

import numpy as np
import matplotlib.pyplot as plt

from .model import MLNN

# cfg = {
#     'func': lambda x: x[:, 0] / (1 + x[:, 1]**2) + 2,
#     'x_num': 2,
#     'x_range': {
#         'x0': (-5, 5),
#         'x1': (-5, 5),
#     },
#     'sample_times': 40000,
# }

def _set_random_seed(seed):    
    '''set random seed'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
class Trainer:
    def __init__(
        self,
        func,
        sample_times,
        x_range,
        x_num,
        layer = 'Linear',     # Linear | Taylor ;
        activation = 'ReLU',  # ReLU   | Tanh   ;
        layer_size = [2, 128, 256, 128, 1],
        batchnorm = True,
        dropout = 0.5,
        batch_size = 128,
        lr = 0.003,
        epochs = 800,
        seed = 0,
        **kwargs,
    ) -> None:
        self.func = func
        self.sample_times = sample_times
        self.x_range = x_range
        self.x_num = x_num
        self.layer = layer
        self.layer_size = layer_size
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        _set_random_seed(self.seed)

    def run(self) -> None:
        mlnn = MLNN(
            layer_size = self.layer_size,
            layer = self.layer, 
            activation = self.activation,
            batchnorm = self.batchnorm
        )
        opt = torch.optim.Adam(mlnn.parameters(), lr = self.lr)
        loss_fn = nn.MSELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('train on', device)
        mlnn.to(device)
        
        distri = Uniform(torch.tensor([-5.0]), torch.tensor([5.0]))
        train_x = distri.sample((self.sample_times, self.x_num,))
        train_x = train_x.squeeze(-1)
        train_y = self.func(train_x).unsqueeze(-1).to(device)
        
        val_x = distri.sample((int(self.sample_times*0.25), self.x_num,))
        val_x = val_x.squeeze(-1)
        val_y = self.func(val_x).unsqueeze(-1).to(device)
        
        best_epoch, min_loss = 0, float('Inf')
            
        for epoch in range(self.epochs):
            mlnn.train()
            opt.zero_grad()
            train_x_clone = train_x.clone().to(device)
            y_hat = mlnn(train_x_clone)
            train_loss = loss_fn(y_hat, train_y)
            train_loss.backward()
            opt.step()
            
            with torch.no_grad():
                mlnn.eval()
                val_x_clone = val_x.clone().to(device)
                y_hat = mlnn(val_x_clone)
                val_loss = loss_fn(y_hat, val_y)
                
            if val_loss.item() < min_loss:
                best_epoch = epoch
                min_loss = val_loss.item()
            
            info_epoch = f'Epoch: {epoch},  train loss:{train_loss.item():.4e},  val loss:{val_loss.item():.4e}  '
            info_best = f'best epoch: {best_epoch},  min loss:{min_loss:.4e}'
            print(info_epoch + info_best)
    def get_eval(self):
        return lambda x: x