import torch
from torch import nn
import os

import numpy as np
import matplotlib.pyplot as plt

from .model import MLNN
        
class Trainer:
    def __init__(
        self,
        func_name: str,
        x_range,
        x_num,
        y_num,
        data_x,
        data_y,
        epochs,
        layer = 'Linear',     # Linear | Taylor ;
        activation = 'ReLU',  # ReLU   | Tanh   ;
        layer_size = [2, 128, 256, 128, 1],
        batchnorm = True,
        dropout = 0.5,
        batch_size = 128,
        lr = 0.003,
    ) -> None:
        self.func_name = func_name
        self.train_times = int(len(data_x) * 0.8)
        self.val_times = len(data_x) - self.train_times
        self.train_x = torch.from_numpy(data_x[:self.train_times, ...])
        self.val_x = torch.from_numpy(data_x[self.train_times:, ...])        
        self.train_y = torch.from_numpy(data_y[:self.train_times, ...])
        self.val_y = torch.from_numpy(data_y[self.train_times:, ...])
        self.x_range = x_range
        self.x_num = x_num
        self.y_num = y_num
        self.layer = layer
        self.layer_size = layer_size
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.mlnn = MLNN(
            layer_size = self.layer_size,
            layer = self.layer, 
            activation = self.activation,
            batchnorm = self.batchnorm
        )
        class_name = self.mlnn.__class__.__name__
        self.model_name = f'{class_name}-{func_name}-bs{batch_size}-lr{lr}'
        self._make_dirs()

    def run(self) -> None:
        try:
            self._load()
            return
        except:
            pass
        opt = torch.optim.Adam(self.mlnn.parameters(), lr = self.lr)
        loss_fn = nn.MSELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('train on', device)
        self.mlnn.to(device)

        train_x = self.train_x.clone().reshape((self.train_times, self.x_num))
        train_y = self.train_y.clone().reshape((self.train_times, self.y_num)).to(device)
        val_x = self.val_x.clone().reshape((self.val_times, self.x_num))
        val_y = self.val_y.clone().reshape((self.val_times, self.y_num)).to(device)
        
        best_epoch, min_loss = 0, float('Inf')
            
        for epoch in range(self.epochs):
            self.mlnn.train()
            opt.zero_grad()
            train_x_clone = train_x.clone().to(device)
            y_hat = self.mlnn(train_x_clone)
            train_loss = loss_fn(y_hat, train_y)
            train_loss.backward()
            opt.step()
            
            with torch.no_grad():
                self.mlnn.eval()
                val_x_clone = val_x.clone().to(device)
                y_hat = self.mlnn(val_x_clone)
                val_loss = loss_fn(y_hat, val_y)
                
            if val_loss.item() < min_loss:
                best_epoch = epoch
                min_loss = val_loss.item()
                torch.save(self.mlnn.state_dict(), self.model_pth + self.model_name)
            
            info_epoch = f'Epoch: {epoch},  train loss:{train_loss.item():.4e},  val loss:{val_loss.item():.4e}  '
            info_best = f'best epoch: {best_epoch},  min loss:{min_loss:.4e}'
            print(info_epoch + info_best)
    
    def get_eval(self, y_id):
        self.mlnn.eval()
        def eval(x: np.ndarray):
            torch_x = torch.tensor(x.T, dtype=torch.float32)
            nn_result = self.mlnn(torch_x).detach().numpy()[:, y_id]
            # for x0, x1, y in zip(torch_x[:, 0], torch_x[:, 1], nn_result):
            #     print(x0, x1, y)
            return np.array(nn_result.flat)
        return eval
    
    def _load(self):
        self.mlnn.load_state_dict(torch.load(self.model_pth + self.model_name, map_location = 'cpu'))
        self.mlnn.eval()
    
    def _make_dirs(self):
        self.results_pth = f'./results_srnn/{self.func_name}/'
        self.model_pth = self.results_pth + f'models/'
        
        pths = [self.model_pth]
        for pth in pths:
            if not os.path.exists(pth):
                os.makedirs(pth)