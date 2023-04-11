import torch
from torch import nn
from torch.utils import data
from torch.distributions.uniform import Uniform

import numpy as np
from scipy.integrate import odeint

import os
import time
import logging
import yaml

from ._utils import *
from ._settings import *
from ..model import *

class Trainer:
    def __init__(
        self,
        model_mode: nn.Module or str,
        func_name: str,
        batch_size: int = 128,
        lr: float = 0.0001,
        seed: int = 0,
        cuda: int = 0,
        **kwargs
    ) -> None:
        '''
        Training model
        
        Args:
            - model: the NN model that need to be trained. MLP or Taylor.
            - func_name: the name of function that be chosen.
            - batch_size: parameter during training.
            - lr: parameter during training.
            - epochs: parameter during training.
            - seed: random seed.
            - cuda: cuda number.
        '''
        # Create Model
        self.func_name = func_name
        set_random_seed(seed)
        if isinstance(model_mode, nn.Module):
            self.model = model_mode
        elif isinstance(model_mode, str):
            self.model = eval(model_mode)(**FuncSettings[func_name]['model'][model_mode])
        
        # Create Model Name
        class_name = self.model.__class__.__name__
        self.model_name = f'{class_name}-{func_name}-bs{batch_size}-lr{lr}-seed{seed}'
        self.make_dirs()
        logging.basicConfig(filename = self.logs_pth + self.model_name + '.log', level = logging.INFO)
        
        # Create training tools
        self.loss_fn = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
        self.train_iter, self.val_iter = self.generate_data_iter(func_name, batch_size)
            
    def fit(self, epochs):
        train_loss_list, val_loss_list = [], []
        min_loss, best_epoch = float('inf'), 0
        
        logging.info(f'train on {self.device}')
        self.model.to(self.device)
        
        for epoch in range(epochs):
            start = time.time()
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            final = time.time()
            
            if val_loss < min_loss:
                min_loss = val_loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_pth + self.model_name)
            
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            if (epoch + 1) % 10 == 0:
                torch.save(train_loss_list, self.loss_pth + 'train_loss')
                torch.save(val_loss_list, self.loss_pth + 'val_loss')
                
            info = f'epoch: {epoch:3},  train loss: {train_loss:.4e},  val loss: {val_loss:.4e},  time:{final-start:.5f},  best epoch: {best_epoch:3}'
            logging.info(info)
            print(info)
        
    def train_epoch(self):
        accu = Accumulator(2)
        for X, y in self.train_iter:
            self.opt.zero_grad()
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.model(X) 
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.opt.step()
            accu.add(loss.item() * len(y), len(y))
        return accu[0] / accu[1]
    
    def val_epoch(self):
        accu = Accumulator(2)
        with torch.no_grad():
            for X, y in self.val_iter:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)
                accu.add(loss.item() * len(y), len(y))
        return accu[0] / accu[1]
    
    def generate_data_iter(self, func_name: str, batch_size: int):
        cfg = FuncSettings[func_name]
        if cfg['type'] == 'normal':
            data_distr = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
            X = data_distr.sample((cfg['sample_times'], cfg['x_len'])).squeeze(-1)
            Y = cfg['func'](X).unsqueeze(-1)
            split = int(len(X) * 0.8)
            trainset = data.TensorDataset(X[:split], Y[:split])
            valset = data.TensorDataset(X[split:], Y[split:])
            
        else:
            x_range = np.asarray(cfg['x_range'], dtype = np.float32)
            train_x0 = (x_range[:, 1] - x_range[:, 0]) * \
                np.random.rand(len(x_range)) + x_range[:, 0]
            tspan = np.linspace(0, cfg['t_f'], int(cfg['t_f'] / cfg['dt']))
            train_x = torch.as_tensor(odeint(cfg['func'], train_x0, tspan), dtype = torch.float32)
            trainset = data.TensorDataset(train_x[:-1], train_x[1:])
            
            val_x0 = (x_range[:, 1] - x_range[:, 0]) * \
                np.random.rand(len(x_range)) + x_range[:, 0]
            tspan = np.linspace(0, cfg['t_f']*0.25, int(cfg['t_f']*0.25 / cfg['dt']))
            val_x = torch.as_tensor(odeint(cfg['func'], val_x0, tspan), dtype = torch.float32)
            valset = data.TensorDataset(val_x[:-1], val_x[1:])
            
        train_iter = data.DataLoader(
            trainset, batch_size = batch_size, shuffle = True
        )
        val_iter = data.DataLoader(
            valset, batch_size = batch_size, shuffle = True
        )
        return train_iter, val_iter
        
        
    def load(self):
        self.model.load_state_dict(torch.load(self.model_pth + self.model_name, map_location = 'cpu'))
    
    def make_dirs(self):
        self.results_pth = f'./results/{self.func_name}/'
        self.model_pth = self.results_pth + f'models/'
        self.loss_pth = self.results_pth + f'losses/{self.model_name}/'
        self.logs_pth = self.results_pth + f'logs/'
        
        pths = [self.model_pth, self.loss_pth, self.logs_pth]
        for pth in pths:
            if not os.path.exists(pth):
                os.makedirs(pth)