import time

import torch
from torch import nn
import os

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

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
        epoch_step=10,
        layer = 'Linear',     # Linear | Taylor ;
        activation = 'ReLU',  # ReLU   | Tanh   ;
        layer_size = [2, 128, 256, 128, 1],
        batchnorm = True,
        dropout = 0.5,
        batch_size = 128,
        lr = 0.003,
        scheduler = "Fixed",  # Fixed | StepLR | LambdaLR # Enze edited
    ) -> None:
        self.func_name = func_name
        self.train_times = int(len(data_x) * 0.8)
        # normalize
        self.data_x_mean = data_x.mean(axis=0)
        self.data_x_std = data_x.std(axis=0)
        data_x = (data_x - self.data_x_mean) / self.data_x_std
        self.data_y_mean = data_y.mean(axis=0)
        self.data_y_std = data_y.std(axis=0)
        data_y = (data_y - self.data_y_mean) / self.data_y_std
        # split data
        self.val_times = len(data_x) - self.train_times
        self.train_x = torch.from_numpy(data_x[:self.train_times, ...])
        self.val_x = torch.from_numpy(data_x[self.train_times:, ...])
        self.train_y = torch.from_numpy(data_y[:self.train_times, ...])
        self.val_y = torch.from_numpy(data_y[self.train_times:, ...])
        # 
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
        self.epoch_step = epoch_step

        self.scheduler = scheduler  # Enze edited

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
            self.mlnn.eval()
            return
        except:
            pass
        opt = torch.optim.Adam(self.mlnn.parameters(), lr = self.lr)

        assert self.scheduler in ["Fixed", "StepLR", "LambdaLR", "CosineAnnealingLR"], "Scheduler must be chosen in [Fixed | StepLR | LambdaLR | CosineAnnealingLR]!"

        if self.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.8)
        elif self.scheduler == "LambdaLR":
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1 / (e / 1000 + 1))
        elif self.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=0.001 * self.lr)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0)

        loss_fn = nn.MSELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('train on', device)
        self.mlnn.to(device)

        train_x = self.train_x.clone().reshape((self.train_times, self.x_num))
        train_y = self.train_y.clone().reshape((self.train_times, self.y_num)).to(device)
        val_x = self.val_x.clone().reshape((self.val_times, self.x_num))
        val_y = self.val_y.clone().reshape((self.val_times, self.y_num)).to(device)
        
        best_epoch, min_loss = 0, float('Inf')
            
        train_loss_list = []
        valid_loss_list = []
        record_timestring_start = get_now_string()
        record_t0 = time.time()
        record_time_epoch_step = record_t0
        for epoch in range(1, self.epochs + 1):
            self.mlnn.train()
            opt.zero_grad()
            train_x_clone = train_x.clone().to(device)
            y_hat = self.mlnn(train_x_clone)
            train_loss = loss_fn(y_hat, train_y)
            train_loss_list.append(train_loss.item())
            train_loss.backward()
            opt.step()
            
            with torch.no_grad():
                self.mlnn.eval()
                val_x_clone = val_x.clone().to(device)
                y_hat = self.mlnn(val_x_clone)
                val_loss = loss_fn(y_hat, val_y)
                valid_loss_list.append(val_loss.item())
                
                
            if val_loss.item() < min_loss:
                best_epoch = epoch
                min_loss = val_loss.item()
                torch.save(self.mlnn.state_dict(), self.model_pth + self.model_name + "_{}.pt".format(record_timestring_start))
            

            if epoch % self.epoch_step == 0 or epoch == self.epochs:
                record_time_epoch_step_tmp = time.time()
                info_epoch = f'Epoch:{epoch}/{self.epochs}  train loss:{train_loss.item():.4e}  val loss:{val_loss.item():.4e}  '
                info_best = f'best epoch:{best_epoch}  min loss:{min_loss:.4e}  '
                info_extended = f'lr:{opt.param_groups[0]["lr"]:.9e}  time:{(record_time_epoch_step_tmp - record_time_epoch_step):.2f}s  time total:{((record_time_epoch_step_tmp - record_t0) / 60.0):.2f}min  time remain:{((record_time_epoch_step_tmp - record_t0) / 60.0 / epoch * (self.epochs - epoch)):.2f}min'
                record_time_epoch_step = record_time_epoch_step_tmp
                print(info_epoch + info_best + info_extended)

            scheduler.step()

        record_timestring_end = get_now_string()
        record_time_cost_min = (time.time() - record_t0) / 60.0
        record_folder_path = "./record_enze/"
        if not os.path.exists(record_folder_path):
            os.makedirs(record_folder_path)
        with open(record_folder_path + "record.csv", "a") as f:
            f.write("{0},{1},{2:.2f},{3},{4},{5},{6},{7:.9f},{8:.9f},{9},{10},{11},{12:.12f}\n".format(
                # timestring_start,timestring_end,time_cost_min,epochs,layer,activation,layer_size,lr,lr_end,scheduler,dropout,best_epoch,min_loss
                record_timestring_start,  # 0
                record_timestring_end,  # 1
                record_time_cost_min,  # 2
                self.epochs,  # 3
                self.layer,  # 4
                self.activation,  # 5
                str(self.layer_size),  # 6
                self.lr,  # 7
                opt.param_groups[0]["lr"],  # 8
                self.scheduler,  # 10
                self.dropout,  # 11
                best_epoch,  # 12
                min_loss,  # 13
            ))

        self.mlnn.to('cpu')
        self.mlnn.eval()
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_list)
        plt.subplot(1, 2, 2)
        plt.plot(valid_loss_list)
        # plt.savefig("loss_srnn.png")
    
    def get_eval(self, y_id):
        self.mlnn.eval()
        def eval(x: np.ndarray):
            x = x.T
            x = (x - self.data_x_mean) / self.data_x_std
            torch_x = torch.tensor(x, dtype=torch.float32)
            nn_result = self.mlnn(torch_x).detach().numpy()[:, y_id]
            y = np.array(nn_result.flat)
            y = y * self.data_y_std + self.data_y_mean
            return y
        return eval
    
    def _load(self):
        print(f"{self.model_pth + self.model_name}")
        self.mlnn.load_state_dict(torch.load(self.model_pth + self.model_name, map_location = 'cpu'))
    
    def _make_dirs(self):
        self.results_pth = f'./results_srnn/{self.func_name}/'
        self.model_pth = self.results_pth + f'models/'
        
        pths = [self.model_pth]
        for pth in pths:
            if not os.path.exists(pth):
                os.makedirs(pth)


def get_now_string(timestring_format="%Y%m%d_%H%M%S_%f"):
    return datetime.now().strftime(timestring_format)
