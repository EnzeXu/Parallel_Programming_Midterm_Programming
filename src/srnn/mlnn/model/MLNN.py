import torch
from torch import nn
from typing import List

from .Taylor import Taylor

class MLNN(nn.Module):
    def __init__(
        self, 
        layer_size: List[int],
        layer: str = 'Linear',
        activation: str = 'ReLU',
        batchnorm: bool = False,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        layer_num = len(layer_size) - 1
        
        # Set layer
        if layer == 'Linear': Layer = nn.Linear
        elif layer == 'Taylor': Layer = Taylor
        else:
            raise ValueError(f'{layer} is not allowed in this model!')
        
        # Set activation function
        if activation == 'ReLU': Act = nn.ReLU
        elif activation == 'Tanh': Act = nn.Tanh
        elif activation == 'Sigmoid': Act = nn.Sigmoid
        else:
            raise ValueError(f'{activation} is not allowed in this model!')
        
        for i in range(layer_num):
            self.add_module(f'{layer}{i+1}', Layer(layer_size[i], layer_size[i+1]))
            if i != layer_num - 1:
                if batchnorm:
                    self.add_module(f'BatchNorm{i+1}', nn.BatchNorm1d(layer_size[i+1]))
                self.add_module(f'{activation}{i+1}', Act())
                if dropout != 0.0:
                    self.add_module(f'Dropout{i+1}', nn.Dropout(dropout))
            
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            X = module(X)
        return X