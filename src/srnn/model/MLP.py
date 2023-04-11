'''
Author: Hongjue Zhao
Email: hongjue0830@zju.edu.cn
'''

import torch
from torch import nn
from typing import List

class MLP(nn.Module):
    '''
    MLP model.
    
    Arg:
    - seq: nn.Sequential. All related layers are in this sequential.
    '''
    def __init__(self, layer_size: List[int]) -> None:
        '''
        initialize a MLP model
        
        Input:
        - layer_size: List[int]. the hidden size of each layer. 
            - Keep it in mind that the first and last size are 
              `in_features` and `out_features`.
        '''
        super().__init__()
        # according to the layer_size list, calculate the number of layers.
        layer_num = len(layer_size) - 1
        layers = []
        for i in range(layer_num):
            layers.append(nn.Linear(layer_size[i], layer_size[i+1]))
            # there is no activation function in the last layer
            if i < layer_num - 1:
                layers.append(nn.ReLU())
        self.seq = nn.Sequential(*layers)
        self.register_params()
        
    def register_params(self):
        def init_linear(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.seq.apply(init_linear)
        
    def forward(self, X: torch.tensor):
        return self.seq(X)
    
if __name__ == '__main__':
    b, d, o = 128, 224, 10
    X = torch.rand((b, d))
    model = MLP([d, 32, 64, o])
    print(model(X).shape)
    
    for layer in model.seq:
        X = layer(X)
        print(f'{layer.__class__.__name__}:\t\t{X.shape}')