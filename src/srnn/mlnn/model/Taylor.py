import torch
from torch import nn
import tensorly as tl
tl.set_backend('pytorch')


class Taylor(nn.Module):
    def __init__(
        self, in_features: int, 
        out_features: int, 
        order: int = 2, 
        rank: int = 16
    ) -> None:
        super().__init__()
        
        self.const = nn.Parameter(torch.empty((out_features, )))
        self.Ps = nn.ParameterList([
            nn.Parameter(torch.empty((out_features, rank)))
            for _ in range(order)
        ])
        # self.Lambda = nn.ParameterList([
        #     nn.Parameter(torch.empty((rank,)))
        #     for _ in range(order)
        # ])
        self.Qs = nn.ParameterList([
            nn.Parameter(torch.empty((in_features, rank)))
            for _ in range(order)
        ])
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.zeros_(self.const)
        for i in range(len(self.Ps)):
            nn.init.xavier_uniform_(self.Ps[i])
            nn.init.xavier_uniform_(self.Qs[i])
            # nn.init.normal_(self.Lambda[i], 0, 0.01)

    def forward(self, X):
        flag_reshape = False
        if X.dim() == 1:
            X = X.reshape(1, -1)
            flag_reshape = True
        
        Y = self.const + sum([
            (X @ self.Qs[i])**(i+1) @ self.Ps[i].T
            for i in range(len(self.Ps))
        ])
        # Y = self.const + sum([
        #     (X @ self.Qs[i])**(i+1) @ torch.diag(self.Lambda[i]) @ self.Ps[i].T
        #     for i in range(len(self.Ps))
        # ])
        
        if flag_reshape:
            return Y.reshape(-1)
        else:
            return Y