import torch
from torch import nn
import tensorly as tl
tl.set_backend('pytorch')


class Taylor(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 X0 = 0, order: int = 3, rank: int = 16) -> None:
        super().__init__()
        self.X0 = X0
        
        self.const = nn.Parameter(torch.empty((out_features, )))
        self.Ps = nn.ParameterList([
            nn.Parameter(torch.empty((out_features, rank)))
            for _ in range(order)
        ])
        self.Lambda = nn.ParameterList([
            nn.Parameter(torch.empty((rank,)))
            for _ in range(order)
        ])
        self.Qs = nn.ParameterList([
            nn.Parameter(torch.empty((in_features, rank)))
            for _ in range(order)
        ])
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.zeros_(self.const)
        for i in range(len(self.Ps)):
            nn.init.normal_(self.Ps[i], 0, 0.01)
            nn.init.normal_(self.Qs[i], 0, 0.01)
            nn.init.zeros_(self.Lambda[i])
            
    def reconstruct(self):
        self.to(torch.device('cpu'))
        order = len(self.Ps)
        out_features = self.Ps[0].shape[0]
        in_features = self.Qs[0].shape[0]
        params = [self.const.detach()]
        
        def chain_identical_khatri_rao(Tensor, n):
            if n == 1:
                return Tensor
            else:
                row, col = Tensor.shape
                res = Tensor
                for j in range(1, n):
                    tmp = Tensor.reshape([row] + \
                        [1 for _ in range(j)] + [col])
                    res = res.unsqueeze(0) * tmp
                return res.reshape((-1, col))
            
        for i in range(order):
            P, Q, Lambda = self.Ps[i], self.Qs[i], self.Lambda[i]
            unfolding = P @ torch.diag(Lambda) @chain_identical_khatri_rao(Q, i + 1).T
            shape = (out_features,) + (in_features,) * (i + 1)
            param = tl.fold(unfolding, mode = 0, shape = shape)
            params.append(param.detach())
        return params

    def forward(self, X):
        flag_reshape = False
        if X.dim() == 1:
            X = X.reshape(1, -1)
            flag_reshape = True
                    
        if type(self.X0) == torch.tensor:
            assert X.shape[1] == self.X0.shape[0]
            self.X0.to(X.device)
        X = X - self.X0
        
        Y = self.const + sum([
            (X @ self.Qs[i])**(i+1) @ torch.diag(self.Lambda[i]) @ self.Ps[i].T
            for i in range(len(self.Ps))
        ])
        
        if flag_reshape:
            return Y.reshape(-1)
        else:
            return Y

        
if __name__ == '__main__':
    d, o, b = 3, 4, 128
    X = torch.randn((b, d))
    model = Taylor(d, o)
    print(model(X).shape)
    params = model.reconstruct()
    print(params)