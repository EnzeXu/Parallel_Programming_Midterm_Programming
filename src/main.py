# import sys
import numpy as np
import torch
from svsr.spl.spl_train import run_spl
from mksr.mksr_solver import run_mksr

# problem specification:
func_name = 'test2'
var_num = 2
x_range = {'x0': (-5, 5), 'x1': (-5, 5), 'x2': (-5, 5), 'x3': (2, 3)}
grammars = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
            'A->x', 'A->C',
            'A->exp(x)', 'A->cos(x)', 'A->sin(x)']
nt_nodes_num = ['A']
np.random.seed(0)

# hyper-parameter:
spl_eval_num = 90
spl_test_num = 10
c_regression_num = 100

from srnn.utils import Trainer
trainer = Trainer(
    model_mode = 'MLP',
    func_name = func_name,
    batch_size = 128,
    lr = 0.0001,
    seed = 0,
    cuda = 0,
)
try:
    trainer.load()
except:
    trainer.fit(2000)
    trainer.load()
    trainer.model.cpu()
    
def neuro_eval(X: np.ndarray):               # assume the NN can eval for any x
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X.T, dtype = torch.float32)
    return trainer.model(X).detach().numpy()

if __name__ == "__main__":
    np.random.seed(0)
    equa = run_mksr(func_name = func_name,
                    var_num = var_num,
                    x_range = x_range,
                    grammars = grammars,
                    nt_nodes_num = nt_nodes_num,
                    neuro_eval = neuro_eval,
                    svsr = run_spl,
                    spl_eval_num = spl_eval_num,
                    spl_test_num = spl_test_num,
                    c_regression_num = c_regression_num)
    print(f"discovered euqation: {equa}")
