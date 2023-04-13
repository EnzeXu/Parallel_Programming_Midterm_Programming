# import sys
import numpy as np
import torch
import os
import time

from srnn.utils import Trainer
from svsr.spl.spl_train import run_spl
from mksr.mksr_solver import run_mksr

# problem specification:
FUNC_NAME = 'test2'
VAR_NUM = 2
X_RANGE = {'x0': (-5, 5), 'x1': (-5, 5), 'x2': (-5, 5), 'x3': (2, 3)}
GRAMMARS = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
            'A->x', 'A->C',
            'A->exp(x)', 'A->cos(x)', 'A->sin(x)']
NT_NODES = ['A']
np.random.seed(0)

# hyper-parameter:
SPL_TRAIN_NUM = 180
SPL_TEST_NUM = 20
C_REGRESSION_NUM = 200

trainer = Trainer(
    model_mode='MLP',
    func_name=FUNC_NAME,
    batch_size=128,
    lr=0.0001,
    seed=0,
    cuda=0,
)
try:
    trainer.load()
except:
    trainer.fit(2000)
    trainer.load()
    trainer.model.cpu()

diff = np.array([])
def neuro_eval(x: np.ndarray):              
    """assume the NN can eval for any x
    """
    # x = x.T
    # temp_x = x
    # rst1 = (x[:, 0]**2 * x[:, 1] + x[:, 0] + 2 * x[:, 1]).reshape((len(x), 1)) # + np.random.normal(0, 0.01, len(x)).T)
    # x = x.T
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x.T, dtype=torch.float32)
    rst2 = trainer.model(x).detach().numpy()
    # global diff
    # new_diff = np.abs(rst1 - rst2)
    # diff = np.append(diff, rst1 - rst2)
    
    # for i in range(len(new_diff)):
    #     if new_diff[i, 0] > 1:
    #         print("fail", temp_x[i, :], rst1[i], rst2[i], new_diff[i, 0])
    #     else:
    #         pass
    #         print("ok", temp_x[i, :], rst1[i], rst2[i], new_diff[i, 0])

    # print(len(diff))
    return rst2.reshape(len(rst2))


if __name__ == "__main__":
    np.random.seed(10)
    # tmp = np.array([[ 4.74205896, -3.59141173], [ 3.20109276, -3.59141173]])
    # tmp = tmp.T
    # print(tmp)
    # neuro_eval(tmp)
    # exit(0)
    equation = run_mksr(
        FUNC_NAME=FUNC_NAME,
        VAR_NUM=VAR_NUM,
        X_RANGE=X_RANGE,
        GRAMMARS=GRAMMARS,
        NT_NODES=NT_NODES,
        neuro_eval=neuro_eval,
        svsr=run_spl,
        SPL_TRAIN_NUM=SPL_TRAIN_NUM,
        SPL_TEST_NUM=SPL_TEST_NUM,
        C_REGRESSION_NUM=C_REGRESSION_NUM)
    np.save(f"results/test2/eval_error.npy", diff)
    print(f"discovered euqation: {equation}")
