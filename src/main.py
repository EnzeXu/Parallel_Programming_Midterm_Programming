# import sys
import numpy as np
import torch
import os
import time

from srnn.utils import Trainer
from svsr.spl.spl_train import run_spl
from mksr.mksr_solver import MKSR

# problem specification:
FUNC_NAME = 'test1'
VAR_NUM = 3
X_RANGE = {'x0': (-5, 5), 'x1': (-5, 5), 'x2': (-5, 5), 'x3': (2, 3)}
GRAMMARS = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
            'A->x', 'A->C',
            'A->exp(x)', 'A->cos(x)', 'A->sin(x)']
NT_NODES = ['A']

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
    trainer.fit(10000)
    trainer.load()
    trainer.model.cpu()

if __debug__:
    diff = []
def neuro_eval(x: np.ndarray):              
    """assume the NN can eval for any x
    """
    torch_x = torch.tensor(x.T, dtype=torch.float32)
    nn_result = trainer.model(torch_x).detach().numpy()
    if __debug__:
        global diff
        x = x.T
        ground_truth_result = (x[:, 0]**2 * x[:, 1] + x[:, 0] + 2 * x[:, 1]).reshape((len(x), 1))
        x = x.T
        diff.extend((nn_result - ground_truth_result).flat)
    return nn_result.reshape(len(nn_result))


if __name__ == "__main__":
    mksr_model = MKSR(
        FUNC_NAME=FUNC_NAME,
        VAR_NUM=VAR_NUM,
        X_RANGE=X_RANGE,
        GRAMMARS=GRAMMARS,
        NT_NODES=NT_NODES,
        neuro_eval=neuro_eval,
        svsr=run_spl,
        SPL_TRAIN_NUM=SPL_TRAIN_NUM,
        SPL_TEST_NUM=SPL_TEST_NUM,
        C_REGRESSION_NUM=C_REGRESSION_NUM,
        random_seed=10)
    mksr_model.run()
    print(f"discovered euqation: {mksr_model}")
    if __debug__:
        with open(f"results/{FUNC_NAME}/eval_error.npy", "w") as diff_file:
            diff_file.write(repr(diff))