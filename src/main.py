# import sys
import numpy as np
import torch

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
C_REGRESSION_NUM = 400

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


def neuro_eval(x: np.ndarray):              
    """assume the NN can eval for any x
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x.T, dtype=torch.float32)
    return trainer.model(x).detach().numpy()


if __name__ == "__main__":
    np.random.seed(0)
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
        C_REGRESSION_NUM=C_REGRESSION_NUM,
        skip_step_1=2,
        skip_step_2=0)
    print(f"discovered euqation: {equation}")
