# import sys
import numpy as np
from spl.spl_train import run_spl
from mksr.mkdr_solver import run_mksr

# problem specification:
var_num = 3
x_range = {'x0': (-5, 5), 'x1': (-5, 5), 'x2': (-5, 5), 'x3': (2, 3)}
grammars = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
            'A->x', 'A->C',
            'A->exp(A)', 'A->cos(x)', 'A->sin(x)']
nt_nodes = ['A']
np.random.seed(0)

# hyper-parameter:
spl_eval_num = 90
spl_test_num = 10
c_regression_num = 100

def NeuroEval(x_list):               # assume the NN can eval for any x
    from numpy import sin, cos, log, exp
    # target_equ = "x0**2 * x1 + x0 + 2 * x1"
    # target_equ = "sin(x0) * x1 + 3"
    # target_equ = "x0 * x1 + x0 + 2 * x1 + exp(x1)"
    # target_equ = "sin(x0) * (2.5 * x1 ** 2 + cos(x1)) + x1 + 3"
    target_equ = "x0 * x1 + x0 + 2 * x1 / x2 + x2 * exp(x1)"
    # target_equ = "(2 + x0) / (2 + x1 ** 2) + 5"
    # target_equ = "(x0 - x1 * x2 / x3**2) / (1-x1**2/x3**2)**0.5"
    # target_equ = "(1 + x0) / (1 + x2**3) + x1"
    for v in range(var_num):
        locals()[f'x{v}'] = x_list[v]
    return eval(target_equ)

if __name__ == "__main__":
    np.random.seed(0)
    equa = run_mksr(var_num = var_num,
                    x_range = x_range,
                    grammars = grammars,
                    nt_nodes = nt_nodes,
                    NeuroEval = NeuroEval,
                    SVSR = run_spl,
                    spl_eval_num = spl_eval_num,
                    spl_test_num = spl_test_num,
                    c_regression_num = c_regression_num)
    print(f"discovered euqation: {equa}")