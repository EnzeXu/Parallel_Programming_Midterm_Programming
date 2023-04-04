import sys
import sympy as sy
import numpy as np
from numpy import sin, cos, log
from spl.spl_train import run_spl
from mksr.combiner import combiner

np.random.seed(1)
output_folder = 'logs/'   ## directory to save discovered results
save_eqs = True           ## if true, discovered equations are saved to "output_folder" dir

# target_equ = "x0**2 * x1 + x0 + 2 * x1"
target_equ = "sin(x0) * x1 + 3"
var_num = 2
x_range = (-5, 5)
task = 'task1'
grammars = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
            'A->x', 'A->C', 
            'A->exp(A)', 'A->cos(x)', 'A->sin(x)']
nt_nodes = ['A']
pivot = [0.233, 0.618]
bst_eqs = []

test_mksr_only = 0
if test_mksr_only:
    # bst_eqs = [sy.simplify('0.618*x0**2 + x0 + 1.236'), sy.simplify('11.8596*x1 + 3.14')]
    bst_eqs = ['0.618*sin(x0) + 3.0', '0.2309*x1 + 3.0']
    cb = combiner(var_num = var_num,
                  pivot = pivot,
                  eqs = bst_eqs)
    print(cb.equation)
    exit(0)


for var_id in range(var_num):
    print(f"For variable x{var_id}")
    X = np.random.uniform(*x_range, 100)
    for v in range(var_num):
        locals()[f'x{v}'] = pivot[v]
    locals()[f'x{var_id}'] = X
    Y = eval(target_equ)
    XY = np.append(X, Y).reshape(2, 100)

    b = int(len(X) * 0.7)
    train_sample = XY[:, :b]
    test_sample = XY[:, b:]

    all_eqs, success_rate, all_times = run_spl(task = task, 
                                            grammars = grammars,
                                            nt_nodes = nt_nodes,
                                            num_run = 2,
                                            train_sample = train_sample,
                                            test_sample = test_sample,
                                            transplant_step = 200,
                                            num_transplant = 2,
                                            eta = 0.999)
    bst_eqs.append(max(all_eqs, key=lambda x: x[1])[0].replace('x', f'x{var_id}'))

print(bst_eqs)

cb = combiner(var_num = var_num,
                pivot = pivot,
                eqs = bst_eqs)
print(cb.equation)                  