import sys
import numpy as np
sys.path.append(r'../')
from spl.spl_train import run_spl

output_folder = 'logs/' ## directory to save discovered results
save_eqs = True                ## if true, discovered equations are saved to "output_folder" dir

task = 'task1'
grammars = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
            'A->x', 'A->x**2', 'A->x**4', 
            'A->exp(A)', 'A->cos(x)', 'A->sin(x)']
nt_nodes = ['A']
num_run = 2

X = np.random.uniform(-2, 2, 100)
Y = X ** 3 + np.sin(X) ** 2 + X
XY = np.append(X, Y).reshape(2, 100)

b = int(len(X) * 0.7)
train_sample = XY[:, :b]
test_sample = XY[:, b:]

all_eqs, success_rate, all_times = run_spl(task = task, 
                                           grammars = grammars,
                                           nt_nodes = nt_nodes,
                                           num_run = num_run,
                                           train_sample = train_sample,
                                           test_sample = test_sample)
                                           
if save_eqs:
    output_file = open(output_folder + task + '.txt', 'w')
    for eq in all_eqs:
        output_file.write(eq + '\n')
    output_file.close()

print('success rate :', "{:.0%}".format(success_rate))
print('average discovery time is', np.round(np.mean(all_times), 3), 'seconds')                                          