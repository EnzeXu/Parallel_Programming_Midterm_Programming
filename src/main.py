import sys
import numpy as np
sys.path.append(r'../')
from spl.spl_train import run_spl

output_folder = 'results_dump/' ## directory to save discovered results
save_eqs = True                ## if true, discovered equations are saved to "output_folder" dir

task = 'nguyen-3'
all_eqs, success_rate, all_times = run_spl(task, 
                                           num_run=100, 
                                           transplant_step=10000)
                                           
if save_eqs:
    output_file = open(output_folder + task + '.txt', 'w')
    for eq in all_eqs:
        output_file.write(eq + '\n')
    output_file.close()

print('success rate :', "{:.0%}".format(success_rate))
print('average discovery time is', np.round(np.mean(all_times), 3), 'seconds')                                          