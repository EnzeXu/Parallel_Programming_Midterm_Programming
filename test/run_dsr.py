import os
import sys
import argparse
import warnings
import time
import numpy as np
from numpy import *
import pandas as pd
from sympy import simplify, expand
from contextlib import redirect_stdout
from dso import DeepSymbolicOptimizer

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ["CUDA_VISIBLE_DEVICES"]="1"

data_folder = 'data/'
config_folder = 'config/dsr/'
output_folder = 'results_dsr/'

parser = argparse.ArgumentParser(description='nggp')
parser.add_argument(
    '--task',
    default='??',
    type=str, help="""please select the benchmark task from the list []""")
args = parser.parse_args()


def simplify_eq(eq):
    return str(expand(simplify(eq)))


def main(args):
    task = args.task
    num_test = 10
    norm_threshold = 1e-4
    save_eqs = True

    all_eqs = []
    num_success = 0

    ## define testing independent variables
    data_x = np.load(f"data/{task}_x.npy")
    data_y = np.load(f"data/{task}_y.npy")
    
    train_num = int(len(data_x) * 0.8)
    
    # x_train = data_x[:train_num, ...]
    # f_train = data_y[:train_num, ...]
    
    ## define testing variables 
    x_test = data_x[train_num:, ...]
    f_test = data_y[train_num:, ...]
     
    num_var = x_test.shape[1] - 1
    
    eval_dict = {
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
    }
    for x_id in range(num_var):
        eval_dict[f"x{x_id + 1}"] = x_test[:, x_id]

    for i_test in range(num_test):
        print("\rTest {}/{}.".format(i_test, num_test), end="")
        sys.stdout.flush()
        
        # with redirect_stdout(None):
        model = DeepSymbolicOptimizer(config_folder + task + ".json")
        result = model.train()
        dsr_eq = simplify_eq(result['expression'])

        print("## returned from dsr:", dsr_eq)                
        all_eqs.append((1e9, dsr_eq))
        
        
        try: 
            f_pred = eval(dsr_eq, eval_dict)
            
            mse = np.linalg.norm((f_pred - f_test) / f_test.max(), 2) / f_test.shape[0]
            print(mse)
            print(dsr_eq)
            
            if mse <= norm_threshold:
                num_success += 1
            all_eqs[-1] = (mse, all_eqs[-1][1])
        except NameError:
            continue 

    if save_eqs:
        output_file = open(output_folder + task + '.txt', 'w')
        for mse, eqs in all_eqs:
            output_file.write(f"{mse}, {eqs}\n")
        output_file.write('success rate : {:.0%}'.format(num_success / num_test))
        output_file.close()

    print()
    print('final result:')
    print('success rate :', "{:.0%}".format(num_success / num_test))


if __name__ == '__main__':
    main(parser.parse_args())