import os
import sys
import argparse
import warnings
import time
import numpy as np
from numpy import *
import pandas as pd
from contextlib import redirect_stdout
from gplearn.genetic import SymbolicRegressor

from config.jin import gp_jin_cfg

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data_folder = 'data/'
output_folder = 'results_gp/'

parser = argparse.ArgumentParser(description='gp')
parser.add_argument(
    '--task',
    default='??',
    type=str, help="""please select the benchmark task from the list []""")
args = parser.parse_args()


variables = [f"x{i}" for i in range(10)]


def main(args):
    task = args.task
    num_test = 10
    population = 2000
    generation = 20
    norm_threshold = 1e-4
    save_eqs = True
    range_const = (-10, 10)

    all_times = []
    all_eqs = []
    num_success = 0
    
    ## define training variables 
    
    data_x = np.load(f"data/{task}_x.npy")
    data_y = np.load(f"data/{task}_y.npy")
    
    train_num = int(len(data_x) * 0.8)
    
    x_train = data_x[:train_num, ...]
    f_train = data_y[:train_num, ...]
    
    ## define testing variables 
    x_test = data_x[train_num:, ...]
    f_test = data_y[train_num:, ...]
    
    for i_test in range(num_test):
        print("\rTest {}/{}.".format(i_test, num_test), end="")
        sys.stdout.flush()
        
        start_time = time.time()
        
        # with redirect_stdout(None):
        est_gp = SymbolicRegressor(population_size=population, generations=generation, 
                                    stopping_criteria=0.01,const_range=range_const,
                                    max_samples=1, function_set=gp_jin_cfg[task], 
                                    feature_names=variables[:x_train.shape[1]])

        est_gp.fit(x_train, f_train)
        gp_eq = str(est_gp._program)
        
        end_time = time.time() - start_time
        all_eqs.append((-1, gp_eq))
        
        
        try: 
            f_pred = est_gp.predict(x_test)
            mse = np.linalg.norm((f_pred - f_test) / f_test.max(), 2) / f_test.shape[0]
            print(mse)
            print(gp_eq)
            
            if mse <= norm_threshold:
                num_success += 1
                all_times.append(end_time)
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