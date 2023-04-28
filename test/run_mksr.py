import os
import sys
import argparse
import warnings
import time
import numpy as np
from numpy import *

from config.jin import TestSettings

import sys
sys.path.append("../src")
from srnn.mlnn.trainer import Trainer
from svsr.spl.train import run_spl
from mksr.solver import MKSR

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data_folder = 'data/'
output_folder = 'results_mksr/'

parser = argparse.ArgumentParser(description='mksr')
parser.add_argument(
    '--task',
    default='??',
    type=str, help="""please select the benchmark task from the list []""")
args = parser.parse_args()


def main(args):
    task = args.task
    num_test = 10
    norm_threshold = 1e-4
    save_eqs = True

    all_eqs = []
    num_success = 0
    
    ## define training variables 
    
    data_x = np.load(f"data/{task}_x.npy")
    data_y = np.load(f"data/{task}_y.npy")
    
    train_num = int(len(data_x) * 0.8)
    
    x_train = data_x[:train_num, ...]
    y_train = data_y[:train_num, ...]
    
    x_test = data_x[train_num:, ...]
    y_test = data_y[train_num:, ...]
    
    
    cfg = TestSettings[task]  
    y_num = cfg['common']['y_num']
    x_num = cfg['common']['x_num']
    
    eval_dict = {}
    for x_id in range(x_num):
        eval_dict[f"x{x_id}"] = x_test[:, x_id]
    trainer = Trainer(
        func_name=task,
        data_x=x_train,
        data_y=y_train,
        **cfg['common'],
        **cfg['srnn_config'])
    trainer.run()
    for i_test in range(num_test):
        print("\rTest {}/{}.".format(i_test, num_test))
        eqs = {}
        neuro_eval = trainer.get_eval(0) # since only one y
        svsr_method = run_spl        
        mksr_model = MKSR(
            func_name=task,
            random_seed=None,
            neuro_eval=neuro_eval,
            svsr_method=svsr_method,
            svsr_cfg=cfg['svsr_config'],
            **cfg['common'],
            **cfg['mvsr_config'])
        mksr_model.run()
        # eval:
        eq = str(mksr_model)
        
        y_pred = eval(eq, eval_dict)
        mse = np.linalg.norm((y_pred - y_test) / y_test.max(), 2) / y_test.shape[0]
        print("eq:", eq)
        print("mse:", mse)
        if mse < norm_threshold:
            print("Success!")
            num_success += 1
        else:
            print("Failed")
        all_eqs.append(eq)
        
        
    if save_eqs:
        output_file = open(output_folder + task + '.txt', 'w')
        for eqs in all_eqs:
            for y, eq in eqs.items():
                output_file.write(f"{y}: {eq}\n")
        output_file.write('success rate : {:.0%}'.format(num_success / num_test))
        output_file.close()

    print()
    print('final result:')
    print('success rate :', "{:.0%}".format(num_success / num_test))


if __name__ == '__main__':
    main(parser.parse_args())