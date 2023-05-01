import os
import sys
import argparse
import warnings
import time
from sympy import simplify, expand
import numpy as np
from numpy import *

from config.alltests import TestSettings

import sys
sys.path.append("../src")
from srnn.mlnn.trainer import Trainer
from svsr.spl.train import run_spl
from dso import DeepSymbolicOptimizer
import pandas as pd
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

def run_dso(
        task,
        train_sample,
        test_sample,
        config
    ):
    file_path = f'data/tmp/{task}_train.csv'
    print(train_sample.T.shape)
    pd.DataFrame(train_sample.T).to_csv(file_path, index=False, header=False)
    config['task']['dataset'] = file_path
    # print(config)
    model = DeepSymbolicOptimizer(config=config)
    # for i in range(train_sample.shape[1]):
    #     u = train_sample[0, i]
    #     fu = train_sample[1, i]
    #     print(u, fu, u**3 + u**2 + u ** 1)
    # exit(0)
    model.set_seeds()
    result = model.train()
    print(result['expression'])
    dsr_eq = str(expand(simplify(result['expression'].replace('x1', 'x'))))
    return dsr_eq

def main(args):
    task = args.task
    num_test = 1
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
    
    eval_dict = {
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
    }
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
        try:
            eqs = {}
            neuro_eval = trainer.get_eval(0) # since only one y
            svsr_method = run_dso
            mksr_model = MKSR(
                func_name=task,
                random_seed=None,
                neuro_eval=neuro_eval,
                svsr_method=svsr_method,
                svsr_cfg={"config": cfg['dsr_config']},
                **cfg['common'],
                **cfg['mvsr_config'])
            mksr_model.run()
            # eval:
            eq = str(mksr_model)
            
            y_pred = eval(eq, eval_dict)
            mse = np.linalg.norm((y_pred - y_test) / y_test.max(), 2) / y_test.shape[0]
            print("eq:", eq)
            print("mse:", mse)
        except:
            print("crashed")
            mse = 1e9
            eq = "crashed"
        if mse < norm_threshold:
            print("Success!")
            num_success += 1
        else:
            print("Failed")
        all_eqs.append((mse, eq))
        
        
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