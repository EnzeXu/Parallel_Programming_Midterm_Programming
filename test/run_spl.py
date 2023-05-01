import numpy as np
import warnings
import argparse
from SymbolicPhysicsLearner.spl_train import run_spl

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data_folder = 'data/'
output_folder = 'results_spl/'

parser = argparse.ArgumentParser(description='spl')
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
    
    data_x = np.load(f"data/{task}_x.npy")
    data_y = np.load(f"data/{task}_y.npy")
    
    train_num = int(len(data_x) * 0.8)
    
    x_train = data_x[:train_num, ...]
    f_train = data_y[:train_num, ...]
    
    ## define testing variables 
    x_test = data_x[train_num:, ...]
    f_test = data_y[train_num:, ...]
    
    x_num = data_x.shape[1]
    eval_dict = {
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
    }
    for x_id in range(x_num):
        eval_dict[f"x{x_id}"] = x_test[:, x_id]
    
    # if __debug__:
    #     x_train = np.random.normal(0, 2, (10, 2))
    #     f_train = x_train[:, 0] + x_train[:, 1]
    
    ret_eqs, _, _ = run_spl(
        task, 
        transplant_step = 10000,
        num_transplant = 10,
        num_run=num_test,
        train_sample=np.hstack([x_train, f_train.reshape(len(f_train), 1)]).T,
        test_sample=np.hstack([x_test, f_test.reshape(len(f_test), 1)]).T,
    )
    print(ret_eqs)
    # if __debug__:
    #     exit(0)
    all_eqs = []
    num_success = 0
    for eq in ret_eqs:
        all_eqs.append((1e9, eq))
        try: 
            f_pred = eval(eq, eval_dict)
            
            mse = np.linalg.norm((f_pred - f_test) / f_test.max(), 2) / f_test.shape[0]
            print(mse)
            print(eq)
            
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