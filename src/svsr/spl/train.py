import numpy as np
import pandas as pd
import time
from .score import simplify_eq, score_with_est
from .base import SplBase
# from .spl_task_utils import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_spl(task,
            grammars,
            nt_nodes,
            num_run,
            train_sample,
            test_sample,
            transplant_step=10000,
            max_len=50, eta=0.9999,
            max_module_init=10, num_aug=5, exp_rate=1/np.sqrt(2), num_transplant=1,
            norm_threshold=1e-5, count_success=True,
            ):
    """
    Executes the main training loop of Symbolic Physics Learner.

    Parameters
    ----------
    task : String object.
        benchmark task name. 
    num_run : Int object.
        number of runs performed.
    transplant_step : Int object.
        number of iterations simulated for training between two transplantations. 
    data_dir : String object.
        directory of training data samples. 
    max_len : Int object.
        maximum allowed length (number of production rules ) of discovered equations.
    eta : Int object.
        penalty factor for rewarding. 
    max_module_init : Int object.
        initial maximum length for module transplantation candidates. 
    num_aug : Int object.
        number of trees for module transplantation. 
    exp_rate : Int object.
        initial exploration rate. 
    num_transplant : Int object.
        number of transplantation candidate update performed throughout traning. 
    norm_threshold : Float object.
        numerical error tolerance for norm calculation, a very small value. 
    count_success : Boolean object. 
        if success rate is recorded. 

    Returns
    -------
    all_eqs: List<Str>
        discovered equations. 
    success_rate: Float
        success rate of all runs performed. 
    all_times: List<Float>
        runtimes for successful runs. 
    """

    num_success = 0
    all_times = []
    all_eqs = []

    # number of module max size increase after each transplantation
    module_grow_step = (max_len - max_module_init) / num_transplant

    for i_test in range(num_run):

        print(f"task {task} round {i_test}")
        best_solution = ('nothing', 0)

        exploration_rate = exp_rate
        max_module = max_module_init
        reward_his = []
        best_modules = []
        aug_grammars = []

        start_time = time.time()
        discovery_time = 0

        for i_itr in range(num_transplant):

            spl_model = SplBase(data_sample=train_sample,
                                base_grammars=grammars,
                                aug_grammars=aug_grammars,
                                nt_nodes=nt_nodes,
                                max_len=max_len,
                                max_module=max_module,
                                aug_grammars_allowed=num_aug,
                                func_score=score_with_est,
                                exploration_rate=exploration_rate,
                                eta=eta)

            _, current_solution, good_modules = spl_model.run(transplant_step,
                                                              num_play=10,
                                                              print_flag=True,
                                                              print_freq=1000,
                                                              norm_threshold=norm_threshold)

            end_time = time.time() - start_time

            if not best_modules:
                best_modules = good_modules
            else:
                best_modules = sorted(
                    list(set(best_modules + good_modules)), key=lambda x: x[1])
            aug_grammars = [x[0] for x in best_modules[-num_aug:]]

            # print([simplify_eq(x[2]) for x in best_modules[-num_aug:]])

            reward_his.append(best_solution[1])

            if current_solution[1] > best_solution[1]:
                best_solution = current_solution
                # print(best_solution[0])
            max_module += module_grow_step
            exploration_rate *= 5

            # check if solution is discovered. Early stop if it is.
            test_score = score_with_est(simplify_eq(
                best_solution[0]), 0, test_sample, eta=eta)[0]
            if test_score >= 1 - norm_threshold:
                num_success += 1
                if discovery_time == 0:
                    discovery_time = end_time
                    all_times.append(discovery_time)
                break
            print()
            print(f"exp_rate = {exploration_rate}, eta = {eta}, exp = {best_solution[0]}")

        all_eqs.append((simplify_eq(best_solution[0]), test_score))
        print('\nround {} complete after {} iterations.'.format(i_test, i_itr+1))
        print('best solution: {}'.format(simplify_eq(best_solution[0])))
        print('test score: {}'.format(test_score))

    success_rate = num_success / num_run
    if count_success:
        print('success rate :', success_rate)

    result = max(all_eqs, key=lambda x: x[1])[0]
    return result
