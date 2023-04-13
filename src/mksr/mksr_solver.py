"""Solve mksr problem use a given svsr method.
"""
import numpy as np
import os
from scipy.optimize import minimize
from numpy import sin, cos, log, exp
import sympy as sy
import re
import json


def _replace_number_with_capital_c(text: str) -> str:
    # TODO(cxt): a more detailed protection rule
    text = text.replace("**2", "**TWO")  # protect ^2
    pattern_scitific_repre = r"(?<!x)([0-9]*\.[0-9]*|[0-9]+)[eE][+-]?[0-9]+"
    text = re.sub(pattern_scitific_repre, "C", text)
    pattern_const = r"(?<!x)([0-9]*\.[0-9]*|[0-9]+)([eE][+-]?[0-9]+)?"
    text = re.sub(pattern_const, "C", text)
    text = text.replace("**TWO", "**2")  # protect ^2
    return text


def _replace_x_with_xi(text: str, Xi: str) -> str:
    pattern = r"(?<!e)x"
    return re.sub(pattern, Xi, text)


def _generate_data(
        equation: str,
        c_list: list,
        var_num: int,
        var_id: int,
        x_range: tuple,
        data_num: int,
        c_regression_num: int,
        neuro_eval: callable,
        record_file_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    """generate data for svsr

    Returns:
    X : shape = (1, test_num)       the variable x{var_id}'s value
    C : shape = (c_count, test_num) the corresponding constant value
    """
    pivot = np.empty(var_num)
    for vid in range(var_id + 1, var_num):
        pivot[vid] = np.random.uniform(*x_range[f"x{vid}"])
    C = np.empty((len(c_list), data_num))
    X = np.empty((1, data_num))
    for tid in range(data_num):
        cur_x = np.empty((var_num, c_regression_num))
        for vid in range(var_num):
            if vid < var_id:
                cur_x[vid, :] = np.random.uniform(
                    *x_range[f"x{vid}"], c_regression_num)
            elif vid == var_id:
                cur_x[vid, :] = X[0, tid] = np.random.uniform(
                    *x_range[f"x{vid}"])
            else:
                cur_x[vid, :] = pivot[vid]
        f_true = neuro_eval(cur_x)
        # print(cur_x)
        for v in range(var_num):
            globals()[f'x{v}'] = cur_x[v]
        # print(equation)
        def eq_test(c):
            for i in range(len(c)):
                locals()['c'+str(i)] = c[i]
            return np.linalg.norm(eval(equation) - f_true, 2)
        c_list = minimize(eq_test, [1.0] * len(c_list),
                          method='Powell', tol=1e-6).x.tolist()
        # print("gg")
        # cheat_result = [cur_x[vid, 0], 2*cur_x[vid, 0]]
        # print(c_list, cheat_result)
        # print(eq_test(c_list), eq_test(cheat_result))
        C[:, tid] = np.array(c_list)
    return X, C


def run_mksr(
        FUNC_NAME: str,
        VAR_NUM: int,
        X_RANGE: dict,
        GRAMMARS: list,
        NT_NODES: set[str],
        neuro_eval: callable,
        svsr: callable,
        SPL_TRAIN_NUM: int = 90,
        SPL_TEST_NUM: int = 10,
        C_REGRESSION_NUM: int = 100,
) -> str:
    """doing multi variable symbolic regression according to the neuro_eval.

    Args:
    func_name: the name of the equation to be discovered.
    var_num: the number of the variable, named as x0, x1, ...
    x_range: range of x. e.g. x_range['x0'] = (-1, 1)
    grammars: grammar list. e.g. 'A->A+A'
    nt_nodes: the set of all non terminal nodes.
    neuro_eval: the function provided by neuro network to do the evaluation.
    svsr: the beneath method to do the single variable regression.
    spl_train_num: how many tests needed to train spl.
    spl_test_num: how many tests needed to test spl.
    c_regression_num: how many tests needed to run the minimize.
    skip_step_1: reserve for future debug.
    skip_step_2: reserve for future debug.

    Returns: The expression in str format.
    """
    if not os.path.exists(f"results/{FUNC_NAME}/mksr"):
        os.makedirs(f"results/{FUNC_NAME}/mksr")
    record_equation_file_prefix = f"results/{FUNC_NAME}/mksr/equation"
    record_data_file_prefix = f"results/{FUNC_NAME}/mksr/data"
    equation = '0.0'  # any constant
    current_result = dict()
    for var_id in range(VAR_NUM):
        # `equa` consider variable [0, var_id),
        # now we expand x_{var_id} to the equation.
        print(f"Expand variable x{var_id}")
        print(f"Current equation: {equation}")
        equation = _replace_number_with_capital_c(equation)
        c_count = equation.count('C')
        c_list = ['c'+str(i) for i in range(c_count)]
        for c_with_id in c_list:
            equation = equation.replace('C', c_with_id, 1)
        print("After replacing:", equation)
        if os.path.exists(f"{record_data_file_prefix}X{var_id}.npy") and os.path.exists(f"{record_data_file_prefix}C{var_id}.npy"):
            X = np.load(f"{record_data_file_prefix}X{var_id}.npy")
            C = np.load(f"{record_data_file_prefix}C{var_id}.npy")
        else:
            X, C = _generate_data(
                equation=equation,
                c_list=c_list,
                var_num=VAR_NUM,
                var_id=var_id,
                x_range=X_RANGE,
                data_num=SPL_TEST_NUM + SPL_TRAIN_NUM,
                c_regression_num=C_REGRESSION_NUM,
                neuro_eval=neuro_eval,
                record_file_prefix=f"{record_data_file_prefix}X{var_id}")
            np.save(f"{record_data_file_prefix}X{var_id}.npy", X)
            np.save(f"{record_data_file_prefix}C{var_id}.npy", C)
        for cid in range(len(C)):
            save_name = f"{record_equation_file_prefix}_x{var_id}_c{cid}"
            if os.path.exists(save_name):
                with open(save_name, "r") as f:
                    equation = f.read()
            else:
                # Step 2 : for each constant, do sr using spl
                XC = np.append(X, C[cid, :]).reshape(
                    2, SPL_TEST_NUM + SPL_TRAIN_NUM)
                train_sample = XC[:, :SPL_TRAIN_NUM]
                test_sample = XC[:, SPL_TRAIN_NUM:]
                all_eqs, success_rate, all_times = svsr(
                    task=f"(x{var_id}, c{cid})",
                    grammars=GRAMMARS,
                    nt_nodes=NT_NODES,
                    num_run=1,
                    train_sample=train_sample,
                    test_sample=test_sample,
                    transplant_step=1000,
                    num_transplant=2,
                    eta=0.999)
                del success_rate, all_times
                result = max(all_eqs, key=lambda x: x[1])[0]
                result = f"({result})"
                result = _replace_x_with_xi(result, f"x{var_id}")
                equation = equation.replace(f'c{cid}', result)
                with open(save_name, "w") as f:
                    f.write(equation)
            current_result[str((var_id, cid))] = equation
        equation = str(sy.simplify(equation))
    return equation
