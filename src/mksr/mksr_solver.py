import numpy as np
from scipy.optimize import minimize
from numpy import sin, cos, log, exp
import sympy as sy
import re
import json


def ReplaceNumbersWithC(text):
    text = text.replace("**2", "**TWO") # protect ^2
    pattern = r"(?<!x)([0-9]*\.[0-9]*|[0-9]+)"
    text = re.sub(pattern, "C", text)
    text = text.replace("**TWO", "**2") # protect ^2
    return text

def ReplaceXwithXi(text, Xi):
    pattern = r"(?<!e)x"
    return re.sub(pattern, Xi, text)

def GenerateData(equa, 
                 var_num, 
                 var_id, 
                 x_range, 
                 spl_test_num, 
                 spl_eval_num,
                 c_regression_num,
                 NeuroEval):
    # Step 1 : generate data. The result is:
    # X : shape = (1, test_num)       the variable x{var_id}'s value
    # C : shape = (c_count, test_num) the corresponding constant value
    equa = ReplaceNumbersWithC(equa)
    c_count = equa.count('C')
    c_lst = ['c'+str(i) for i in range(c_count)]
    for c in c_lst: 
        equa = equa.replace('C', c, 1)
    print("After replacing:", equa)
    pivot = np.empty(var_num)
    for vid in range(var_id + 1, var_num):
        pivot[vid] = np.random.uniform(*x_range[f"x{vid}"])
    C = np.empty((c_count, spl_test_num + spl_eval_num))
    X = np.empty((1, spl_test_num + spl_eval_num))
    for tid in range(spl_eval_num + spl_test_num):
        cur_x = np.empty((var_num, c_regression_num)) 
        for vid in range(var_num):
            if vid < var_id:
                cur_x[vid, :] = np.random.uniform(*x_range[f"x{vid}"], c_regression_num)
            elif vid == var_id:
                cur_x[vid, :] = X[0, tid] = np.random.uniform(*x_range[f"x{vid}"])
            else:
                cur_x[vid, :] = pivot[vid]
        f_true = NeuroEval(cur_x)
        # print(cur_x)
        for v in range(var_num):
            globals()[f'x{v}'] = cur_x[v]
        def eq_test(c):
            for i in range(len(c)): 
                locals()['c'+str(i)] = c[i]
            return np.linalg.norm(eval(equa) - f_true, 2)
        c_lst = minimize(eq_test, [1.0] * len(c_lst), method='Powell', tol=1e-6).x.tolist() 
        C[:, tid] = np.array(c_lst)
    return X, C, equa

def run_mksr(func_name,
             var_num,
             x_range,
             grammars,
             nt_nodes,
             NeuroEval,
             SVSR,
             spl_eval_num = 90,
             spl_test_num = 10,
             c_regression_num = 100,
             skip_step_1 = None,
             skip_step_2 = None):
    record_file_name = f"results/{func_name}/mksr_last_result.json"
    equa = '0.0'  # any constant
    current_result = dict()
    for var_id in range(var_num):
        # `equa` consider variable [0, var_id), 
        # now we expand x_{var_id} to the equation.
        print(f"Expand variable x{var_id}")
        print(f"Current equation: {equa}")
        
        X, C, equa = GenerateData(equa = equa,
                        var_num = var_num, 
                        var_id = var_id, 
                        x_range = x_range, 
                        spl_test_num = spl_test_num, 
                        spl_eval_num = spl_eval_num,
                        c_regression_num = c_regression_num,
                        NeuroEval = NeuroEval)
        for cid in range(len(C)):
            if skip_step_2:
                skip_step_2 -= 1
                if "history_result" not in locals().keys():
                    with open(record_file_name, "r") as f:
                        history_result = json.load(f)
                equa = history_result[str((var_id, cid))]
            else:
                # Step 2 : for each constant, do sr using spl
                XC = np.append(X, C[cid, :]).reshape(2, spl_test_num + spl_eval_num)
                train_sample = XC[:, :spl_eval_num]
                test_sample = XC[:, spl_eval_num:]
                all_eqs, success_rate, all_times = SVSR(task = f"(x{var_id}, c{cid})", 
                                                        grammars = grammars,
                                                        nt_nodes = nt_nodes,
                                                        num_run = 1,
                                                        train_sample = train_sample,
                                                        test_sample = test_sample,
                                                        transplant_step = 1000,
                                                        num_transplant = 2,
                                                        eta = 0.999)
                result = max(all_eqs, key=lambda x: x[1])[0]
                result = f"({result})"
                result = ReplaceXwithXi(result, f"x{var_id}")
                equa = equa.replace(f'c{cid}', result)
            current_result[str((var_id, cid))] = equa          
        equa = str(sy.simplify(equa))
    with open(record_file_name, "w") as f:
        json.dump(current_result, f)
    return equa