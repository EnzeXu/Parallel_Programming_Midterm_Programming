# import sys
import re
import sympy as sy
import numpy as np
from numpy import sin, cos, log, exp
from scipy.optimize import minimize
from spl.spl_train import run_spl
# from mksr.combiner import combiner

# problem specification:
var_num = 3
x_range = {'x0': (-5, 5), 'x1': (-5, 5), 'x2': (-5, 5)}
grammars = ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
            'A->x', 'A->C', 
            'A->exp(A)', 'A->cos(x)', 'A->sin(x)']
nt_nodes = ['A']
np.random.seed(0)

# hyper-parameter:
spl_eval_num = 90
spl_test_num = 10
c_regression_num = 100

def NeuroEval(x_list):               # assume the NN can eval for any x
    # target_equ = "x0**2 * x1 + x0 + 2 * x1"
    # target_equ = "sin(x0) * x1 + 3"
    # target_equ = "x0 * x1 + x0 + 2 * x1 + exp(x1)"
    # target_equ = "sin(x0) * (2.5 * x1 ** 2 + cos(x1)) + x1 + 3"
    target_equ = "x0 * x1 + x0 + 2 * x1 / x2 + x2 * exp(x1)"
    for v in range(var_num):
        locals()[f'x{v}'] = x_list[v]
    return eval(target_equ)
def ReplaceNumbersWithC(text):
    pattern = r"(?<!x)([0-9]*\.[0-9]*|[0-9]+)"
    return re.sub(pattern, "C", text)
def ReplaceXwithXi(text, Xi):
    pattern = r"(?<!e)x"
    return re.sub(pattern, Xi, text)

if __name__ == "__main__":
    equa = '0.0'  # any constant

    for var_id in range(var_num):
        # `equa` consider variable [0, var_id), 
        # now we expand x_{var_id} to the equation.
        print(f"Expand variable x{var_id}")
        print(f"Current equation: {equa}")
        
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
                locals()[f'x{v}'] = cur_x[v]
            def eq_test(c):
                for i in range(len(c)): 
                    locals()['c'+str(i)] = c[i]
                return np.linalg.norm(eval(equa) - f_true, 2)
            c_lst = minimize(eq_test, [1.0] * len(c_lst), method='Powell', tol=1e-6).x.tolist() 
            C[:, tid] = np.array(c_lst)

        # Step 2 : for each constant, do sr using spl
        for cid in range(len(c_lst)):
            XC = np.append(X, C[cid, :]).reshape(2, spl_test_num + spl_eval_num)

            train_sample = XC[:, :spl_eval_num]
            test_sample = XC[:, spl_eval_num:]

            all_eqs, success_rate, all_times = run_spl(task = f"(x{var_id}, c{cid})", 
                                                    grammars = grammars,
                                                    nt_nodes = nt_nodes,
                                                    num_run = 2,
                                                    train_sample = train_sample,
                                                    test_sample = test_sample,
                                                    transplant_step = 200,
                                                    num_transplant = 2,
                                                    eta = 0.999)
            result = max(all_eqs, key=lambda x: x[1])[0]
            result = f"({result})"
            result = ReplaceXwithXi(result, f"x{var_id}")
            equa = equa.replace(f'c{cid}', result)
        equa = str(sy.simplify(equa))
    
    print(f"discovered euqation: {equa}")