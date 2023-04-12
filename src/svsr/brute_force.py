import numpy as np
from sympy import simplify
from scipy.optimize import minimize
from numpy import cos, sin, log, exp, tan, sqrt, pi

def run_bfsr(task, 
             grammars,
             train_sample,
             x_range,
             node_limit = 8,
             hash_num = 5):
    termi = []
    unary = []
    binary = []
    for g in grammars:
        assert(g[0:3] == 'A->')
        g = g[3:]
        if g.count('A') == 0:
            termi.append(g)
        elif g.count('A') == 1:
            unary.append(g)
        elif g.count('A') == 2:
            binary.append(g)
    x = np.random.uniform(*x_range, size=(hash_num, 1))
    C = np.random.uniform(*x_range, size=(hash_num, 1))
    print(f"task: {task}, {binary}, {unary}, {termi}")
    equa = [[], []]
    uniq = set()
    E = np.e
    for g in termi:
        g = str((g))
        val = str(eval(str(g)))
        if val not in uniq:
            uniq.add(val)
            equa[1].append(g)
    for num in range(2, node_limit + 1):
        equa.append([])
        for g in unary:
            for eq in equa[num - 1]:
                neq = str((g.replace('A', eq)))
                val = str(eval(str(neq)))
                if val not in uniq:
                    uniq.add(val)
                    equa[-1].append(neq)
        for g in binary:
            for i in range(1, num - 1):
                for eq1 in equa[i]:
                    for eq2 in equa[num - i - 1]:
                        neq = str((g.replace('A', eq1, 1).replace('A', eq2, 1)))
                        val = str(eval(str(neq)))
                        if val not in uniq:
                            uniq.add(val)
                            equa[-1].append(neq)
        print(len(equa[-1]))
        print(equa[-1])