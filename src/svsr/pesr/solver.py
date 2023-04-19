"""
Priority Extention Symbolic Regrssion
"""

import numpy as np
from sympy import simplify
from scipy.optimize import minimize
from numpy import cos, sin, log, exp, tan, sqrt, pi
from sklearn.metrics import mean_squared_error
from collections import namedtuple


class PESR:
    def __init__(
        self,
        task,
        train_sample,
        test_sample,
        grammars,
        max_allowed_number,
        norm_threshold,
        x_range,
        c_limit,
        eta, 
        node_limit,
        hash_num,
        np_rng,
        **kwargs,
    ) -> None:
        self.task = task
        samples = np.append(train_sample, test_sample, axis=1)
        self.x_samples = samples[0, :]
        self.y_samples = samples[1, :]
        self.grammars = grammars
        self.max_allowed_number = max_allowed_number
        self.norm_threshold = norm_threshold
        self.x_range = x_range
        self.c_limit = c_limit
        self.node_limit = node_limit
        self.hash_num = hash_num
        self.np_rng = np_rng
        self.eta = eta
        self.termi = []
        self.unary = []
        self.binary = []
        for g in grammars:
            assert (g[0:3] == 'A->')
            g = g[3:]
            if g.count('A') == 0:
                self.termi.append(g)
            elif g.count('A') == 1:
                self.unary.append(g)
            elif g.count('A') == 2:
                self.binary.append(g)
        self.hash_x = self.np_rng.uniform(*x_range, size=(hash_num, 1))
        self.hash_c = self.np_rng.uniform(*x_range, size=(hash_num, 1))
        self.equation = ""
        self.result = []
        
    def __str__(self):
        return self.equation
    
    def _hash_eval(self, eq):
        return eval(str(eq).replace('x', 'self.hash_x').replace('C', 'self.hash_c'))
        
    def _get_reward(self, coe):
        sli = self.np_rng.choice(len(self.x_samples), 10)
        def reward(equa):
            eq = equa.expr      
            f_true = self.y_samples[sli] 
            c_count = eq.count('C')
            if c_count == 0:       ## no numerical values
                x = self.x_samples[sli]
                f_pred = eval(eq)
            elif c_count >= self.c_limit:    ## discourage over complicated numerical estimations
                return 0
            else:                  ## with numerical values: coefficient estimationwith Powell method
                c_lst = ['c'+str(i) for i in range(c_count)]
                for c in c_lst: 
                    eq = eq.replace('C', c, 1)

                def eq_test(c):
                    x = self.x_samples[sli]
                    for i in range(len(c)): globals()['c'+str(i)] = c[i]
                    return np.linalg.norm(eval(eq) - f_true, 2)
                try:
                    c_lst = minimize(eq_test, [1.0] * len(c_lst), method='Powell', tol=1e-6).x.tolist() 
                    c_lst = [np.round(x, 4) if abs(x) > 1e-2 else 0 for x in c_lst]
                    eq_est = eq
                    for i in range(len(c_lst)):
                        eq_est = eq_est.replace('c'+str(i), str(c_lst[i]), 1)
                    eq = eq_est.replace('+-', '-')
                    x = self.x_samples[sli]
                    f_pred = eval(eq)
                except:
                    f_pred = 0
            r = coe / (1 + np.linalg.norm(f_pred - f_true, 2))
            if r > 1 - self.norm_threshold:
                self.result.append((r, eq))
                # print(self.result[-1])
            return r
        return reward

    def run(self) -> None:
        print(f"task: {self.task}, {self.binary}, {self.unary}, {self.termi}")
        equa = [[] for _ in range(self.node_limit + 1)]
        uniq = set()
        Equation = namedtuple("Equa", ["expr", "has_trigo"])
        for g in self.termi:
            g = str((g))
            val = str(self._hash_eval(g))
            if val not in uniq:
                uniq.add(val)
                equa[1].append(Equation(expr=g, has_trigo=0))
        for node_num in range(2, self.node_limit + 1):
            for g in self.unary:
                for eq in equa[node_num - 1]:
                    if g[:3] in ['sin', 'cos'] and eq.has_trigo:
                        continue
                    nexpr = str(g.replace('A', eq.expr))
                    val = str(self._hash_eval(nexpr))
                    if val not in uniq:
                        uniq.add(val)
                        equa[node_num].append(Equation(nexpr, eq.has_trigo or g[:3] in ['sin', 'cos']))
            for g in self.binary:
                for i in range(1, node_num - 1):
                    for eq1 in equa[i]:
                        for eq2 in equa[node_num - i - 1]:
                            nexpr = str(g.replace('A', eq1.expr, 1).replace('A', eq2.expr, 1))
                            nexpr = f"({nexpr})"
                            val = str(self._hash_eval(nexpr))
                            if val not in uniq:
                                uniq.add(val)
                                equa[node_num].append(Equation(nexpr, eq1.has_trigo or eq2.has_trigo))
            print(node_num, len(equa[node_num]))
            equa[node_num].sort(key=self._get_reward(self.eta ** node_num), reverse=1)
            if len(equa[node_num]) > self.max_allowed_number:
                equa[node_num] = equa[node_num][:self.max_allowed_number]
            # sli = self.np_rng.choice(len(equa[node_num]), 10)
            # print(np.array(equa[node_num])[sli])
        self.equation = max(self.result)[1]
                


def run_pesr(**kwargs) -> str:
    pesr = PESR(**kwargs)
    pesr.run()
    return str(pesr)
