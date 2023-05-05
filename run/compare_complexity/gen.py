from sympy import simplify
import numpy as np
import math
import random

UNARY = 4
BINARY = 4
TERMINAL = 5
N = 50

f = [[0, 0] for i in range(N + 1)]
tr_list = [[[], []] for i in range(N + 1)]
f[0][0] = TERMINAL
f[0][1] = 0
for i in range(1, N + 1):
    f[i][0] = 0
    f[i][1] = f[i - 1][0] * UNARY
    tr_list[i][1].append((f[i - 1][0] * UNARY, i - 1, 0))
    for j in range(i - 1):
        for d1 in range(2):
            for d2 in range(2):
                tr_list[i][d1 | d2].append((f[j][d1] * f[i - 2 - j][d2] * BINARY, j, d1, i - 2 - j, d2))
                f[i][d1 | d2] += f[j][d1] * f[i - 2 - j][d2] * BINARY

def _Sample(n, m):
    if n == 0:
        assert(m == 0)
        return random.choice(["x", "3.14", "2.56", "4.333", "5.222"])
    pr_list = [x[0] / f[n][m] for x in tr_list[n][m]]
    cho_id = np.random.choice(range(len(pr_list)), 1, p = pr_list)[0]
    cho = tr_list[n][m][cho_id]
    if len(cho) == 3:
        return random.choice(['sin', 'cos', 'exp', 'log']) + f"({_Sample(cho[1], cho[2])})"
    else:
        return f"({_Sample(cho[1], cho[2])})" + random.choice(['+', '-', '*', '/']) + f"({_Sample(cho[3], cho[4])})"

def Sample(n):
    sm = f[n][0] + f[n][1]
    pr_list = [f[n][0] / sm, f[n][1] / sm]
    m = np.random.choice((0, 1), 1, p=pr_list)[0]
    return _Sample(n, m)

SAMPLE_TIME = 1000
def Calc(n):
    sm = 0
    cnt = 0
    for i in range(SAMPLE_TIME):
        try:
            ori = Sample(n)
            expr = str(simplify(ori))
            compl = expr.count("cos") + expr.count("sin") + expr.count("exp") + expr.count("log")
            compl += (expr.count('*') + expr.count('/') + expr.count('+') + expr.count('-')) * 2
            if compl <= n:
                cnt += 1
                sm += math.log(f[compl][0] + f[compl][1])
        except:
            pass
    return sm / cnt

for i in range(N + 1):
    print(i, f[i][0] + f[i][1], math.log(f[i][0] + f[i][1]), Calc(i))