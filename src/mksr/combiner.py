import numpy as np
import sympy as sy
from numpy import sin, cos, log

class combiner:
    def __init__(self, 
                 var_num,
                 pivot,
                 eqs,
                 rtol = 1e-3):
        self.var_num = var_num
        self.pivot = pivot
        self.eqs = [sy.simplify(x) for x in eqs]
        self.rtol = rtol
        self.run()
    def _is_close(self, x, y):
        return np.isclose(x, y, rtol = self.rtol)
    def run(self):
        for v in range(self.var_num):
            globals()[f'x{v}'] = self.pivot[v]
        def merge(e1, e2):
            if not self._is_close(eval(str(e1)), eval(str(e2))):
                raise Exception(f"fail to merge - not equal - {e1} {e2}")
            if e1.is_number:
                return str(e2)
            elif e2.is_number:
                return str(e1)
            elif e1.is_Add and e2.is_Add:
                coe1 = sy.S.Zero
                coe2 = sy.S.Zero
                a1 = list(e1.args)
                a2 = list(e2.args)
                if a1[0].is_number:
                    coe1 = a1[0]
                    a1 = a1[1:]
                if a2[0].is_number:
                    coe2 = a2[0]
                    a2 = a2[1:]
                a1.sort(key=lambda x: eval(str(x)))
                a2.sort(key=lambda x: eval(str(x)))
                exp = []
                while a1 or a2:
                    if a1 and a2 and self._is_close(eval(str(a1[-1])), eval(str(a2[-1]))):
                        exp.append(merge(a1.pop(), a2.pop()))
                    elif not a1 or (a2 and eval(str(a1[-1])) < eval(str(a2[-1]))):
                        part = a2.pop()
                        exp.append(str(part))
                        coe1 -= eval(str(part))
                    else:
                        part = a1.pop()
                        exp.append(str(part))
                        coe2 -= eval(str(part))
                if not self._is_close(float(coe1), 0):
                    exp.append(str(coe1))
                return '+'.join(exp)
            elif e1.is_Mul and e2.is_Mul:
                coe1 = sy.S.One
                coe2 = sy.S.One
                a1 = list(e1.args)
                a2 = list(e2.args)
                if a1[0].is_number:
                    coe1 = a1[0]
                    a1 = a1[1:]
                if a2[0].is_number:
                    coe2 = a2[0]
                    a2 = a2[1:]
                a1.sort(key=lambda x: eval(str(x)))
                a2.sort(key=lambda x: eval(str(x)))
                exp = []
                while a1 or a2:
                    if a1 and a2 and self._is_close(eval(str(a1[-1])), eval(str(a2[-1]))):
                        exp.append(merge(a1.pop(), a2.pop()))
                    elif not a1 or (a2 and eval(str(a1[-1])) < eval(str(a2[-1]))):
                        part = a2.pop()
                        exp.append(str(part))
                        coe1 /= eval(str(part))
                    else:
                        part = a1.pop()
                        exp.append(str(part))
                        coe2 /= eval(str(part))
                if not self._is_close(float(coe1), 1):
                    exp.append(str(coe1))
                return '*'.join(exp)
            elif e1.is_Pow and e2.is_Pos:
                return f"pow({merge(e1.args[0], e2.args[0])}, {merge(e1.args[1], e2.args[1])})"
            elif str(e1.__class__) == 'sin' and str(e2.__class__) == 'sin':
                return f"sin({merge(e1.args[0], e2.args[0])})"
            elif str(e1.__class__) == 'cos' and str(e2.__class__) == 'cos':
                return f"cos({merge(e1.args[0], e2.args[0])})"
            elif str(e1.__class__) == 'log' and str(e2.__class__) == 'log':
                return f"log({merge(e1.args[0], e2.args[0])})"
            else:
                raise Exception("fail to merge")
            
        self.equation = self.eqs[0]
        for i in range(1, len(self.eqs)):
            self.equation = merge(self.equation, self.eqs[i])
            self.equation = sy.simplify(self.equation)