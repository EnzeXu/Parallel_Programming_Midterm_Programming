"""Solve mksr problem use a given svsr method.
"""
import numpy as np
import os
from scipy.optimize import minimize
from numpy import sin, cos, log, exp
import sympy as sy
import re


def _replace_number_with_capital_c(text: str) -> str:
    """replace number with capital c

    For 'x**2', the 2 should not be regard as constant.
    Reserve all the space.

    examples: 
        >>> _replace_number_with_capital_c("x0+x1+x4**2+x2+1e-9+10.0 + 11")
        'x0+x1+x4**2+x2+C+C + C'
    """
    # TODO(cxt): a more detailed protection rule
    text = text.replace("**2", "**TWO")  # protect ^2
    pattern_scitific_repre = r"(?<!x)([0-9]*\.[0-9]*|[0-9]+)[eE][+-]?[0-9]+"
    text = re.sub(pattern_scitific_repre, "C", text)
    pattern_const = r"(?<!x)([0-9]*\.[0-9]*|[0-9]+)([eE][+-]?[0-9]+)?"
    text = re.sub(pattern_const, "C", text)
    text = text.replace("**TWO", "**2")  # protect ^2
    return text


def _replace_x_with_xi(text: str, Xi: str) -> str:
    """replace all x with xi

    examples:
        >>> _replace_x_with_xi("x+ x +exp(10)", "x1")
        'x1+ x1 +exp(10)'
    """
    pattern = r"(?<!e)x"
    return re.sub(pattern, Xi, text)


class MKSR:
    def __init__(
        self,
        func_name: str,
        x_num: int,
        x_range: dict,
        neuro_eval: callable,
        svsr_method: callable,
        data_train_num: int = 90,
        data_test_num: int = 10,
        c_regression_num: int = 100,
        random_seed: int = 10,
        **kwargs,
    ) -> None:
        """Init the MKSR.

        Args:
            func_name:        the name of the equation to be discovered.
            var_num:          the number of the variable, named as x0, x1, ...
            x_range:          range of x. e.g. x_range['x0'] = (-1, 1)
            grammars:         grammar list. e.g. 'A->A+A'
            nt_nodes:         the set of all non terminal nodes.
            neuro_eval:       the function provided by neuro network to do the evaluation.
            svsr_method:             the beneath method to do the single variable regression.

        Returns: 
            The expression in str format.
        """
        self.func_name = func_name
        self.x_num = x_num
        self.x_range = x_range
        self.neuro_eval = neuro_eval
        self.svsr_method = svsr_method
        self.data_train_num = data_train_num
        self.data_test_num = data_test_num
        self.c_regression_num = c_regression_num
        self.np_rng = np.random.default_rng(seed=random_seed)
        self.equation = ""
        self.kwargs = kwargs

    def __repr__(self):
        return repr(self.equation)

    def _generate_data(
            self,
            equation: str,
            const_num: int,
            var_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """generate data for svsr using simpy.optimize.minimize.

        Returns:
            X : shape = (1, test_num)       the variable x{var_id}'s value
            C : shape = (c_count, test_num) the corresponding constant value
        """
        data_num = self.data_test_num + self.data_train_num
        pivot = np.empty(self.x_num)
        for vid in range(var_id + 1, self.x_num):
            pivot[vid] = self.np_rng.uniform(*self.x_range[f"x{vid}"])
        C = np.empty((const_num, data_num))
        X = np.empty((1, data_num))
        for tid in range(data_num):
            cur_x = np.empty((self.x_num, self.c_regression_num))
            for vid in range(self.x_num):
                if vid < var_id:
                    cur_x[vid, :] = self.np_rng.uniform(
                        *self.x_range[f"x{vid}"], self.c_regression_num)
                elif vid == var_id:
                    cur_x[vid, :] = X[0, tid] = self.np_rng.uniform(
                        *self.x_range[f"x{vid}"])
                else:
                    cur_x[vid, :] = pivot[vid]
            f_true = self.neuro_eval(cur_x)
            for v in range(self.x_num):
                globals()[f'x{v}'] = cur_x[v]

            def eq_test(c):
                for i in range(len(c)):
                    locals()['c'+str(i)] = c[i]
                return np.linalg.norm(eval(equation) - f_true, 2)
            c_list = minimize(eq_test, [1.0] * const_num,
                              method='Powell', tol=1e-6).x.tolist()
            C[:, tid] = np.array(c_list)
        return X, C

    def run(self) -> None:
        if not os.path.exists(f"results/{self.func_name}/mksr"):
            os.makedirs(f"results/{self.func_name}/mksr")
        record_equation_file_prefix = f"results/{self.func_name}/mksr/equation"
        record_data_file_prefix = f"results/{self.func_name}/mksr/data"
        equation = '0.0'  # any constant
        for var_id in range(self.x_num):
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
                X, C = self._generate_data(equation=equation, const_num=c_count, var_id=var_id)
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
                        2, self.data_test_num + self.data_train_num)
                    train_sample = XC[:, :self.data_train_num]
                    test_sample = XC[:, self.data_train_num:]
                    result = self.svsr_method(
                        task=f"(x{var_id}, c{cid})",
                        train_sample=train_sample,
                        test_sample=test_sample,
                        x_range=self.x_range[f"x{var_id}"],
                        np_rng=self.np_rng,
                        **self.kwargs['svsr_config'])
                    result = f"({result})"
                    result = _replace_x_with_xi(result, f"x{var_id}")
                    equation = equation.replace(f'c{cid}', result)
                    with open(save_name, "w") as f:
                        f.write(equation)
            equation = str(sy.simplify(equation))
        self.equation = equation

if __name__ == "__main__":
    import doctest
    doctest.testmod()