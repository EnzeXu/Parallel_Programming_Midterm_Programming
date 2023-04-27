"""
Test cases config
"""
import numpy as np
_default_mvsr_config = {
    'data_train_num': 180,
    'data_test_num': 20,
    'c_regression_num': 200,
}
TestSettings = {
    'Jin-1': { 'target_func': lambda x:  2.5 * x[:, 0] ** 4 - 1.3 * x[:, 0] ** 3 + 0.5 * x[:, 1] ** 2 - 1.7 * x[:, 1],
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 1500,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 2000,
            'num_transplant': 2,
            'exp_rate': 5/np.sqrt(2),
            'eta': 0.999,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->(A**2)', 'A->(A**3)'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Jin-2': { 'target_func': lambda x:  8.0 * x[:, 0] ** 2 + 8.0 * x[:, 1] ** 3 - 15,
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 1500,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 2000,
            'num_transplant': 2,
            'exp_rate': 5/np.sqrt(2),
            'eta': 0.999,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->(A**2)', 'A->(A**3)'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
}

from gplearn.functions import make_function


def _power2(x):
    return x**2
def _power3(x):
    return x**3

power2 = make_function(function=_power2, name='power2', arity=1)
power3 = make_function(function=_power3, name='power3', arity=1)
gp_jin_cfg = {
    'Jin-1': ("add", "sub", "mul", "div", power2, power3), 
}
