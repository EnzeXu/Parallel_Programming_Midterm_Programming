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
            'y_num': 1,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 7000,
            'num_transplant': 2,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
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
            'y_num': 1,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 7000,
            'num_transplant': 2,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->(A**2)', 'A->(A**3)'
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
    'Jin-3': { 'target_func': lambda x:  0.2 * x[:, 0]**3 + 1.5 * x[:, 1]**3 - 1.2 * x[:, 1] - 0.5 * x[:, 0],
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'y_num': 1,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 7000,
            'num_transplant': 2,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->(A**2)', 'A->(A**3)'
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
    'Jin-4': { 'target_func': lambda x:  1.5 * np.exp(x[:, 0]) + 5.0 * np.cos(x[:, 1]),
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'y_num': 1,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 7000,
            'num_transplant': 2,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->sin(A)', 'A->cos(A)', 'A->exp(A)'
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
    'Jin-5': { 'target_func': lambda x:  6.0 * np.sin(x[:, 0]) * np.cos(x[:, 1]),
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'y_num': 1,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 7000,
            'num_transplant': 2,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->sin(A)', 'A->cos(A)', 'A->exp(A)'
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
    'Jin-6': { 'target_func': lambda x:  1.35 * x[:, 0] * x[:, 1] + 5.5 * np.sin((x[:, 0] - 1) * (x[:, 1] - 1)),
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'y_num': 1,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 10000,
            'num_transplant': 10,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->sin(A)', 'A->cos(A)', 'A->exp(A)'
            ],
            'nt_nodes': {
                'A'
            },
        },
    },

    'Nguyen-09': { 'target_func': lambda x:  np.sin(x[:, 0]) + np.sin(x[:, 1])**2,
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'y_num': 1,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 7000,
            'num_transplant': 2,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->sin(A)', 'A->cos(A)'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Nguyen-10': { 'target_func': lambda x:  2 * np.sin(x[:, 0]) * np.cos(x[:, 1]),
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'y_num': 1,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 7000,
            'num_transplant': 2,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->sin(A)', 'A->cos(A)'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Nguyen-11': { 'target_func': lambda x:  np.exp(x[:, 1] * np.log(x[:, 0])),
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'y_num': 1,
            'x_range': {
                'x0': (1, 5),
                'x1': (1, 5),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 7000,
            'num_transplant': 2,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->log(A)', 'A->exp(A)'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Nguyen-12': { 'target_func': lambda x:  x[:, 1]**4 - x[:, 1]**3 + 0.5 * x[:, 0] ** 2 - x[:, 0],
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'y_num': 1,
            'x_range': {
                'x0': (-3, 3),
                'x1': (-3, 3),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [2, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 7000,
            'num_transplant': 2,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->sin(A)', 'A->cos(A)'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
}

from gplearn.functions import make_function

def _protected_exponent(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.)

def _power2(x):
    return x**2
def _power3(x):
    return x**3

power2 = make_function(function=_power2, name='power2', arity=1)
power3 = make_function(function=_power3, name='power3', arity=1)
exponential = make_function(function=_protected_exponent, name='exp', arity=1)

gp_cfg = {
    'Jin-1': ("add", "sub", "mul", "div", power2, power3), 
    'Jin-2': ("add", "sub", "mul", "div", power2, power3), 
    'Jin-3': ("add", "sub", "mul", "div", power2, power3), 
    'Jin-4': ("add", "sub", "mul", "div", 'sin', 'cos', exponential), 
    'Jin-5': ("add", "sub", "mul", "div", 'sin', 'cos', exponential), 
    'Jin-6': ("add", "sub", "mul", "div", 'sin', 'cos', exponential), 
}
