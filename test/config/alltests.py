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
                'x0': (0, 2),
                'x1': (0, 2),
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
            'num_transplant': 4,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.999,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->sin(A)',
            ],
            'nt_nodes': {
                'A'
            },
        },
        'dsr_config': {
            "task" : {
                "task_type" : "regression",
                "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "const"],
                "poly_optimizer_params" : {
                    "degree": 3,
                    "coef_tol": 1e-6,
                    "regressor": "dso_least_squares",
                    "regressor_params": {}
                },
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
            'transplant_step': 10000,
            'num_transplant': 4,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.999,
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
            'transplant_step': 10000,
            'num_transplant': 4,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.999,
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
    'Nguyen-11': { 'target_func': lambda x:  np.exp(x[:, 0] * np.log(x[:, 1])),
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
            'transplant_step': 10000,
            'num_transplant': 4,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.999,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->log(x)', 'A->exp(A)'
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
            'transplant_step': 10000,
            'num_transplant': 4,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.999,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->A**2', 'A->A**3',
            ],
            'nt_nodes': {
                'A'
            },
        }
    },

    'Korns-2': { 'target_func': lambda x:  0.23 + 14.2 * (x[:, 1] + x[:, 2]) / (3 * x[:, 0]),
        'data_num': 10000,
        'common': {
            'x_num': 3,
            'y_num': 1,
            'x_range': {
                'x0': (1, 5),
                'x1': (1, 5),
                'x2': (1, 5),
            },
        },
        'srnn_config':  {
            'epochs': 4000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [3, 128, 256, 128, 1],
            'lr': 1e-4,
            'dropout': 0.2,
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 10000,
            'num_transplant': 3,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Korns-3': { 'target_func': lambda x:  -5.41 + 4.9 * (x[:, 0] - x[:, 1] + x[:, 2] / x[:, 3]) / (3 * x[:, 3]),
        'data_num': 10000,
        'common': {
            'x_num': 4,
            'y_num': 1,
            'x_range': {
                'x0': (-50, 50),
                'x1': (-50, 50),
                'x2': (-50, 50),
                'x3': (-50, 50),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [4, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 10000,
            'num_transplant': 3,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
            ],
            'nt_nodes': {
                'A'
            },
        }
    },    
    'Korns-8': { 'target_func': lambda x:  6.87 + 11 * (7.23 * x[:, 0] * x[:, 1] * x[:, 2]) ** 0.5,
        'data_num': 10000,
        'common': {
            'x_num': 3,
            'y_num': 1,
            'x_range': {
                'x0': (0, 50),
                'x1': (0, 50),
                'x2': (0, 50),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [3, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 10000,
            'num_transplant': 3,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->(A**0.5)'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Korns-9': { 'target_func': lambda x:  x[:, 0]**0.5 * np.exp(x[:, 1]) / (np.log(x[:, 2]) * x[:, 3]**2),
        'data_num': 10000,
        'common': {
            'x_num': 4,
            'y_num': 1,
            'x_range': {
                'x0': (0, 50),
                'x1': (-50, 50),
                'x2': (1, 50),
                'x3': (-50, 50),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [4, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 10000,
            'num_transplant': 3,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->(A**0.5)', 'A->exp(A)', 'A->log(A)',
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Korns-10': { 'target_func': lambda x: 0.81 + 24.3 * (2 * x[:, 0] + 3 * x[:, 1]**2) / (4 * x[:, 2]**3 + 5 * x[:, 3]**4),
        'data_num': 10000,
        'common': {
            'x_num': 4,
            'y_num': 1,
            'x_range': {
                'x0': (-50, 50),
                'x1': (-50, 50),
                'x2': (-50, 50),
                'x3': (-50, 50),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [4, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 10000,
            'num_transplant': 3,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->(A**2)', 'A->(A**3)', 'A->(A**4)',
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Korns-12': { 'target_func': lambda x: 2 - 2.1 * np.cos(9.8 * x[:, 0]) * np.sin(1.3 * x[:, 1]),
        'data_num': 10000,
        'common': {
            'x_num': 2,
            'y_num': 1,
            'x_range': {
                'x0': (-50, 50),
                'x1': (-50, 50),
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
            'num_transplant': 3,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->cos(A)', 'A->sin(A)',
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Korns-13': { 'target_func': lambda x: 32 - 3 * np.tan(x[:, 0]) * np.tan(x[:, 1]) / (np.tan(x[:, 2]) * np.tan(x[:, 3])),
        'data_num': 10000,
        'common': {
            'x_num': 4,
            'y_num': 1,
            'x_range': {
                'x0': (-50, 50),
                'x1': (-50, 50),
                'x2': (-50, 50),
                'x3': (-50, 50),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [4, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 10000,
            'num_transplant': 3,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->tan(A)',
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Korns-14': { 'target_func': lambda x: 22 - 4.2 * (np.cos(x[:, 0]) - np.tan(x[:, 1])) * np.tanh(x[:, 2]) / np.sin(x[:, 3]),
        'data_num': 10000,
        'common': {
            'x_num': 4,
            'y_num': 1,
            'x_range': {
                'x0': (-50, 50),
                'x1': (-50, 50),
                'x2': (-50, 50),
                'x3': (-50, 50),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [4, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 10000,
            'num_transplant': 3,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->cos(A)', 'A->sin(A)', 'A->tan(A)', 'A->tanh(A)'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'Korns-15': { 'target_func': lambda x: 12 - 6 * np.tan(x[:, 0]) / np.exp(x[:, 1]) * (np.log(x[:, 2]) - np.tan(x[:, 3])),
        'data_num': 10000,
        'common': {
            'x_num': 4,
            'y_num': 1,
            'x_range': {
                'x0': (-50, 50),
                'x1': (-50, 50),
                'x2': (1, 50),
                'x3': (-50, 50),
            },
        },
        'srnn_config':  {
            'epochs': 2000,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [4, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 10000,
            'num_transplant': 3,
            'exp_rate': 1/np.sqrt(2),
            'eta': 0.99,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                'A->tan(A)', 'A->exp(A)', 'A->log(A)'
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

def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def _power2(x):
    return x**2
def _power3(x):
    return x**3

power2 = make_function(function=_power2, name='power2', arity=1)
power3 = make_function(function=_power3, name='power3', arity=1)
exponential = make_function(function=_protected_exponent, name='exp', arity=1)
logarithm = make_function(function=_protected_log, name='log', arity=1)

gp_cfg = {
    'Jin-1': ("add", "sub", "mul", "div", power2, power3), 
    'Jin-2': ("add", "sub", "mul", "div", power2, power3), 
    'Jin-3': ("add", "sub", "mul", "div", power2, power3), 
    'Jin-4': ("add", "sub", "mul", "div", 'sin', 'cos', exponential), 
    'Jin-5': ("add", "sub", "mul", "div", 'sin', 'cos', exponential), 
    'Jin-6': ("add", "sub", "mul", "div", 'sin', 'cos', exponential), 
    'Nguyen-09': ("add", "sub", "mul", "div", 'sin', 'cos'), 
    'Nguyen-10': ("add", "sub", "mul", "div", 'sin', 'cos'), 
    'Nguyen-11': ("add", "sub", "mul", "div", logarithm, exponential), 
    'Nguyen-12': ("add", "sub", "mul", "div", power2, power3), 
}
