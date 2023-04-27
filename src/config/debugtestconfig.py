from torch import sin, cos, log, exp
import numpy as np

def _Lorenz(x: np.ndarray, t) -> np.ndarray:
    dx = np.zeros_like(x)
    dx[0] = 10.0*(x[1] - x[0])
    dx[1] = 28.0*x[0] - x[1] - x[0]*x[2]
    dx[2] = x[0]*x[1] - 3.0*x[2]
    return dx

def _Duffing(x: np.ndarray, t) -> np.ndarray:
    dx = np.zeros_like(x)
    dx[0] = x[1]
    dx[1] = x[0] - x[0]**3
    return dx

TestSettings = {
    'test1': {
        'func': lambda x: x[:, 0] * x[:, 1] + x[:, 0] + 2 * x[:, 1] / x[:, 2] + x[:, 2] * exp(x[:, 1]),
        'type': 'normal',
        'sample_times': 3000,
        'x_num': 3,
        'x_range': {
            'x0': (-5, 5),
            'x1': (-5, 5),
            'x2': (-5, 5),
        },
        'model': {
            'MLP': {
                'layer_size': [3, 64, 64, 64, 64, 64, 64, 1],
            },
            'Taylor': {
                'in_features':      3, 
                'out_features':     1, 
                'order':            3, 
                'rank':             8
            }
        },
        'mlp_fit_times': 10000,
        'data_train_num': 180,
        'data_test_num': 20,
        'c_regression_num': 200,
        'svsr_config': {
            'method': 'spl',
            'num_run': 1,
            'transplant_step': 1000,
            'num_transplant': 2,
            'eta': 0.99,
            'grammars': [
                'A->A+A', 'A->A-A', 'A->A*A', 
                'A->A/A', 'A->x', 'A->C',
                'A->exp(x)', 'A->cos(x)', 'A->sin(x)'
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
    
    'test2': {
        'func': lambda x: x[:, 0]**2 * x[:, 1] + x[:, 0] + 2 * x[:, 1],
        'type': 'normal',
        'sample_times': 1000,
        'x_num': 2,
        'x_range': {
            'x0': (-5, 5),
            'x1': (-5, 5),
        },
        'model': {
            'MLP': {
                'layer_size': [2, 128, 128, 1],
            },
            'Taylor': {
                'in_features':      2, 
                'out_features':     1, 
                'order':            1, 
                'rank':             8
            }
        },
        'mlp_fit_times': 4000,
        'data_train_num': 180,
        'data_test_num': 20,
        'c_regression_num': 200,
        'svsr_config': {
            'method': 'spl',
            'num_run': 1,
            'transplant_step': 1000,
            'num_transplant': 2,
            'eta': 0.99,
            'grammars': [
                'A->A+A', 'A->A-A', 'A->A*A', 
                'A->A/A', 'A->x', 'A->C',
                'A->exp(x)', 'A->cos(x)', 'A->sin(x)'
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
    
    'test3': {
        'func': lambda x: sin(x[:,0]) * x[:,1] + 3,
        'type': 'normal',
        'sample_times': 1000,
        'x_num': 2,
        'x_range': {
            'x0': (-5, 5),
            'x1': (-5, 5),
        },
        'model': {
            'MLP': {
                'layer_size': [2, 128, 128, 1],
            },
            'Taylor': {
                'in_features':      2, 
                'out_features':     1, 
                'order':            3, 
                'rank':             8
            }
        },
        'mlp_fit_times': 4000,
        'data_train_num': 180,
        'data_test_num': 20,
        'c_regression_num': 200,
        'svsr_config': {
            'method': 'spl',
            'num_run': 1,
            'transplant_step': 1000,
            'num_transplant': 2,
            'eta': 0.99,
            'grammars': [
                'A->A+A', 'A->A-A', 'A->A*A', 
                'A->A/A', 'A->x', 'A->C',
                'A->exp(x)', 'A->cos(x)', 'A->sin(x)'
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
    
    'test4': {
        'func': lambda x: sin(x[:,0]) * (2.5 * x[:,1] ** 2 + cos(x[:,1])) + x[:,1] + 3,
        'type': 'normal',
        'sample_times': 1000,
        'x_num': 2,
        'x_range': {
            'x0': (-5, 5),
            'x1': (-5, 5),
        },
        'model': {
            'MLP': {
                'layer_size': [2, 128, 128, 1],
            },
            'Taylor': {
                'in_features':      2, 
                'out_features':     1, 
                'order':            3, 
                'rank':             8
            }
        },
        'mlp_fit_times': 4000,
        'data_train_num': 180,
        'data_test_num': 20,
        'c_regression_num': 200,
        'svsr_config': {
            'method': 'spl',
            'num_run': 1,
            'transplant_step': 1000,
            'num_transplant': 2,
            'eta': 0.99,
            'grammars': [
                'A->A+A', 'A->A-A', 'A->A*A', 
                'A->A/A', 'A->x', 'A->C',
                'A->exp(x)', 'A->cos(x)', 'A->sin(x)'
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
  
    'test5': {
        'target_func': lambda x: x[:, 0] / (1 + x[:, 1]**2) + 2,
        'data_num' : 8000,
        'common': {
            'x_num': 2,
            'x_range': {
                'x0': (-5, 5),
                'x1': (-3, 3),
            },
        },
        'srnn_config': {
            'epochs': 1500,
            'layer': 'Linear',
            'activation': 'ReLU',
        },
        'mvsr_config': {
            'data_train_num': 180,
            'data_test_num': 20,
            'c_regression_num': 200,
        },
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 2000,
            'num_transplant': 2,
            'exp_rate': 5/np.sqrt(2),
            'eta': 0.999,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)', 
                'A->(A/A)', 'A->x', 'A->C',
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
    
    'Lorenz': {
        'func': _Lorenz,
        'type': 'ode',
        'dt': 0.01,
        't_f': 1000,
        'x_range': [[-8, -6], [-7, -5], [-20, -10]],
        'model': {
            'MLP': {
                'layer_size': [3, 64, 64, 3]
            },
            'Taylor': {
                'in_features':      3, 
                'out_features':     3, 
                'order':            2, 
                'rank':             16
            }
        }
    },
    
    'Duffing': {
        'func': _Duffing,
        'type': 'ode',
        'dt': 0.01,
        't_f': 1000,
        'x_range': [[-1.1, 1.1], [-1.1, 1.1]],
        'model': {
            'MLP': {
                'layer_size': [2, 64, 64, 2]
            },
            'Taylor': {
                'in_features':      2, 
                'out_features':     2, 
                'order':            3, 
                'rank':             16
            }
        }
    },
}