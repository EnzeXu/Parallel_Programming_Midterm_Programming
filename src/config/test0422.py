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
    'test1': { 'target_func': lambda x: x[:, 0] / (1 + x[:, 1]**2) + 2,
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'x_range': {
                'x0': (-5, 5),
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
                'A->(A/A)', 'A->x', 'A->C'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'test2': { 'target_func': lambda x: np.sqrt(x[:, 2]**2 + x[:, 3] ** 2 - 2 * x[:, 2] * x[:, 3] * np.cos(x[:, 0] - x[:, 1])),
        'data_num': 8000,
        'common': {
            'x_num': 4,
            'x_range': {
                'x0': (-5, 5),
                'x1': (-5, 5),
                'x2': (-5, 5),
                'x3': (-5, 5),
            },
        },
        'srnn_config': {
            'epochs': 1500,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [4, 128, 256, 128, 1],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 20000,
            'num_transplant': 2,
            'exp_rate': 5/np.sqrt(2),
            'eta': 0.999,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C',
                # 'A->(x*x + A)**0.5',
                'A->(A**0.5)'
                'A->cos(A)', 'A->sin(A)',
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
    'test3': {
        # x[:, 0]: x
        # x[:, 1]: u
        # x[:, 2]: t
        'target_func': lambda x: (x[:, 0] - x[:, 1] * x[:, 2]) / np.sqrt(1 - x[:, 1]**2 / 1e16),
        'data_num': 8000,
        'common': {
            'x_num': 3,
            'x_range': {
                'x0': (-1e9, 1e9),
                'x1': (-1e8, 1e8),
                'x2': (10, 50),
            },
        },
        'srnn_config': {
            'epochs': 1500,
            'layer': 'Linear',
            'activation': 'ReLU',
            'layer_size': [3, 128, 256, 128, 1],
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
                'A->(A**0.5)'
                'A->cos(A)', 'A->sin(A)',
            ],
            'nt_nodes': {
                'A'
            },
        },
    },
    'test4': { 'target_func': lambda x: x[:, 0] ** 4 - x[:, 0] ** 3 + 0.5 * x[:, 1] ** 2 - x[:, 1],
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'x_range': {
                'x0': (-5, 5),
                'x1': (-5, 5),
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
            'transplant_step': 5000,
            'num_transplant': 4,
            'exp_rate': 5/np.sqrt(2),
            'eta': 0.999,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'test5': { 'target_func': lambda x: x[:, 1] ** 4 - x[:, 1] ** 3 + 0.5 * x[:, 0] ** 2 - x[:, 0],
        'data_num': 8000,
        'common': {
            'x_num': 2,
            'x_range': {
                'x0': (0, 1),
                'x1': (0, 1),
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
            'transplant_step': 5000,
            'num_transplant': 4,
            'exp_rate': 5/np.sqrt(2),
            'eta': 0.999,
            'grammars': [
                'A->(A+A)', 'A->(A-A)', 'A->(A*A)',
                'A->(A/A)', 'A->x', 'A->C'
            ],
            'nt_nodes': {
                'A'
            },
        }
    },
    'toggle': {    
        'traj_num': 10000,
        'traj_points': 10,
        'params': [4, 4, 3, 3],
        'dt': 0.01,        
        'common': {
            'x_num': 2,
            'y_num': 2,
            'x_range': {
                'x0': (0, 2),
                'x1': (0, 2),
            },
        },
        'srnn_config':  {
            'epochs': 10000,
            'layer': 'Taylor',
            'activation': 'Tanh',
            'layer_size': [2, 128, 256, 128, 2],
        },
        'mvsr_config': _default_mvsr_config,
        'svsr_config': {
            'num_run': 1,
            'transplant_step': 5000,
            'num_transplant': 2,
            'exp_rate': 5/np.sqrt(2),
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
}
