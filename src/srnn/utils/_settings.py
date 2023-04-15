from torch import sin, cos, log, exp
from ._dynamic import *

FuncSettings = {
    'test1': {
        'func': lambda x: x[:, 0] * x[:, 1] + x[:, 0] + 2 * x[:, 1] / x[:, 2] + x[:, 2] * exp(x[:, 1]),
        'type': 'normal',
        'sample_times': 3000,
        'x_len': 3,
        'model': {
            'MLP': {
                'layer_size': [3, 64, 64, 64, 64, 64, 64, 1]
            },
            'Taylor': {
                'in_features':      3, 
                'out_features':     1, 
                'order':            3, 
                'rank':             8
            }
        }
    },
    
    'test2': {
        'func': lambda x: x[:, 0]**2 * x[:, 1] + x[:, 0] + 2 * x[:, 1],
        'type': 'normal',
        'sample_times': 500,
        'x_len': 2,
        'model': {
            'MLP': {
                'layer_size': [2, 128, 128, 1]
            },
            'Taylor': {
                'in_features':      2, 
                'out_features':     1, 
                'order':            1, 
                'rank':             8
            }
        }
    },
    
    'test3': {
        'func': lambda x: sin(x[:,0]) * x[:,1] + 3,
        'type': 'normal',
        'sample_times': 1000,
        'x_len': 2,
        'model': {
            'MLP': {
                'layer_size': [2, 128, 128, 1]
            },
            'Taylor': {
                'in_features':      2, 
                'out_features':     1, 
                'order':            3, 
                'rank':             8
            }
        }
    },
    
    'test4': {
        'func': lambda x: sin(x[:,0]) * (2.5 * x[:,1] ** 2 + cos(x[:,1])) + x[:,1] + 3,
        'type': 'normal',
        'sample_times': 1000,
        'x_len': 2,
        'model': {
            'MLP': {
                'layer_size': [2, 128, 128, 1]
            },
            'Taylor': {
                'in_features':      2, 
                'out_features':     1, 
                'order':            3, 
                'rank':             8
            }
        }
    },
    
    'Lorenz': {
        'func': Lorenz,
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
        'func': Duffing,
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