balldrop_exp = ['Baseball',
                'Blue Basketball',
                'Green Basketball',
                'Volleyball',
                'Bowling Ball',
                'Golf Ball',
                'Tennis Ball',
                'Whiffle Ball 1',
                'Whiffle Ball 2',
                'Yellow Whiffle Ball',
                'Orange Whiffle Ball']


## production rules for each benchmark for SPL
rule_map = {
    "Jin-1": ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
              'A->x0', 'A->x1', 'A->C',
              'A->(A**2)', 'A->(A**3)'], 
    "Jin-2": ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
              'A->x0', 'A->x1', 'A->C',
              'A->(A**2)', 'A->(A**3)'], 
    "Jin-3": ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
              'A->x0', 'A->x1', 'A->C',
              'A->(A**2)', 'A->(A**3)'], 
    "Jin-4": ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
              'A->x0', 'A->x1', 'A->C',
              'A->sin(A)', 'A->cos(A)', 'A->exp(A)'], 
    "Jin-5": ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
              'A->x0', 'A->x1', 'A->C',
              'A->sin(A)', 'A->cos(A)', 'A->exp(A)'], 
    "Jin-6": ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
              'A->x0', 'A->x1', 'A->C',
              'A->sin(A)', 'A->cos(A)', 'A->exp(A)'], 
    'nguyen-1': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'], 

    'nguyen-2': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'],  

    'nguyen-3': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'], 

    'nguyen-4': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'], 

    'nguyen-5': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                 'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x', 
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'nguyen-6': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                 'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x', 
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'nguyen-7': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 
                 'A->x', 'A->cos(x)', 'A->sin(x)', 'A->log(B)', 'A->sqrt(B)', 
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'nguyen-8': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 
                 'A->log(A)', 'A->sqrt(A)'],

    'nguyen-9': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 
                 'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'B->B+B', 'B->B-B', 
                 'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                 'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'nguyen-10': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 
                  'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'B->B+B', 'B->B-B', 
                  'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'nguyen-11': ['A->x', 'A->y', 'A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
                  'A->exp(A)', 'A->log(B)', 'A->sqrt(B)', 'A->cos(B)', 'A->sin(B)', 
                  'B->B+B', 'B->B-B', 'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'nguyen-12': ['A->(A+A)', 'A->(A-A)', 'A->A*A', 'A->A/A',
                  'A->x', 'A->x**2', 'A->x**4', 'A->y', 'A->y**2', 'A->y**4', 
                  'A->1', 'A->2', 'A->exp(A)', 
                  'A->cos(x)', 'A->sin(x)', 'A->cos(y)', 'A->sin(y)'], 

    'nguyen-1c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->x', 'A->x**2', 'A->x**4', 
                  'A->A*C', 'A->exp(x)', 'A->cos(C*x)', 'A->sin(C*x)'], 

    'nguyen-2c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->x', 'A->x**2', 'A->x**4', 
                  'A->A*C', 'A->exp(x)', 'A->cos(C*x)', 'A->sin(C*x)'], 

    'nguyen-5c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                  'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x', 'A->A*C',
                  'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1', 'B->B*C'],

    'nguyen-7c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 'A->A*C',
                  'A->x', 'A->cos(x)', 'A->sin(x)', 'A->log(B)', 'A->sqrt(B)', 
                  'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1', 'B->B*C'],

    'nguyen-8c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                  'A->x', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C',
                  'A->log(A)', 'A->sqrt(A)'], 

    'nguyen-9c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->A*C', 
                  'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'A->exp(B)', 
                  'B->B*C', 'B->1', 'B->B+B', 'B->B-B',
                  'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    **dict.fromkeys(balldrop_exp, ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->A*C', 
                                   'A->1', 'A->x', 'A->x*x', 'A->x*x*x', 
                                   'A->exp(A)', 
                                   'A->log(C*cosh(A))']),

    **dict.fromkeys(['dp_f1', 'dp_f2'],
                    ['A->C*wdot*cos(x1-x2)', 'A->A+A', 'A->A*A', 'A->C*A', 
                     'A->W', 'W->w1', 'W->w2', 'W->wdot', 'W->W*W', 
                     'A->cos(T)', 'A->sin(T)', 'T->x1', 'T->x2', 'T->T+T', 'T->T-T',
                     'A->sign(S)', 'S->w1', 'S->w2', 'S->wdot', 'A->S+S', 'B->S-S']), 

    **dict.fromkeys(['lorenz_x', 'lorenz_y', 'lorenz_z'], 
                    ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->A*C', 
                     'A->x', 'A->y', 'A->z'])
}


## non-terminal nodes for each task for SPL
ntn_map = {
    'Jin-1': ['A'],
    'Jin-2': ['A'],
    'Jin-3': ['A'],
    'Jin-4': ['A'],
    'Jin-5': ['A'],
    'Jin-6': ['A'],
    'nguyen-1': ['A'], 
    'nguyen-2': ['A'],  
    'nguyen-3': ['A'],  
    'nguyen-4': ['A'], 
    'nguyen-5': ['A', 'B'], 
    'nguyen-6': ['A', 'B'], 
    'nguyen-7': ['A', 'B'], 
    'nguyen-8': ['A'],  
    'nguyen-9': ['A', 'B'], 
    'nguyen-10': ['A', 'B'], 
    'nguyen-11': ['A', 'B'], 
    'nguyen-12': ['A'], 
    'nguyen-1c': ['A'], 
    'nguyen-2c': ['A'], 
    'nguyen-5c': ['A', 'B'], 
    'nguyen-7c': ['A', 'B'], 
    'nguyen-8c': ['A'], 
    'nguyen-9c': ['A', 'B'], 
    **dict.fromkeys(balldrop_exp, ['A']),
    **dict.fromkeys(['dp_f1', 'dp_f2'], ['A', 'W', 'T', 'S']),
    **dict.fromkeys(['lorenz_x', 'lorenz_y', 'lorenz_z'], ['A'])
}