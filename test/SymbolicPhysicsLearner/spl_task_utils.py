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
    'Nguyen-1': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'], 

    'Nguyen-2': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'],  

    'Nguyen-3': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'], 

    'Nguyen-4': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'], 

    'Nguyen-5': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                 'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x', 
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'Nguyen-6': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                 'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x', 
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'Nguyen-7': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 
                 'A->x', 'A->cos(x)', 'A->sin(x)', 'A->log(B)', 'A->sqrt(B)', 
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'Nguyen-8': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 
                 'A->log(A)', 'A->sqrt(A)'],

    'Nguyen-09': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 
                 'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'B->B+B', 'B->B-B', 
                 'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                 'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'Nguyen-10': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 
                  'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'B->B+B', 'B->B-B', 
                  'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'Nguyen-11': ['A->x', 'A->y', 'A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
                  'A->exp(A)', 'A->log(B)', 'A->sqrt(B)', 'A->cos(B)', 'A->sin(B)', 
                  'B->B+B', 'B->B-B', 'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'Nguyen-12': ['A->(A+A)', 'A->(A-A)', 'A->A*A', 'A->A/A',
                  'A->x', 'A->x**2', 'A->x**4', 'A->y', 'A->y**2', 'A->y**4', 
                  'A->1', 'A->2', 'A->exp(A)', 
                  'A->cos(x)', 'A->sin(x)', 'A->cos(y)', 'A->sin(y)'], 

    'Nguyen-1c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->x', 'A->x**2', 'A->x**4', 
                  'A->A*C', 'A->exp(x)', 'A->cos(C*x)', 'A->sin(C*x)'], 

    'Nguyen-2c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->x', 'A->x**2', 'A->x**4', 
                  'A->A*C', 'A->exp(x)', 'A->cos(C*x)', 'A->sin(C*x)'], 

    'Nguyen-5c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                  'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x', 'A->A*C',
                  'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1', 'B->B*C'],

    'Nguyen-7c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 'A->A*C',
                  'A->x', 'A->cos(x)', 'A->sin(x)', 'A->log(B)', 'A->sqrt(B)', 
                  'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1', 'B->B*C'],

    'Nguyen-8c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                  'A->x', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C',
                  'A->log(A)', 'A->sqrt(A)'], 

    'Nguyen-9c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->A*C', 
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
    'Nguyen-1': ['A'], 
    'Nguyen-2': ['A'],  
    'Nguyen-3': ['A'],  
    'Nguyen-4': ['A'], 
    'Nguyen-5': ['A', 'B'], 
    'Nguyen-6': ['A', 'B'], 
    'Nguyen-7': ['A', 'B'], 
    'Nguyen-8': ['A'],  
    'Nguyen-09': ['A', 'B'], 
    'Nguyen-10': ['A', 'B'], 
    'Nguyen-11': ['A', 'B'], 
    'Nguyen-12': ['A'], 
    'Nguyen-1c': ['A'], 
    'Nguyen-2c': ['A'], 
    'Nguyen-5c': ['A', 'B'], 
    'Nguyen-7c': ['A', 'B'], 
    'Nguyen-8c': ['A'], 
    'Nguyen-9c': ['A', 'B'], 
    **dict.fromkeys(balldrop_exp, ['A']),
    **dict.fromkeys(['dp_f1', 'dp_f2'], ['A', 'W', 'T', 'S']),
    **dict.fromkeys(['lorenz_x', 'lorenz_y', 'lorenz_z'], ['A'])
}