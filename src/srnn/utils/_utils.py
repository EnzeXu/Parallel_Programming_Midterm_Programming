import torch

import numpy as np
import matplotlib.pyplot as plt


class Accumulator:
    '''used to accumulate related metrics'''
    def __init__(self, n: int):
        self.arr = [0] * n
        
    def add(self, *args):
        self.arr = [a + float(b) for a, b in zip(self.arr, args)]
    
    def __getitem__(self, idx: int):
        return self.arr[idx]
    

def set_random_seed(seed):    
    '''set random seed'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)