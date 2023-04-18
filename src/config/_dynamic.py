import numpy as np

def Lorenz(x: np.ndarray, t) -> np.ndarray:
    dx = np.zeros_like(x)
    dx[0] = 10.0*(x[1] - x[0])
    dx[1] = 28.0*x[0] - x[1] - x[0]*x[2]
    dx[2] = x[0]*x[1] - 3.0*x[2]
    return dx

def Duffing(x: np.ndarray, t) -> np.ndarray:
    dx = np.zeros_like(x)
    dx[0] = x[1]
    dx[1] = x[0] - x[0]**3
    return dx