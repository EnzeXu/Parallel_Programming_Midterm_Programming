import numpy as np
import torch
from scipy.integrate import odeint

class ODE:
    def __init__(self, x_range, params, dt) -> None:
        self.x_range = np.asarray(x_range, dtype = np.float32)
        self.params = params
        self.x_len = len(x_range)
        self.dt = dt
    
    def _equation(self, x, t) -> np.ndarray:
        '''
        ODE Equations. you should make sure that the parameters
        have ``x`` and ``t``.
        '''
        raise NotImplementedError
    
    def solve(self, t_f):
        '''
        solve the ODE equations using ``odeint``.
        '''
        tspan = np.linspace(0, t_f, int(t_f / self.dt))
        x0 = (self.x_range[:, 1] - self.x_range[:, 0]) * \
            np.random.rand(self.x_len) + self.x_range[:, 0]
        x = odeint(self._equation, x0, tspan)
        return x
    

class Fluid(ODE):
    def __init__(self, x_range: np.ndarray, params, dt) -> None:
        super().__init__(x_range, params, dt)
        
    def _equation(self, x, t) -> np.ndarray:
        mu, omega, lamb, A = self.params
        dx = np.zeros_like(x)
        dx[0] = mu * x[0] - omega * x[1] + A * x[0] * x[2]
        dx[1] = omega * x[0] + mu * x[1] + A * x[1] * x[2]
        dx[2] = -lamb * (x[2] - x[0]**2 - x[1]**2)
        return dx

    
class Duffing(ODE):
    def __init__(self, x_range: np.ndarray, params, dt) -> None:
        super().__init__(x_range, params, dt)
    
    def _equation(self, x, t) -> np.ndarray:
        alpha, beta = self.params
        dx = np.zeros_like(x)
        dx[0] = x[1]
        dx[1] = alpha*x[0] - beta*x[0]**3
        return dx
    

class PredatorPrey(ODE):
    def __init__(self, x_range: np.ndarray, params, dt) -> None:
        super().__init__(x_range, params, dt)
    
    def _equation(self, x, t) -> np.ndarray:
        alpha, beta, delta, gamma = self.params
        dx = np.zeros_like(x)
        dx[0] = alpha*x[0] - beta*x[0]*x[1]
        dx[1] = delta*x[0]*x[1] - gamma*x[1]
        return dx
    
    
class Lorenz(ODE):
    def __init__(self, x_range: np.ndarray, params, dt) -> None:
        super().__init__(x_range, params, dt)
    
    def _equation(self, x, t) -> np.ndarray:
        P, Ra, b = self.params
        dx = np.zeros_like(x)
        dx[0] = P*(x[1] - x[0])
        dx[1] = Ra*x[0] - x[1] - x[0]*x[2]
        dx[2] = x[0]*x[1] - b*x[2]
        return dx
    
    
class Toggle(ODE):
    '''
    Paper: https://www.nature.com/articles/35002131#Sec7
    Parameter: https://2013.igem.org/Team:Duke/Modeling/Kinetic_Model
    '''
    def __init__(self, x_range, params, dt) -> None:
        super().__init__(x_range, params, dt)
        
    def _equation(self, x, t) -> np.ndarray:
        alpha_1, alpha_2, beta, gamma = self.params
        dx = np.zeros_like(x)
        dx[0] = alpha_1 / (1 + x[1] ** beta) - x[0]
        dx[1] = alpha_2 / (1 + x[0] ** gamma) - x[1]
        return dx
    
def generate_dynamcis_data(
    dynamic_mode: str,
    dynamic_kwargs: dict,
    traj_num: int = 5000,
    traj_points: int = 10,
):
    dynamic: ODE = eval(dynamic_mode)(**dynamic_kwargs)
    X, DX = [], []
    for _ in range(traj_num):
        x = dynamic.solve((traj_points + 2) * dynamic_kwargs['dt'])
        X.append(x[1:-1])
        DX.append((x[2:] - x[:-2]) / 2 / dynamic_kwargs['dt'])
    X = torch.as_tensor(
        np.concatenate(X, axis = 0), dtype = torch.float32
    )
    Y = torch.as_tensor(
        np.concatenate(DX, axis = 0), dtype = torch.float32
    )
    return X, Y