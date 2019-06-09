import numpy as np
import linkalman
from linkalman.models import BaseConstantModel

class CycleEM(BaseConstantModel):
    """
    Solve HMM with Time series specification in xi. The dimensions of measurement matrices are 
    determined by x and y. 
    x must contain 1 as its first column
    """
    
    def __init__(self):
        self.mod = None
        self.x_dim = None
        self.y_dim = None

    def get_f(self, theta, dim_x, dim_y):
        """
        Create mapping M=f(theta)
        """
        dim_xi = 9
        rho_delta = 1 / (1 + np.exp(theta[0]))
        F = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, rho_delta, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1,-1, -1, -1, -1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0]])
        B = np.zeros([dim_xi, dim_x])
        B[1][0] = (1-rho_delta) * theta[1]
        Q = np.diag(np.append(np.exp([theta[2], theta[3], theta[4]]), [0 for _ in range(dim_xi - 3)]))
        H = np.zeros([dim_y, dim_xi])
        for i in range(dim_y):
            H[i: 0:1] = theta[i + 5]
        theta_xi_start = 5 + dim_y
        xi_1_0 = np.array([[theta[theta_xi_start + i]] for i in range(dim_xi)])
        D_list = []
        idx_D_start = 5 + dim_y + dim_xi
        for i in range(dim_y):
            D_list.append([theta[idx_D_start + j + i * dim_x] for j in range(dim_x)])
        D = np.array(D_list)
        
        # Reverse Cholesky Decomposition to generate R
        idx_R_start = 5 + dim_y + dim_xi + dim_y * dim_x
        R = gen_PSD(theta[idx_R_start:])
        M = {'F': F, 'Q': Q, 'H': H, 'B': B, 'D': D, 'R': R, 'xi_1_0': xi_1_0, 'P_1_0': P_1_0}
        return M