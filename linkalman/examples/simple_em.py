import numpy as np
import linkalman
from linkalman.models import BaseConstantModel as BCM

class SimpleEM(BCM):
    """
    Solve HMM with 1-D xi and y. The dimension of B and D are determined by x.
    The model is suitable for y and xwith low dimensions.
    """

    def __init__(self):
        super.BCM.__init__()

    def get_f(self, theta, dim_x):
        """
        Create mapping M=f(theta)
        """
        F = np.array([[theta[0]]])
        Q = np.array([[np.exp(theta[1])]])
        H = np.array([[theta[2]]])
        R = np.array([[np.exp(theta[3])]])
        xi_1_0 = np.array([[theta[4]]])
        P_1_0 = np.array([[np.exp(theta[5])]])
        B = np.array([theta[i] for i in range(6, 6 + dim_x)])
        D = np.array([theta[i] for i in range(6 + dim_x, 6 + dim_x * 2)])
        
        M = {'F': F, 'Q': Q, 'H': H, 'B': B, 'D': D, 'R': R, 'xi_1_0': xi_1_0, 'P_1_0': P_1_0}
        return M
        
