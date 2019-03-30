import numpy as np
from copy import deepcopy as copy
from ../core/utils import inv

class Smoother(object):

    def __init__(self, kf):
        """
        Initialize a Kalman Smoother. Refer to linkalman/doc/theory.pdf for details
        """
        self.Ft = kf.Ft
        self.Bt = kf.Bt
        self.Ht = kf.Ht
        self.Dt = kf.Dt
        self.Qt = kf.Qt
        self.Rt = kf.Rt
        self.Yt = kf.Yt
        self.Xt = kf.Xt
        self.xi_t_1t = kf.xi_t_1t
        self.P_t_1t = kf.P_t_1t
        self.xi_t_t = kf.xi_t_t
        self.P_t_t = kf.P_t_t
        self.K_t = kf.K_t
        self.T = kf.T

        # Create output matrices
        self.xi_t_T = [self.xi_t_t[-1]]
        self.P_t_T = [self.P_t_t[-1]]
        
    def _smooth(self, t):
        """
        Update Kalman Smoother at time t
        """
        Jt = self.P_t_t[t] * self.Ft[t].T * inv(self.P_t_1t[t+1])
        xi_t_T = self.xi_t_t[t] + Jt * (self.xi_t_T[-1] - self.P_t_1t[t+1])
        P_t_T = self.P_t_t[t] + Jt * (self.P_t_T[-1] - self.P_t_1t[t+1]) * Jt.T
        return xi_t_T, P_t_T
        

    def __call__(self):
        """
        Run backward smoothing
        """
        for t in reversed(range(self.T - 1)):
            (xi_t_T, P_t_T) = self._smooth(t)
            self.xi_t_T.append(xi_t_T)
            self.P_t_T.append(P_t_T)
        
        # Reverse the order
        self.xi_t_T = list(reversed(self.xi_t_T))
        self.P_t_T = list(reversed(self.P_t_T))
            

