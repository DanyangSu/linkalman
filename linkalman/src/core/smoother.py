import numpy as np
from copy import deepcopy as copy

class Smoother(object):

    def __init__(self, f, kf, Xt, Yt):
        """
        Initialize a Kalman Smoother. Refer to linkalman/doc/theory.pdf for details
        """
        self.Ft = f.Ft
        self.Bt = f.Bt
        self.Ht = f.Ht
        self.Dt = f.Dt
        self.Qt = f.Qt
        self.Rt = f.Rt
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
        self.xi2 = []
        self.delta2 = []
        self.chi2 = []
        
    def _smooth(self, t):
        """
        Update Kalman Smoother at time t
        """
        Jt = self.P_t_t[t] * self.Ft[t].T * inv(self.P_t_1t[t+1])
        xi_t_T = self.xi_t_t[t] + Jt * (self.xi_t_T[-1] - self.P_t_1t[t+1])
        P_t_T = self.P_t_t[t] + Jt * (self.P_t_T[-1] - self.P_t_1t[t+1]) * Jt.T
        return xi_t_T, P_t_T

    def _E_delta2(self, t):
        xi2_t_T = self.xi_T[t] * self.xi_T[t].T + self.P_t_T[t]
        xi_t_T_xi_1t_T = self.xi_t_t[t-1].T + (P_t_T[t] + xi_t_T * ((xi_t_T - xi_t_1t).T)) * Jt[t-1].T
        delta2 = xi2_t_T - self.F[t-1] * xi_t_T_xi_1t_T.T - self.Bt[t-1] * x[t-1] * self.xi_t_T[t] \\
                - xi_t_T_xi_1t_T * self.F[t-1].T + self.Ft[t-1] * xi2_t_T[t-1] * self.Ft[t-1].T \\
                + self.Bt[t-1] * self.Xt[t-1] * self.xi_t_T[t-1].T * self.Ft[t-1].T \\
                - self.xi_t_T[t] * self.Xt[t-1].T * self.Bt[t-1].T \\
                + self.Ft[t-1] * self.xi_t_T[t-1] * self.Xt[t-1].T * self.Bt[t-1].T \\
                + self.Bt[t-1] * self.Xt[t-1] * self.Xt[t-1].T * self.Bt[t-1].T
        return delta2

    def _E_chi2(self, t):
        # preprocess 
        
        term1 = self.Yt[t] - self.Dt[t] * self.Xt[t]
        term2 = self.Ht[t] * (self.xi_t_t[t] + self.Jt[t] \\
                * (self.xi_t_T[t+1] - self.xi_t_1t[t+1])) * (self.Yt[t] - self.Dt[t] * self.Xt[t]).T
        term3 = term2.T
        term4 = self.Ht[t] * self.xi2_t_T * self.Ht[t].T
        
        chi2 = term1 + term2 + term3 + term4
        return chi2

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
            
        # Calculate delta2 can chi2 in the E-step of the EM algorithm
        for t in range(self.T):
            self.E_delta2.append(self._E_delta2(t))
            self.E_chi2.append(self._E_chi2(t))

