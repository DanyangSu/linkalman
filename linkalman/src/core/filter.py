import numpy as np
import scipy
from copy import deepcopy as copy

class Filter(object):

    def __init__(self, f):
        """
        Initialize a Kalman Filter. Refer to linkalman/doc/theory.pdf for definition of arguments
        Note that the HMM is assumed to have gone through LDL transformation
        """
        self.Ft = f.Ft
        self.Bt = f.Bt
        self.Ht = f.Ht
        self.Dt = f.Dt
        self.Qt = f.Qt
        self.Rt = f.Rt
        self.Yt = Yt
        self.Xt = Xt
        self.T = len(self.Ft)
        self.xi_length = Ft[0].shape[1]
        self.y_length = Yt[0].shape[0]
        
        # Create output matrices
        self.xi_t_1t = [f.x1_1_0]
        self.P_t_1t = [f.P_1_0]
        self.xi_t_t = []
        self.P_t_t = []

    def _joseph_form(self, K, H, P_t_t, Q, sigma2):
        """
        Update P_t_t using Joseph Form. This function handles sequential update
        """
        M = np.matrix(np.eye(self.xi_length)) - K*H
        return M * P_t_t * M.T + K*K.T*sigma2

    def _sequential_update(self, t):
        """
        Sequentially update Kalman Filter at time t
        """

        # LDL 
        (L_t, R_t) = self.Rt.LDL(t)
        (Y_t, H_t, D_t) = self._LDL(L_t, t)

        xi_t_t = copy(self.xi_t_1t[t])
        P_t_t = copy(self.P_t_1t[t])
        for i in range(self.y_length):
            if self.Yt[i] is None:  # skip missing measurements
                continue
            else:
                H_i = self.Ht[i]
                K = P_t_t * H_i.T / (H_i*P_t_t*H_i.T + sigma2)
                xi_t_t = xi_t_t + K*(self.Yt[i] - H_i * xi_t_t - self.Dt[i] * self.Xt[t])
                P_t_t = self._joseph_form(K, H_i, P_t_t, self.Qt[t], self.Rt[t][i][i])
        if t < self.T - 1:
            xi_t1_t = self.Ft[t] * xi_t_t + self.Bt[t] * self.Xt[t]
            P_t1_t = self.Ft[t] * P_t_t * self.Ft[t].T + self.Qt[t+1]
        else:
            xi_t1_t = None
            P_t1_t = None

        # Restore to original form
        self._restore()

        return xi_t_t, P_t_t, xi_t1_t, P_t1_t, K

    def __call__(self):
        """
        Run forward filtering
        """
        # Filter
        for t in range(self.T):
            (xi_t_t, P_t_t, xi_t1_t, P_t1_t, K_t) = self._sequential_update(t)
            self.xi_t_t.append(xi_t_t)
            self.P_t_t.append(P_t_t)
            self.K_t.append(K_t)
            if t < self.T - 1:
                self.xi_t_1t.append(xi_t1_t)
                self.P_t_1t.append(P_t1_t)

        return ('xi_t_1t': self.xi_t_1t, 'xi_t_t': self.xi_t_t, 'P_t_1t': self.P_t_1t, 'P_t_t': self.P_t_t, 'K_t': self.K_t)

