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
        self.xi_t_1t = [f.xi_1_0]
        self.P_t_1t = [f.P_1_0]
        self.xi_t_t = []
        self.P_t_t = []

    def _joseph_form(self, K, H, P_t_1t, R):
        """
        Update P_t_t using Joseph Form. This function handles sequential update
        """
        M = np.eye(self.xi_length) - K.dot(H)
        return M.dot(P_t_1t).dot(M.T) + K.dot(R).dot(K.T)

    def _sequential_update(self, t):
        """
        Sequentially update Kalman Filter at time t
        """
        # LDL 
        Y_t, H_t, D_t, R_t = self._LDL(t)
        xi_t_t = copy(self.xi_t_1t[t])
        P_t_t = copy(self.P_t_1t[t])
        for i in range(self.y_length):
            if np.isnan(Y_t[i]):  # skip missing measurements
                continue
            else:
                H_i = H_t[i].reshape(1, -1)
                D_i = D_t[i].reshape(1, -1)
                sigma2 = R_t[i][i]
                K = P_t_t.dot(H_i.T) / (H_i.dot(P_t_t).dot(H_i.T) + sigma2)
                xi_t_t = xi_t_t + K.dot((Y_t[i] - H_i.dot(xi_t_t) - D_i.dot(self.Xt[t])))
                P_t_t = self._joseph_form(K, H_i, P_t_t, R_t[i][i])
        if t < self.T - 1:
            xi_t1_t = self.Ft[t].dot(xi_t_t) + self.Bt[t].dot(self.Xt[t])
            P_t1_t = self.Ft[t].dot(P_t_t).dot(self.Ft[t].T) + self.Qt[t+1]
        else:
            xi_t1_t = None
            P_t1_t = None

        return xi_t_t, P_t_t, xi_t1_t, P_t1_t, K

    def _LDL(self, t): 
        """
        Transform HMM using LDL methods.
        """
        # Preprocess Rt and Yt if Yt has missing measurements
        is_missing = np.isnan(self.Yt[t])
        if np.any(is_missing):
            for i in is_missing:
                if i:
                    self.Yt[t][i] = 0
                    self.Rt[t][i] = 0
                    self.Rt[t][:, i] = 0
                    self.Ht[t][i] = 0
                    self.Dt[t][i] = 0
        L_t, R_t, _ = linalg.ldl(self.Rt[t])
        L_inv, _ = linalg.lapack.clapack.dtrtri(L_t, lower=True)
        Y_t = L_inv.dot(self.Yt[t])
        H_t = L_inv.dot(self.Ht[t])
        D_t = L_inv.dot(self.Dt[t])

        # Fill nan for missing measurements
        for i in is_missing:
            if i:
                Y_t[i] = np.nan
        return Y_t, H_t, D_t, R_t

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

