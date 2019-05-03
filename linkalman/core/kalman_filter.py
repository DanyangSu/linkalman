import numpy as np
from typing import List, Any, Callable, Tuple
import scipy
from copy import deepcopy as copy
from .utils import mask_nan, inv, M_wrap

__all__ = ['Filter']

class Filter(object):
    """
    Given an HMM:

    xi_{t+1} = F_t * xi_t + B_t * x_t + v_t     (v_t ~ N(0, Qt))
    y_t = H_t * xi_t + D_t * x_t + w_t     (w_t ~ N(0, Rt))

    and initial conditions:

    xi_1_0 = E(xi_1) 
    P_1_0 = Cov(xi_1)

    We want to solve:

    xi_t_{t-1} = E(xi_t|Info(t-1))
    P_t_{t-1} = Cov(xi_t|Info(t-1))

    where Info(t) is the information set at time t. With Gaussian
    asusmptions on v_t and w_t, we are able to characterize the
    distribution of the HMM. Refer to doc/theory.pdf for details.
    """

    def __init__(self, Mt: List[np.ndarray]) -> None:
        """
        Initialize a Kalman Filter. Filter take system matrices 
        Mt returns characteristics of the HMM. Note that self.Qt
        correspond to Q_{t+1}.

        Parameters:
        ----------
        Mt : list of system matrices.
        """
        self.Ft = M_wrap(Mt['Ft'])
        self.Bt = M_wrap(Mt['Bt'])
        self.Ht = M_wrap(Mt['Ht'])
        self.Dt = M_wrap(Mt['Dt'])
        self.Qt = M_wrap(Mt['Qt'])
        self.Rt = M_wrap(Mt['Rt'])
        self.xi_length = self.Ft[0].shape[0]
        self.y_length = self.Ht[0].shape[0]
        self.Yt = None
        self.Xt = None
        self.T = len(self.Ft)
        
        # Create output matrices
        self.xi_t_1t = [Mt['xi_1_0']]
        self.P_t_1t = [Mt['P_1_0']]
        self.xi_t_t = []
        self.P_t_t = []
        self.Yt_missing = []  # Keep track of missing measurements

    def __call__(self, Xt: List[np.ndarray], Yt: List[np.ndarray]) -> None:
        """
        Run forward filtering, given input measurements and regressors

        Parameters:
        ----------
        Xt : regressors, must be deterministic and has no missing values
        Yt : measurements, may contain missing values
        """
        self.Yt = Yt
        self.Xt = Xt

        # Filter
        for t in range(self.T):
            xi_t_t, P_t_t, xi_t1_t, P_t1_t = self._sequential_update(t)
            self.xi_t_t.append(xi_t_t)
            self.P_t_t.append(P_t_t)
            self.xi_t_1t.append(xi_t1_t)
            self.P_t_1t.append(P_t1_t)
        
        # Drop xi_T1_T and P_T1_T
        self.xi_t_1t.pop(-1)
        self.P_t_1t.pop(-1)

    def _joseph_form(self, K: np.ndarray, H: np.ndarray, 
            P_t_1t: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Update P_t_t using Joseph Form, which guarantees 
        PSD of the covariance matrix.

        Parameters: 
        ----------
        K : Kalman Gain. 
        H : Ht[t]
        P_t_1t : Cov(xi_t|Info(t-1))
        R : Rt[t]

        Returns:
        ----------
        P_t_t : Cov(xi_t|Info(t))
        """
        M = np.eye(self.xi_length) - K.dot(H)
        P_t_t = M.dot(P_t_1t).dot(M.T) + K.dot(R).dot(K.T)
        return P_t_t

    def _sequential_update(self, t: int) -> Tuple[np.ndarray]:
        """
        Sequentially update Kalman Filter at time t. If Q_t
        if not diagonal, we first transform it to a diagonal
        matrix using LDL transformation. 

        Parameters:
        ----------
        t : time index

        Returns: 
        ----------
        xi_t_t : E(xi_t|Info(t)) 
        P_t_t : Cov(xi_t|Info(t))
        xi_t1_t : E(xi_{t+1}|Info(t))
        P_t1_t : Cov(xi_{t+1}|Info(t))
        """
        # LDL 
        Y_t, H_t, D_t, R_t, is_missing = self._LDL(t)
        self.Yt_missing.append(is_missing)

        # Start sequential updating
        xi_t_t = self.xi_t_1t[t].copy()
        P_t_t = self.P_t_1t[t].copy()

        # Skip missing measurements
        for i in range(self.y_length):
            if is_missing[i]:
                continue
            else:
                H_i = H_t[i].reshape(1, -1)
                D_i = D_t[i].reshape(1, -1)
                sigma2 = R_t[i][i]
                K = P_t_t.dot(H_i.T) / (H_i.dot(P_t_t).dot(H_i.T) + sigma2)
                xi_t_t = xi_t_t + K.dot((Y_t[i] - H_i.dot(xi_t_t) 
                    - D_i.dot(self.Xt[t])))
                P_t_t = self._joseph_form(K, H_i, P_t_t, R_t[i][i])

        # Calculate xi_t1_t and P_t1_t 
        xi_t1_t = self.Ft[t].dot(xi_t_t) + self.Bt[t].dot(self.Xt[t])
        P_t1_t = self.Ft[t].dot(P_t_t).dot(self.Ft[t].T) + self.Qt[t]
        return xi_t_t, P_t_t, xi_t1_t, P_t1_t

    def _LDL(self, t: int) -> Tuple[np.ndarray]: 
        """
        Transform HMM using LDL methods. 

        Parameters:
        ----------
        t : time index of HMM system

        Returns:
        ----------
        Y_t : transformed Yt[t], NaN values are replaced with 0
        H_t : transformed Yt[t], rows that correspond to NaN 
            in Yt[t] are replaced with 0
        D_t : transformed Dt[t], rows that correspond to NaN 
            in Yt[t] are replaced with 0
        R_t : transformed Rt[t], rows and columns that correspond
            to NaN in Yt[t] are replaced with 0, then diagonalized
        is_missing : indicator whether elements in Yt[t] are missing
        """
        # Preprocess Rt and Yt if Yt has missing measurements
        is_missing = np.hstack(np.isnan(self.Yt[t]))
        if np.any(is_missing):
            self.Yt[t] = mask_nan(is_missing, self.Yt[t])
            self.Rt[t] = mask_nan(is_missing, self.Rt[t])
            self.Ht[t] = mask_nan(is_missing, self.Ht[t], dim='row')
            self.Dt[t] = mask_nan(is_missing, self.Dt[t], dim='row')

        # Diagonalize Y_t, H_t, and D_t
        L_t, R_t, L_inv = self.Rt.ldl(t)
        Y_t = L_inv.dot(self.Yt[t])
        H_t = L_inv.dot(self.Ht[t])
        D_t = L_inv.dot(self.Dt[t])
        return Y_t, H_t, D_t, R_t, is_missing

    def get_filtered_y(self) -> List[np.ndarray]:
        """
        Generated filtered Yt. It will also generate
        filtered values for missing measurements.

        Returns:
        ----------
        Yt_filtered : filtered Yt
        """
        Yt_filtered = []
        for t in self.T:
            yt_f = self.Ht[t].dot(self.xi_t_1t[t]) + \
                    self.Dt[t].dot(self.Xt[t])
            Yt_filtered.append(yt_f)
        return Yt_filtered

