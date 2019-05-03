import numpy as np
from typing import List, Any, Callable, Tuple
from copy import deepcopy as copy
from scipy import linalg
from .utils import inv
from . import Filter

__all__ = ['Smoother']

class Smoother(object):
    """
    Given a filtered object, Smoother returns smoothed state estimation.
    Given an HMM:

    xi_{t+1} = F_t * xi_t + B_t * x_t + v_t     (v_t ~ N(0, Qt))
    y_t = H_t * xi_t + D_t * x_t + w_t     (w_t ~ N(0, Rt))

    and initial conditions:

    xi_1_0 = E(xi_1) 
    P_1_0 = Cov(xi_1)

    We want to solve:

    xi_t_{t-1} = E(xi_t|Info(T))
    P_t_{t-1} = Cov(xi_t|Info(T))

    where Info(t) is the information set at time t, and T = max(t). 
    Using forward filtering then backward smoothing, we are able to 
    characterize the distribution of the HMM based on the full 
    information set up to T. Refer to doc/theory.pdf for details.
    """

    def __init__(self) -> None:
        """
        Initialize a Kalman Smoother. self.delta2 and self.chi2 
        are used for EM algorithms later.
        """
        self.xi_t_T = []
        self.P_t_T = []
        self.xi2_t_T = []
        self.xi_t_xi_1t_T = []
        self.delta2 = []
        self.chi2 = []
        
    def __call__(self, kf: Filter) -> None:
        """
        Run backward smoothing. 

        Parameters: 
        ----------
        kf : a Filter instance
        """
        # Include filtered results
        self.__dict__.update(kf.__dict__)

        # Start backward smoothing
        for t in reversed(range(self.T)):
            xi_t_T, P_t_T, xi2_t_T, xi_t1_xi_t_T = self._smooth(t)
            self.xi_t_T.append(xi_t_T)
            self.P_t_T.append(P_t_T)
            self.xi2_t_T.append(xi2_t_T)
            self.xi_t_xi_1t_T.append(xi_t1_xi_t_T)

        # match index for xi_t_xi_1t_T
        self.xi_t_xi_1t_T.append(None)
        self.xi_t_xi_1t_T.pop(0)

        # Reverse the order of t to restore chronological order
        self.xi_t_T = list(reversed(self.xi_t_T))
        self.P_t_T = list(reversed(self.P_t_T))
        self.xi2_t_T = list(reversed(self.xi2_t_T))
        self.xi_t_xi_1t_T = list(reversed(self.xi_t_xi_1t_T))
            
        # Calculate delta2 can chi2 in the E-step of the EM algorithm
        for t in range(self.T):
            self.delta2.append(self._E_delta2(t))
            self.chi2.append(self._E_chi2(t))

    def _smooth(self, t: int) -> Tuple[np.ndarray]:
        """
        Update Kalman Smoother at time t. Refer to doc/theory.pdf 
        for details on the notation of each variables.

        Parameters:
        ----------
        t : time index

        Returns:
        ----------
        xi_t_T : E(xi_t|Info(T))
        P_t_T : Cov(xi_t|Info(T))
        xi2_t_T : E(xi_t * xi_t.T|Info(T))
        xi_t1_xi_t_T : E(xi_{t+1} * xi_t.T|Info(T))
        """
        # If t < T, use backward smoothing formula
        if t < self.T - 1:
            Jt = self.P_t_t[t].dot(self.Ft[t].T).dot(
                    inv(self.P_t_1t[t+1]))
            xi_t_T = self.xi_t_t[t] + Jt.dot(self.xi_t_T[-1] - \
                    self.xi_t_1t[t+1])
            P_t_T = self.P_t_t[t] + Jt.dot(self.P_t_T[-1] - \
                    self.P_t_1t[t+1]).dot(Jt.T)
            xi2_t_T = xi_t_T.dot(xi_t_T.T) + P_t_T
            xi_t1_xi_t_T = self.xi_t_T[-1].dot(self.xi_t_t[t].T) + \
                    (self.xi2_t_T[-1] - self.xi_t_T[-1].dot(
                        self.xi_t_1t[t+1].T)).dot(Jt.T)

        # If t = T, use results from Kalman Filter 
        else:
            xi_t_T = self.xi_t_t[-1]
            P_t_T = self.P_t_t[-1]
            xi2_t_T = xi_t_T.dot(xi_t_T.T) + P_t_T
            xi_t1_xi_t_T = None
        return xi_t_T, P_t_T, xi2_t_T, xi_t1_xi_t_T

    def _E_delta2(self, t: int) -> np.ndarray:
        """
        Calculated expected value of delta2. See Appendix E 
        in doc/theory.pdf for details. 

        Parameters:
        ----------
        t : time index

        Returns:
        ----------
        delta2 : expectation term for xi in MLE
        """
        # For initial state, use xi_1_0 and P_1_0 instead
        if t == 0:
            term2 = self.xi_t_1t[t].dot(self.xi_t_T[t].T)
            delta2 = self.xi2_t_T[t] - term2 - term2.T + \
                    self.xi_t_1t[t].dot(self.xi_t_1t[t].T)

        # For other state, use formular derived in doc/theory.pdf Appendix E
        else:
            Bx = self.Bt[t-1].dot(self.Xt[t-1])
            term3 = Bx.dot(self.xi_t_T[t].T)
            term4 = self.xi_t_xi_1t_T[t].dot(self.Ft[t-1].T)
            term5 = self.Ft[t-1].dot(self.xi2_t_T[t-1]).dot(self.Ft[t-1].T)
            term6 = Bx.dot(self.xi_t_T[t-1].T).dot(self.Ft[t-1].T)
            delta2 = self.xi2_t_T[t] - term4.T - term3 - term4 + term5 + \
                    term6 - term3.T + term6.T + Bx.dot(Bx.T)
        return delta2

    def _E_chi2(self, t: int) -> np.ndarray:
        """
        Calculate expected value of chi2. See Appendix F 
        in doc/theory.pdf for details.

        Parameters:
        ----------
        t : time index

        Returns:
        ----------
        chi2 : expectation term for y in MLE
        """
        Dx = self.Dt[t].dot(self.Xt[t])
        term1 = (self.Yt[t] - Dx).dot((self.Yt[t] - Dx).T)
        term2 = self.Ht[t].dot(self.xi_t_T[t]).dot((self.Yt[t] - Dx).T)
        term4 = self.Ht[t].dot(self.xi2_t_T[t]).dot(self.Ht[t].T)
        chi2 = term1 - term2 - term2.T + term4
        return chi2

    def get_smoothed_y(self) -> List[np.ndarray]:
        """
        Generated smoothed Yt. It will also generate
        smoothed values for missing measurements.

        Returns:
        ----------
        Yt_smoothed : smoothed Yt
        """
        Yt_smoothed = []
        for t in self.T:
            yt_s = self.Ht[t].dot(self.xi_t_T[t]) + \
                    self.Dt[t].dot(self.Xt[t])
            Yt_smoothed.append(yt_s)
        return Yt_smoothed

