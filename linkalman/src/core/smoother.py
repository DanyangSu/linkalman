import numpy as np
from copy import deepcopy as copy
from scipy import linalg

def inv(h_array):
    """
    PSD matrix inverse
    """
    return linalg.pinvh(h_array)


class Smoother(object):

    def __init__(self, kf):
        """
        Initialize a Kalman Smoother. Refer to linkalman/doc/theory.pdf for details
        """
        self.__dict__.update(kf.__dict__)

        # Create additional matrices
        self.xi_t_T = []
        self.P_t_T = []
        self.xi2_t_T = []
        self.xi_t_xi_1t_T = []
        self.delta2 = []
        self.chi2 = []
        
    def _smooth(self, t):
        """
        Update Kalman Smoother at time t
        """
        if t < self.T - 1:
            Jt = self.P_t_t[t].dot(self.Ft[t].T).dot(inv(self.P_t_1t[t+1]))
            xi_t_T = self.xi_t_t[t] + Jt.dot(self.xi_t_T[-1] - self.P_t_1t[t+1])
            P_t_T = self.P_t_t[t] + Jt.dot(self.P_t_T[-1] - self.P_t_1t[t+1]).dot(Jt.T)
            xi2_t_T = xi_t_T.dot(xi_t_T.T) + P_t_T
            xi_t1_xi_t_T = self.xi_t_T[-1].dot(self.xi_t_t[t].T) + (self.xi2_t_T[-1] - 
                    self.xi_t_T[-1].dot(self.xi_t_1t[t+1])).dot(Jt.T)
        else:
            xi_t_T = self.xi_t_t[-1]
            P_t_T = self.P_t_t[-1]
            xi2_t_T = xi_t_T.dot(xi_t_T.T) + P_t_T
            xi_t1_xi_t_T = None
        return xi_t_T, P_t_T, xi2_t_T, xi_t1_xi_t_T

    def _E_delta2(self, t):
        """
        Calculated expected value of delta2. See Appendix E in doc/theory.pdf for details
        """
        if t == 0:
            term2 = self.xi_t_1t[t].dot(self.xi_t_T[t].T)
            delta2 = self.xi2_t_T[t] - term2 - term2.T + self.xi_t_1t[t].dot(self.xi_t_1t[t].T)
        else:
            Bx = self.Bt[t-1].dot(self.Xt[t-1])
            term3 = Bx.dot(self.xi_t_T[t].T)
            term4 = self.xi_t_xi_1t_T[t].dot(self.Ft[t-1].T)
            term5 = self.Ft[t-1].dot(self.xi2_t_T[t-1]).dot(self.Ft[t-1].T)
            term6 = Bx.dot(self.xi_t_T[t-1].T).dot(self.Ft[t-1].T)
            delta2 = self.xi2_t_T[t] - term4.T - term3 - term4 + term5 + term6 - term3.T +
                term6.T + Bx.dot(Bx.T)
        return delta2

    def _E_chi2(self, t):
        """
        Calculate expected value of chi2. See Appendix F in doc/theory.pdf for details
        """
        Dx = self.Dt[t].dot(self.Xt[t])
        term1 = (self.Yt[t] - Dx).dot((self.Yt[t] - Dx).T)
        term2 = self.Ht[t].dot(self.xi_t_T[t]).dot((self.Yt[t] - Dx).T)
        term4 = self.Ht[t].dot(self.xi2_t_T[t]).dot(self.Ht[t].T)
        chi2 = term1 - term2 - term2.T + term4
        return chi2

    def __call__(self):
        """
        Run backward smoothing
        """
        for t in reversed(range(self.T - 1)):
            xi_t_T, P_t_T, xi2_t_T, xi_t1_xi_t_T = self._smooth(t)
            self.xi_t_T.append(xi_t_T)
            self.P_t_T.append(P_t_T)
            self.xi2_t_T.append(xi2_t_T)
            self.xi_t_xi_1t_T.append(xi_t1_x_t_T)

        # match index for xi_t_xi_1t_T
        self.xi_t_xi_1t_T.append(None)
        self.xi_t_xi_1t_T.pop(0)

        # Reverse the order
        self.xi_t_T = list(reversed(self.xi_t_T))
        self.P_t_T = list(reversed(self.P_t_T))
        self.xi2_t_T = list(reversed(self.xi2_t_T))
        self.xi_t_xi_1t_T = list(reversed(self.xi_t_xi_1t_T))
            
        # Calculate delta2 can chi2 in the E-step of the EM algorithm
        for t in range(self.T):
            self.delta2.append(self._E_delta2(t))
            self.chi2.append(self._E_chi2(t))

