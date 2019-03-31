import numpy as np
from scipy.optimize import minimize
from functools import partial
from ../core/utils.py import *
import scipy
from scipy import linalg

class EM(object):

    def __init__(self, f_theta):
        """
        Initialize an EM Optimizer. Refer to linkalman/doc/theory.pdf for details.
        f_theta is a customized function that returns system Mt from theta
        """



    def _reverse()

    def _G1(self, Mt, t):
        
        return -0.5 * scipy.log(scipy.linalg.det(Mt['Qt'][t])) - \\
                0.5 * numpy.matrix.trace(Mt['Qt'].inv(t) * self._E_delta2(t))

    def _G2(self, Mt, t):
        
        return -0.5 * scipy.log(scipy.linalg.det(Mt['Rt'][t])) - \\
                0.5 * numpy.matrix.trace(Mt['Rt'].inv(t) * self._E_chi2(t))

    def _E_delta2(self, t):
        E_xi = 


    def _E_chi2(self, t):
        pass

    def _G(self, ks, f_theta):
        
        G = 0
        for t in range(self.T):
            G += self._G1(Mt, t) + self._G2(Mt, t)
        return G


    def __call__(self, theta_init, threshold=0.01):
        """
        Perform the EM algorithm until G converges
        """
        dist = 1
        G_init = np.inf
        while dist > threshold:
            (theta_opt, G_opt) = self._EM(theta_init)
            dist = abs(G_init - G_opt)
            G_init = G_opt
            theta_init = theta_opt
        return theta_init

    def _em(self, theta_init):
        Mt = self.f_theta(theta_init)
        _check_M(Mt)
        
        # Diagonalize HMM
        Mt_diag = self.diag(Mt)
        
        # E-Step
        kf = Filter(**M_diag)
        ks = Smoother(kf())
        E_delta2 = self._E_delta2()
        E_chi2 = self._E_chi2()

        # M-Step
        MLE = self.G(E_delta2, E_chi2, self.f_theta)
        theta_opt = minimize()

        return theta_opt







