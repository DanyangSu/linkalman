import numpy as np
from functools import partial
import scipy
from scipy import linalg
from .kalman_filter import Filter
from .kalman_smoother import Smoother
import nlopt
from copy import deepcopy
from typing import List, Any, Callable

__all__ = ['EM']

class EM(object):

    def __init__(self, Ft: Callable) -> None:
        """
        Initialize an EM Optimizer. Starting with an initial Mt, the 
        algorithm iteratively updates theta until it converges. Refer to 
        linkalman/doc/theory.pdf for details.

        Parameters:
        ----------
        Ft : mapping from theta to Mt
        """
        self.Ft = Ft

    def fit(self, theta: List[float], Xt: List[np.ndarray], 
            Yt: List[np.ndarray], threshold: float=0.01) -> List[float]:
        """
        Perform the EM algorithm until G converges
        """
        dist = 1
        theta_init = deepcopy(theta) 
        G_init = np.inf
        while dist > threshold:
            theta_opt, G_opt = self._em(theta_init)
            dist = abs(G_init - G_opt)
            G_init = G_opt
            theta_init = theta_opt
        return theta_init

    @staticmethod
    def E_step(Mt, Xt, Yt):
        """
        Perform E-step
        """
        kf = Filter(Mt)
        kf(Xt, Yt)
        ks = Smoother(kf)
        ks()
        return ks

    def _em(self, theta_init):
        """
        Perform E-step and M-step within each iteration
        """
        # Generate system matrices
        Mt = self.Ft(theta_init)
        
        # E-Step
        ks = E_step(Mt, Xt, Yt)

        # M-Step
        obj = partial(self._G, ks)
        opt = nlopt.opt('nlopt.LN_BOBYQA', length(theta_init))
        opt.set_max_objective(obj)
        theta_opt = opt.optimize(theta_init)
        mle_opt = opt.last_optimum_value()
        return theta_opt, mle_opt
    
    def _G(self, ks, theta):
        """
        Calculate expected likelihood
        """  
        G = 0
        for t in range(self.T):
            G += self._G1(ks, t) + self._G2(ks, t)
        return G

    def _G1(self, ks, t):
        """
        Calculate expected likelihood of xi_t
        """ 
        return -0.5 * scipy.log(ks.Qt.pdet(t)) - \
                0.5 * np.trace(ks.Qt.pinv(t).dot(ks.delta2[t]))

    def _G2(self, ks, t):
        """
        Calculate expected likelihood of y_t
        """
        return -0.5 * scipy.log(ks.Rt.pdet(t)) - \
                0.5 * np.trace(ks.Rt.pinv(t).dot(ks.chi2[t]))






