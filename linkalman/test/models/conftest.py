import pytest
import numpy as np
from linkalman.core.utils import *
import scipy
import pandas as pd
from scipy.optimize import minimize


@pytest.fixture()
def ft_ll_mvar_diffuse():
    """
    Local linear model with 2 measurements
    """
    def ft_(theta, T, **kwargs):
        def f(theta):
            F = np.array([[1, 1], [0, 1]])
            Q = np.array([[theta[0], 0], [0, theta[1]]])
            R = np.array([[theta[2], theta[3]], [theta[3], theta[4]]])
            H = np.array([[1, 0], [theta[5], 0]])
            D = np.array([[2, 0], [1, 0]])
            M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'D': D} 
            return M
        Mt = ft(theta, f, T, **kwargs)
        return Mt
    return ft_


@pytest.fixture()
def theta_ll_mvar_diffuse():
    theta = np.array([0.3, 0.8, 0.5, 0.4, 0.6, 0.7])
    return theta


@pytest.fixture()
def Yt_mvar_diffuse_smooth():
    """
    Local linear model with complete yt, refer to Chapter 5 of Koopman and Durbin (2012)
    """
    y = np.zeros((4, 2, 1))
    y[0] = np.array([1, 2]).reshape(-1, 1)
    y[1] = np.array([np.nan, np.nan]).reshape(-1, 1)
    y[2] = np.array([np.nan, 3.5]).reshape(-1, 1)
    y[3] = np.array([3, 5]).reshape(-1, 1)
    return y


@pytest.fixture()
def df_Y():
    df = pd.DataFrame({'y': np.random.randn(30)})
    return df


@pytest.fixture()
def f_ar1():
    def f_(theta):
        """
        AR(1) model. In general, MLE is biased, so the focus should be 
        more on prediction fit, less on parameter estimation. The 
        formula here for Ar(1) is:
        y_t = c + Fy_{t-1} + epsilon_{t-1}
        """
        # Define theta
        phi_1 = 1 / (np.exp(theta[0])+1)
        sigma = np.exp(theta[1]) 
        sigma_R = np.exp(theta[2])
        # Generate F
        F = np.array([[phi_1]])
        # Generate Q
        Q = np.array([[sigma]]) 
        # Generate R
        R = np.array([[sigma_R]])
        # Generate H
        H = np.array([[1]])
        # Generate B
        B = np.array([[theta[3]]])
        # Collect system matrices
        M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'B': B}

        return M

    return f_


@pytest.fixture()
def scipy_solver():
    def solver_(param, obj_func, **kwargs):
        """
        Simple solver for LLY
        """
        obj_ = lambda x: -obj_func(x)
        res = minimize(obj_, param, **kwargs)
        theta_opt = np.array(res.x)
        fval_opt = res.fun

        return theta_opt, fval_opt

    return solver_


@pytest.fixture()
def f_arma32():
    def f_(theta):
        """
        ARMA(3, 2) model. 
        """
        F = np.array([[theta[0], theta[1], theta[2]],
        [1, 0, 0],
        [0, 1, 0]])
        Q = np.array([[1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]) 
        R = np.array([[0]])
        H = np.array([[1, 1.2, 1.3]])
        D = np.array([[2]])
        M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'D': D}

        return M

    return f_
