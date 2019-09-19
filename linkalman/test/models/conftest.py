import pytest
import numpy as np
from linkalman.core.utils import *


@pytest.fixture()
def ft_ll_mvar_diffuse():
    """
    Local linear model with 2 measurements
    """
    def ft_(theta, T):
        def f(theta):
            F = np.array([[1, 1], [0, 1]])
            Q = np.array([[theta[0], 0], [0, theta[1]]])
            R = np.array([[theta[2], theta[3]], [theta[3], theta[4]]])
            H = np.array([[1, 0], [theta[5], 0]])
            D = np.array([[2, 0], [1, 0]])
            M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'D': D} 
            return M
        Mt = ft(theta, f, T)
        return Mt
    return ft_


@pytest.fixture()
def theta_ll_mvar_diffuse():
    theta = [0.3, 0.8, 0.5, 0.4, 0.6, 0.7]
    return theta


@pytest.fixture()
def Yt_mvar_diffuse_smooth():
    """
    Local linear model with complete yt, refer to Chapter 5 of Koopman and Durbin (2012)
    """
    y = [np.array([1, 2]).reshape(-1, 1), 
         np.array([np.nan, np.nan]).reshape(-1, 1), 
         np.array([np.nan, 3.5]).reshape(-1, 1),
         np.array([3, 5]).reshape(-1, 1)]
    return y

