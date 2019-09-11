import pytest
import numpy as np
from linkalman.core.utils import *


# Generate input data
@pytest.fixture()
def Mt():
    Mt = {'Ft': [np.ones((3, 3))],
            'Bt': [np.ones((3, 2))],
            'Ht': [np.ones((4, 3))],
            'Dt': [np.ones((4, 2))],
            'Qt': [np.ones((3, 3))],
            'Rt': [np.ones((4, 4))],
            'xi_1_0': np.ones((3, 1)),
            'P_1_0': np.ones((3, 3))}
    return Mt


@pytest.fixture()
def Yt():
    Yt = [np.ones((4, 1))]
    return Yt


@pytest.fixture()
def Xt():
    Xt = [np.ones((2, 1))]
    return Xt


@pytest.fixture()
def perm_mat():
    mat = np.array([[1, 2, 3, 4],
                    [2, 5, 6, 7],
                    [3, 6, 8, 9],
                    [4, 7, 9, 0]])
    return mat


@pytest.fixture()
def perm_vector():
    vec = np.array([[1], [2], [3], [4]])
    return vec


@pytest.fixture()
def ft_ar1():
    """
    Standard ar1 process
    """
    def ft_(theta, T):
        def f(theta):
            phi_1 = 1 / (np.exp(theta[0])+1)
            sigma_Q = np.exp(theta[1])
            sigma_R = np.exp(theta[2])
            F = np.array([[phi_1]])
            Q = np.array([[sigma_Q]])
            R = np.array([[sigma_R]])
            H = np.array([[1]])
            B = np.array([[0.1]])
            M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'B': B} 
            return M
        ft(theta, f, T)
    return ft_


@pytest.fixture()
def y_ar1():
    """
    DataFrame for AR1 process
    """
    y = np.array([1, 2, 2.5]).reshape(-1, 1)
    return y


@pytest.fixture()
def theta_ar1():
    """
    theta for AR1 process
    """
    theta = [-0.1, -0.2, -0.3]
    return theta


@pytest.fixture()
def theta_mvar():
    theta_ = [0, 0.3, -0.1, -0.2, 0.1, 0.2, 0.15, 0.25]
    return theta_


@pytest.fixture()
def theta_mvar_diffuse():
    theta_ = [-30, 0.3, -0.1, -0.2, 0.1, 0.2, 0.15, 0.25]
    return theta_


@pytest.fixture()
def ft_mvar():
    """
    Multi-measurement ar1 process
    """
    def ft_(theta, T):
        def f(theta):
            phi_1 = 1 / (np.exp(theta[0])+1)
            if phi_1 > 1 - 0.001:
                phi_1 = 1
            sigma_Q = np.exp(theta[1])

            F = np.array([[phi_1]])
            Q = np.array([[sigma_Q]])
            R = gen_PSD(theta[2:8], 3)
            H = np.array([[1], [2], [2.4]])
            B = np.array([[0.1]])
            D = np.array([[-0.1], [-0.2], [0.1]])
            M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'B': B, 'D': D} 
            return M

        Mt = ft(theta, f, T)
        return Mt

    return ft_


@pytest.fixture()
def Yt_mvar():
    """
    Contain missing measurements
    """
    Yt = [np.array([1, 2, 2.1]).reshape(-1, 1),
            np.array([np.nan, 2.2, 3]).reshape(-1, 1),
            np.array([2, np.nan, 3.2]).reshape(-1, 1)]
    return Yt


@pytest.fixture()
def Xt_mvar():
    Xt = [np.array([[0.2]]), np.array([[0.3]]), np.array([[0.4]])]
    return Xt
