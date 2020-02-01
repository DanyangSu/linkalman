import pytest
import numpy as np
from linkalman.core.utils import *
import pandas as pd


# Generate input data
@pytest.fixture()
def Mt():
    Mt = {'Ft': build_tensor(np.ones((3, 3)), 10),
            'Bt': build_tensor(np.ones((3, 2)), 10),
            'Ht': build_tensor(np.ones((4, 3)), 10),
            'Dt': build_tensor(np.ones((4, 2)), 10),
            'Qt': build_tensor(np.ones((3, 3)), 10),
            'Rt': build_tensor(np.ones((4, 4)), 10),
            'xi_1_0': np.ones((3, 1)),
            'P_1_0': np.ones((3, 3))}
    return Mt


@pytest.fixture()
def df1():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    return df


@pytest.fixture()
def Yt():
    Yt = np.ones((1, 4, 1))
    return Yt


@pytest.fixture()
def Xt():
    Xt = np.ones((1, 2, 1))
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
def Yt_1d():
    """
    1d Yt
    """
    y = np.array([[[1]], [[np.nan]], [[2.5]]])
    return y


@pytest.fixture()
def Xt_1d():
    """
    1d Xt
    """
    x = np.array([[[0.1]], [[0.2]], [[0.3]]])
    return x


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
    theta_ = [-30, 0.3]
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
            R = np.array([[3, 2, 1], 
                          [2, 4, 3],
                          [1, 3, 6]])
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
    Yt = np.zeros((4, 3, 1))
    Yt[0] = np.array([1, 2, 2.1]).reshape(-1, 1)
    Yt[1] = np.array([np.nan, 2.2, 3]).reshape(-1, 1)
    Yt[2] = np.array([np.nan, np.nan, np.nan]).reshape(-1, 1)
    Yt[3] = np.array([2, np.nan, 3.2]).reshape(-1, 1)
    return Yt


@pytest.fixture()
def Xt_mvar():
    Xt = np.array([[[0.2]], [[0.3]], [[0.4]], [[0.1]]])
    return Xt


@pytest.fixture()
def ft_rw_1():
    """
    Random walk process with one measurements
    """
    def ft_(theta, T):
        xi_1_0 = np.array([[0.2]])
        P_1_0 = np.array([[2]])
        def f(theta):
            phi_1 = 1
            sigma_Q = np.exp(theta[0])
            sigma_R = np.exp(theta[1])
            F = np.array([[phi_1]])
            Q = np.array([[sigma_Q]])
            R = np.array([[sigma_R]])
            H = np.array([[1]])
            B = np.array([[0.1]])
            M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'B': B} 
            return M
        Mt = ft(theta, f, T, xi_1_0=xi_1_0, P_1_0=P_1_0)
        return Mt
    return ft_


@pytest.fixture()
def theta_rw():
    theta_ = [0.2, 0.1]
    return theta_


@pytest.fixture()
def theta_ar2_mvar():
    theta = [0.2, 0.3, 1, 1, 2]
    return theta


@pytest.fixture()
def ft_ar2_mvar():
    """
    ft for ar2 process 
    """
    def ft_(theta, T):
        def f(theta):
            F = np.array([[theta[0], theta[1]], [1, 0]])
            B = np.array([[0.2], [0]])
            Q = np.array([[theta[2], 0], [0, 0]])
            H = np.array([[2, 0], [3, 0], [4, 1]])
            D = np.array([[0.1], [2], [3]])
            R = np.array([[4, 2, 1], 
                          [2, 5, 3],
                          [1, 3, 6]])
            M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'B': B, 'D': D} 
            return M
        x_0 = np.array([[1]])
        Mt = ft(theta, f, T, x_0=x_0)
        return Mt

    return ft_


@pytest.fixture()
def ft_ar2_mvar_kw():
    """
    ft for ar2 process with special x_0 through kwargs
    """
    def ft_(theta, T, **kwargs):
        def f(theta):
            F = np.array([[theta[0], theta[1]], [1, 0]])
            B = np.array([[0.2], [0]])
            Q = np.array([[theta[2], 0], [0, 0]])
            H = np.array([[2, 0], [3, 0], [4, 1]])
            D = np.array([[0.1], [2], [3]])
            R = np.array([[4, 2, 1], 
                          [2, 5, 3],
                          [1, 3, 6]])
            M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'B': B, 'D': D} 
            return M
        x_0 = np.array([[1]])
        Mt = ft(theta, f, T, **kwargs)
        return Mt

    return ft_


@pytest.fixture()
def Yt_ar2_mvar():
    Yt = np.zeros((4, 3, 1))
    Yt[0] = np.array([1, 2, 3]).reshape(-1, 1)
    Yt[1] = np.array([2, np.nan, 4]).reshape(-1, 1)
    Yt[2] = np.array([np.nan, np.nan, np.nan]).reshape(-1, 1)
    Yt[3] = np.array([np.nan, 2.5, 3.5]).reshape(-1, 1)
    return Yt


@pytest.fixture()
def Xt_ar2_mvar():
    Xt = np.array([[[1]], [[2]], [[1.5]], [[0.8]]])
    return Xt


@pytest.fixture()
def ft_rw_1_diffuse():
    """
    Random walk process with one measurements and diffuse state
    """
    def ft_(theta, T):
        def f(theta):
            phi_1 = 1
            sigma_Q = theta[0]
            sigma_R = theta[1]
            F = np.array([[phi_1]])
            Q = np.array([[sigma_Q]])
            R = np.array([[sigma_R]])
            H = np.array([[1]])
            M = {'F': F, 'Q': Q, 'H': H, 'R': R} 
            return M
        Mt = ft(theta, f, T)
        return Mt
    return ft_


@pytest.fixture()
def Yt_1d_missing():
    """
    1d Yt
    """
    y = [np.array([[np.nan]]), np.array([[1]]), np.array([[2.5]])]
    return y


@pytest.fixture()
def theta_ll_1d_diffuse():
    theta = [0.2, 0.3, 0.8]
    return theta


@pytest.fixture()
def ft_ll_1d_diffuse():
    """
    Local linear model with 1 measurements
    """
    def ft_(theta, T):
        def f(theta):
            F = np.array([[1, 1], [0, 1]])
            Q = np.array([[theta[0], 0], [0, theta[1]]])
            R = np.array([[theta[2]]])
            H = np.array([[1, 0]])
            M = {'F': F, 'Q': Q, 'H': H, 'R': R} 
            return M
        Mt = ft(theta, f, T)
        return Mt
    return ft_

    
@pytest.fixture()
def Yt_1d_full():
    """
    1d Yt fully observed
    """
    y = [np.array([[2]]), np.array([[1.3]]), np.array([[2.5]]), np.array([[3.1]])]
    return y


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
            H = np.array([[1, 0], [2, 0]])
            D = np.array([[2, 0], [1, 0]])
            M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'D': D} 
            return M
        Mt = ft(theta, f, T)
        return Mt
    return ft_


@pytest.fixture()
def Yt_mvar_diffuse():
    """
    Local linear model with complete yt, refer Chapter 5 of Koopman and Durbin (2012)
    """
    y = [np.array([1, 2]).reshape(-1, 1), 
         np.array([2, np.nan]).reshape(-1, 1), 
         np.array([np.nan, 3.5]).reshape(-1, 1),
         np.array([3, 5]).reshape(-1, 1)]
    return y


@pytest.fixture()
def theta_ll_mvar_diffuse():
    theta = [0.3, 0.8, 0.5, 0.4, 0.6]
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


@pytest.fixture()
def Yt_mvar_diffuse_smooth_vec():
    """
    Local linear model with complete yt, refer to Koopman (1997)
    """
    y = [np.array([1, 2]).reshape(-1, 1), 
         np.array([2.4, 3.2]).reshape(-1, 1),
         np.array([3, 5]).reshape(-1, 1)]
    return y


@pytest.fixture()
def Yt_mvar_diffuse_missing():
    """
    Yt with missing measurements at t
    """
    y = [np.array([np.nan, 2]).reshape(-1, 1), 
         np.array([np.nan, np.nan]).reshape(-1, 1), 
         np.array([2.5, np.nan]).reshape(-1, 1),
         np.array([3, 5]).reshape(-1, 1)]
    return y


@pytest.fixture()
def ft_ll_mvar_1d():
    """
    Create a 1d equivalence of mvar with missing measurements.
    Refer to ft_ll_mvar_diffuse().
    """
    def ft_(theta, T):
        F = np.array([[1, 1], [0, 1]])
        Ft = [F.copy() for _ in range(T)]
        Q = np.array([[theta[0], 0], [0, theta[1]]])
        Qt = [Q.copy() for _ in range(T)]
        Rt = [np.array([[theta[4]]]),
             np.array([[1]]),  # not being used
             np.array([[theta[2]]]), 
             np.array([[1]])]  # not relevant
        Ht = [np.array([[2, 0]]), 
              np.array([[1.5, 0]]),
              np.array([[1, 0]]),
              np.array([[4, 0]])]

        Bt = [np.array([[0, 0]]).reshape(-1, 1) for _ in range(T)]
        Dt = [np.array([[0]]) for _ in range(T)]
        xi_1_0 = np.array([[0], [0]])
        P_1_0 = np.diag([np.nan, np.nan])

        Mt = {'Ft': Ft, 'Qt': Qt, 'Bt': Bt, 'Ht': Ht, 'Dt': Dt, 
                'Rt': Rt, 'xi_1_0': xi_1_0, 'P_1_0': P_1_0} 
        return Mt
    return ft_


@pytest.fixture()
def Yt_mvar_1d():
    y = [np.array([[2]]),
         np.array([[np.nan]]),
         np.array([[2.5]]),
         np.array([[7]])]
    return y

@pytest.fixture()
def ft_q():
    """
    Test update q
    """
    def ft_(theta, T):
        F = np.array([[1, 0, 0, 0, 0]] * 5)
        Ft = [F.copy() for _ in range(T)]
        Q = np.eye(5)
        Qt = [Q.copy() for _ in range(T)]
        R = np.eye(2)
        Rt = [R.copy() for _ in range(T)]
        H = np.array([[2, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        Ht = [H.copy() for _ in range(T)]


        Bt = [np.zeros([5, 1]) for _ in range(T)]
        Dt = [np.zeros([2, 1]) for _ in range(T)]
        xi_1_0 = np.zeros([5, 1])
        P_1_0 = np.diag([np.nan] * 5)

        Mt = {'Ft': Ft, 'Qt': Qt, 'Bt': Bt, 'Ht': Ht, 'Dt': Dt, 
                'Rt': Rt, 'xi_1_0': xi_1_0, 'P_1_0': P_1_0} 
        return Mt
    return ft_


@pytest.fixture()
def Yt_q():
    y = [np.array([[2], [1.1]]),
         np.array([[2.2], [1.14]])]
    return y
