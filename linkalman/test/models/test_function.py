import pytest
import numpy as np
import pandas as pd
from linkalman.models import F_theta, create_col

# Test creat_col
def test_create_col():
    """
    Test append column suffix
    """
    col = ['a', 'b']
    suffix = '_test'
    expected_result = ['a_test', 'b_test']
    result = create_col(col, suffix=suffix)
    assert(expected_result == result)

def test_create_col_default():
    """
    Test default suffix
    """
    col = ['a', 'b']
    expected_result = ['a_pred', 'b_pred']
    result = create_col(col)
    assert(expected_result == result)

# Test F_theta
def test_F_theta():
    """
    Test regular behavior
    """
    def f(theta):
        array = np.array([theta[0], theta[1]])
        F = B = H = D = Q = R = xi_1_0 = P_1_0 = array
        return {'F': F,
                'B': B,
                'H': H,
                'D': D,
                'Q': Q,
                'R': R,
                'xi_1_0': xi_1_0,
                'P_1_0': P_1_0}

    theta = [2, 3]
    T = 2
    Mt = F_theta(theta, f, T)
    assert(np.array_equal(Mt['Bt'][1], np.array([theta[0], theta[1]])))
    
def test_F_theta_missing_keys():
    """
    Test if raise exception if f(theta) is missing some keys
    """
    f = lambda theta: {'F': np.array([theta[0], theta[1]])}
    theta = [2, 3]
    T = 2

    with pytest.raises(ValueError):
        Mt = F_theta(theta, f, T)

def test_F_theta_wrong_type_xi():
    """
    Test if raise exception when xi is not arrays
    """
    def f(theta):
        array = np.array([theta[0], theta[1]])
        F = B = H = D = Q = R = xi_1_0 = P_1_0 = array
        return {'F': F,
                'B': B,
                'H': H,
                'D': D,
                'Q': Q,
                'R': R,
                'xi_1_0': 1,
                'P_1_0': P_1_0}

    theta = [2, 3]
    T = 2
    with pytest.raises(TypeError):
        Mt = F_theta(theta, f, T)

def test_F_theta_wrong_type_P():
    """
    Test if raise exception when xi is not arrays
    """
    def f(theta):
        array = np.array([theta[0], theta[1]])
        F = B = H = D = Q = R = xi_1_0 = P_1_0 = array
        return {'F': F,
                'B': B,
                'H': H,
                'D': D,
                'Q': Q,
                'R': R,
                'xi_1_0': xi_1_0,
                'P_1_0': 'strings'}

    theta = [2, 3]
    T = 2
    with pytest.raises(TypeError):
        Mt = F_theta(theta, f, T)
