import pytest
from linkalman.models import BaseConstantModel as BCM
import numpy as np
import pandas as pd


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
    Mt = BCM.F_theta(theta, f, T)
    assert(np.array_equal(Mt['Bt'][1], np.array([theta[0], theta[1]])))
    
def test_F_theta_missing_keys():
    """
    Test if raise exception if f(theta) is missing some keys
    """
    f = lambda theta: {'F': np.array([theta[0], theta[1]])}
    theta = [2, 3]
    T = 2

    with pytest.raises(ValueError):
        Mt = BCM.F_theta(theta, f, T)

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
        Mt = BCM.F_theta(theta, f, T)

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
        Mt = BCM.F_theta(theta, f, T)

# Test gen_PSD
def test_gen_PSD():
    """
    Test normal behavior 
    """
    theta = [1,2,3,4,5,6]
    dim = 3
    expected_results = np.array([[1, 2, 4],
                                [2, 13, 23],
                                [4, 23, 77]])
    results = BCM.gen_PSD(theta, dim)
    assert(np.array_equal(expected_results, results))

def test_gen_PSD_wrong_theta_size():
    """
    Test if raise exception when theta wrong size
    """
    theta = [1,2,3,4,5,6]
    dim = 2
    with pytest.raises(ValueError):
        PSD = BCM.gen_PSD(theta, dim)

