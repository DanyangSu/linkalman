import pytest
import numpy as np
from scipy import linalg
import pandas as pd
from linkalman.core import Filter
from copy import deepcopy
from linkalman.core.utils import *


# Test _joseph_form
def test_joseph_form(ft_ar1, y_ar1, theta_ar1):
    """
    Test normal run
    """

    kf = Filter(ft_ar1, for_smoother=True)
    L = np.array([[2, 3], [4, 5]])
    P = np.array([[3, 4], [4, 5]])
    KRK = np.ones([2, 2])
    result = kf._joseph_form(L, P, KRK=KRK)
    expected_result = np.array([[106, 188], [188, 334]]) 

    np.testing.assert_array_equal(result, expected_result)


# Test init
def test_attr_input(ft_mvar, theta_mvar, Yt_mvar, Xt_mvar):
    """
    Test normal run
    """
    kf = Filter(ft_mvar, for_smoother=True)
    kf.init_attr(theta_mvar, Yt_mvar, Xt_mvar)
    assert len(kf.l_t) == len(Yt_mvar) and \
            len(kf.L_star_t[0]) == Yt_mvar[0].shape[0]


def test_init_attr_diffuse(ft_mvar, theta_mvar_diffuse, Yt_mvar, Xt_mvar):
    """
    Test if init_attr for diffuse
    """
    kf = Filter(ft_mvar, for_smoother=True)
    kf.init_attr(theta_mvar_diffuse, Yt_mvar, Xt_mvar)
    assert kf.q == 1 and \
            len(kf.L0_t[0]) == Yt_mvar[0].shape[0]


# Test _LDL
def test_LDL(ft_mvar, theta_mvar, Yt_mvar, Xt_mvar):
    """
    Test normal run
    """
    # kf = Filter(ft_mvar, for_smoother=True)
    # kf.init(theta_mvar, Yt_mvar, Xt_mvar)
    
    # n_t, Y_t, H_t, D_t, R_t, L_t, L_inv, \
    #         partitioned_index = self._LDL(1)
    assert True

def test_LDL_missing():
    """
    Test shuffle result when missing y_t
    """
    assert True

