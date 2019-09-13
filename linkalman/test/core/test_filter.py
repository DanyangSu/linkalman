import pytest
import numpy as np
from scipy import linalg
import pandas as pd
from linkalman.core import Filter
from copy import deepcopy
from linkalman.core.utils import *


# Test _joseph_form
def test_joseph_form(ft_ar1, theta_ar1):
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
def test_LDL(ft_mvar, theta_mvar_diffuse, Yt_mvar, Xt_mvar):
    """
    Test normal run
    """
    kf = Filter(ft_mvar, for_smoother=True)
    kf.init_attr(theta_mvar_diffuse, Yt_mvar, Xt_mvar)
    
    n_t, Y_t, H_t, D_t, R_t, L_t, L_inv, \
            partitioned_index = kf._LDL(0)
    assert n_t == 3
    
    R_t_move = np.array([[3, 2, 1], 
                         [2, 4, 3], 
                         [1, 3, 6]])
    L_t_expected, R_t_expected, _ = linalg.ldl(R_t_move) 
    L_inv_expected, _ = linalg.lapack.dtrtri(
            L_t_expected, lower=True)
    np.testing.assert_array_equal(L_t, L_t_expected)
    np.testing.assert_array_equal(R_t, R_t_expected)

    Y_t_expected = linalg.pinv(L_t_expected).dot(
            np.array([1, 2, 2.1]).reshape(-1, 1))
    np.testing.assert_array_almost_equal(Y_t, Y_t_expected)
    
    H_t_expected = L_inv_expected.dot(
            np.array([1, 2, 2.4]).reshape(-1, 1))
    np.testing.assert_array_almost_equal(H_t, H_t_expected)

    expected_partitioned_index = np.array([0, 1, 2])
    np.testing.assert_array_equal(partitioned_index, 
            expected_partitioned_index)


def test_LDL_first_missing(ft_mvar, theta_mvar_diffuse, Yt_mvar, Xt_mvar):
    """
    Test when first measurement is missing
    """
    kf = Filter(ft_mvar, for_smoother=True)
    kf.init_attr(theta_mvar_diffuse, Yt_mvar, Xt_mvar)
    
    n_t, Y_t, H_t, D_t, R_t, L_t, L_inv, \
            partitioned_index = kf._LDL(1)
    assert n_t == 2
    
    R_t_move = np.array([[4, 3, 2], 
                         [3, 6, 1], 
                         [2, 1, 3]])
    L_t_expected, R_t_expected, _ = linalg.ldl(R_t_move) 
    L_inv_expected, _ = linalg.lapack.dtrtri(
            L_t_expected, lower=True)
    np.testing.assert_array_equal(L_t, L_t_expected)
    np.testing.assert_array_equal(R_t, R_t_expected)

    Y_t_expected = linalg.pinv(L_t_expected).dot(
            np.array([2.2, 3, 0]).reshape(-1, 1))
    np.testing.assert_array_almost_equal(Y_t, Y_t_expected)
    
    H_t_expected = L_inv_expected.dot(
            np.array([2, 2.4, 1]).reshape(-1, 1))
    np.testing.assert_array_almost_equal(H_t, H_t_expected)

    expected_partitioned_index = np.array([1, 2, 0])
    np.testing.assert_array_equal(partitioned_index, 
            expected_partitioned_index)


def test_LDL_full_missing(ft_mvar, theta_mvar_diffuse, Yt_mvar, Xt_mvar):
    """
    Test when all measurements are missing
    """
    kf = Filter(ft_mvar, for_smoother=True)
    kf.init_attr(theta_mvar_diffuse, Yt_mvar, Xt_mvar)
    
    n_t, Y_t, H_t, D_t, R_t, L_t, L_inv, \
            partitioned_index = kf._LDL(2)
    assert n_t == 0
    
    R_t_move = np.array([[3, 2, 1], 
                         [2, 4, 3], 
                         [1, 3, 6]])
    L_t_expected, R_t_expected, _ = linalg.ldl(R_t_move) 
    L_inv_expected, _ = linalg.lapack.dtrtri(
            L_t_expected, lower=True)
    np.testing.assert_array_equal(L_t, L_t_expected)
    np.testing.assert_array_equal(R_t, R_t_expected)

    Y_t_expected = linalg.pinv(L_t_expected).dot(
            np.array([0, 0, 0]).reshape(-1, 1))
    np.testing.assert_array_almost_equal(Y_t, Y_t_expected)
    
    H_t_expected = L_inv_expected.dot(
            np.array([1, 2, 2.4]).reshape(-1, 1))
    np.testing.assert_array_almost_equal(H_t, H_t_expected)

    expected_partitioned_index = np.array([0, 1, 2])
    np.testing.assert_array_equal(partitioned_index, 
            expected_partitioned_index)


def test_LDL_middle_missing(ft_mvar, theta_mvar_diffuse, Yt_mvar, Xt_mvar):
    """
    Test when middle measurement is missing
    """
    kf = Filter(ft_mvar, for_smoother=True)
    kf.init_attr(theta_mvar_diffuse, Yt_mvar, Xt_mvar)
    
    n_t, Y_t, H_t, D_t, R_t, L_t, L_inv, \
            partitioned_index = kf._LDL(3)
    assert n_t == 2
    
    R_t_move = np.array([[3, 1, 2], 
                         [1, 6, 3], 
                         [2, 3, 4]])
    L_t_expected, R_t_expected, _ = linalg.ldl(R_t_move) 
    L_inv_expected, _ = linalg.lapack.dtrtri(
            L_t_expected, lower=True)
    np.testing.assert_array_equal(L_t, L_t_expected)
    np.testing.assert_array_equal(R_t, R_t_expected)

    Y_t_expected = linalg.pinv(L_t_expected).dot(
            np.array([2, 3.2, 0]).reshape(-1, 1))
    np.testing.assert_array_almost_equal(Y_t, Y_t_expected)
    
    H_t_expected = L_inv_expected.dot(
            np.array([1, 2.4, 2]).reshape(-1, 1))
    np.testing.assert_array_almost_equal(H_t, H_t_expected)

    expected_partitioned_index = np.array([0, 2, 1])
    np.testing.assert_array_equal(partitioned_index, 
            expected_partitioned_index)


# Test sequential_update
def test_sequential_update_uni(ft_rw_1, theta_rw, Yt_1d, Xt_1d):
    """
    Test normal run in univariate case
    """
    t = 0
    index = 1
    ob = index - 1
    kf = Filter(ft_rw_1, for_smoother=True)
    kf.init_attr(theta_rw, Yt_1d, Xt_1d)
    kf._sequential_update(t)
    K = kf.P_star_t[t][ob] / (kf.P_star_t[t][ob] + kf.Rt[t][ob][ob])
    v = kf.Yt[t][ob] - kf.xi_t[t][ob] - kf.Dt[t][ob].dot(kf.Xt[t])
    expected_xi_t_11 = kf.xi_t[t][ob] + K * v
    expected_P_t_11 = kf.P_star_t[t][ob].dot(kf.Rt[t][ob][ob]) / (
            kf.P_star_t[t][ob] + kf.Rt[t][ob][ob])
    expected_P_t1_0 = kf.Ft[t].dot(expected_P_t_11).dot(
            kf.Ft[t]) + kf.Qt[t]
    expected_xi_t1_0 = kf.Ft[t].dot(expected_xi_t_11) + \
            kf.Bt[t].dot(kf.Xt[t])
    np.testing.assert_array_almost_equal(expected_xi_t_11, 
            kf.xi_t[t][1])
    np.testing.assert_array_almost_equal(expected_P_t_11, 
            kf.P_star_t[t][1])
    np.testing.assert_array_almost_equal(expected_P_t1_0, 
            kf.P_star_t[t+1][0])
    np.testing.assert_array_almost_equal(expected_xi_t1_0, 
            kf.xi_t[t+1][0])

    
def test_sequential_update_uni_missing(
        ft_rw_1, theta_rw, Yt_1d, Xt_1d):
    """
    Test run in univariate case with missing y
    """
    t = 1
    index = 1
    ob = index - 1
    kf = Filter(ft_rw_1, for_smoother=True)
    kf.init_attr(theta_rw, Yt_1d, Xt_1d)
    for t_ in range(t+1):
        kf._sequential_update(t_)
    K = kf.P_star_t[t][ob] / (kf.P_star_t[t][ob] + kf.Rt[t][ob][ob])
    v = kf.Yt[t][ob] - kf.xi_t[t][ob] - kf.Dt[t][ob].dot(kf.Xt[t])
    expected_xi_t_11 = None
    expected_P_t_11 = None
    expected_P_t1_0 = kf.Ft[t].dot(kf.P_star_t[t][0]).dot(
            kf.Ft[t]) + kf.Qt[t]
    expected_xi_t1_0 = kf.Ft[t].dot(kf.xi_t[t][0]) + \
            kf.Bt[t].dot(kf.Xt[t])
    assert expected_xi_t_11 == kf.xi_t[t][1]
    assert expected_P_t_11 == kf.P_star_t[t][1]
    np.testing.assert_array_almost_equal(expected_P_t1_0, 
            kf.P_star_t[t+1][0])
    np.testing.assert_array_almost_equal(expected_xi_t1_0, 
            kf.xi_t[t+1][0])


def test_sequential_update_mvar_full_obs(ft_ar2_mvar, theta_ar2_mvar, 
        Yt_ar2_mvar, Xt_ar2_mvar):
    """
    Test normal run in multi-variate case full measurements
    """
    t = 0
    kf = Filter(ft_ar2_mvar, for_smoother=True)
    kf.init_attr(theta_ar2_mvar, Yt_ar2_mvar, Xt_ar2_mvar)
    kf._sequential_update(t)
    Mt = kf.ft(kf.theta, kf.T)

    Ht = Mt['Ht'][t]
    Bt = Mt['Bt'][t]
    Dt = Mt['Dt'][t]
    Ft = Mt['Ft'][t]
    Qt = Mt['Qt'][t]
    Rt = Mt['Rt'][t]
    Upsilon = Ht.dot(kf.P_star_t[t][0]).dot(Ht.T) + Rt 
    K = kf.P_star_t[t][0].dot(Mt['Ht'][t].T).dot(linalg.pinvh(Upsilon))
    v = kf.Yt[t] - Ht.dot(kf.xi_t[t][0]) - Dt.dot(kf.Xt[t])
    
    expected_xi_t1_0 = Ft.dot(kf.xi_t[t][0] + K.dot(v)) + Bt.dot(kf.Xt[t])
    P_t_0 = kf.P_star_t[t][0] 
    P_t_t = P_t_0 - P_t_0.dot(Ht.T).dot(linalg.pinvh(Upsilon)).dot(
            Ht).dot(P_t_0)
    expected_P_t1_0 = Ft.dot(P_t_t).dot(Ft.T) + Qt
    np.testing.assert_array_almost_equal(expected_P_t1_0, 
            kf.P_star_t[t+1][0])
    np.testing.assert_array_almost_equal(expected_xi_t1_0, 
            kf.xi_t[t+1][0])

    
def test_sequential_update_mvar_missing_middle(ft_ar2_mvar, theta_ar2_mvar, 
        Yt_ar2_mvar, Xt_ar2_mvar):
    """
    Test normal run in multi-variate case missing middle measurements
    """
    t = 1
    kf = Filter(ft_ar2_mvar, for_smoother=True)
    kf.init_attr(theta_ar2_mvar, Yt_ar2_mvar, Xt_ar2_mvar)
    for t_ in range(t+1):
        kf._sequential_update(t_)
    Mt = kf.ft(kf.theta, kf.T)

    Ht = Mt['Ht'][t][[0, 2]]
    Bt = Mt['Bt'][t]
    Dt = Mt['Dt'][t][[0, 2]]
    Ft = Mt['Ft'][t]
    Qt = Mt['Qt'][t]
    Rt = Mt['Rt'][t][[0, 2]][:,[0, 2]]
    Upsilon = Ht.dot(kf.P_star_t[t][0]).dot(Ht.T) + Rt 
    K = kf.P_star_t[t][0].dot(Ht.T).dot(linalg.pinvh(Upsilon))
    v = kf.Yt[t][[0, 1]] - Ht.dot(kf.xi_t[t][0]) - Dt.dot(kf.Xt[t])
    
    expected_xi_t1_0 = Ft.dot(kf.xi_t[t][0] + K.dot(v)) + Bt.dot(kf.Xt[t])
    P_t_0 = kf.P_star_t[t][0] 
    P_t_t = P_t_0 - P_t_0.dot(Ht.T).dot(linalg.pinvh(Upsilon)).dot(
            Ht).dot(P_t_0)
    expected_P_t1_0 = Ft.dot(P_t_t).dot(Ft.T) + Qt
    np.testing.assert_array_almost_equal(expected_P_t1_0, 
            kf.P_star_t[t+1][0])
    np.testing.assert_array_almost_equal(expected_xi_t1_0, 
            kf.xi_t[t+1][0])


def test_sequential_update_mvar_all_missing(ft_ar2_mvar, theta_ar2_mvar, 
        Yt_ar2_mvar, Xt_ar2_mvar):
    """
    Test normal run in multi-variate case missing all measurements
    """
    t = 2
    kf = Filter(ft_ar2_mvar, for_smoother=True)
    kf.init_attr(theta_ar2_mvar, Yt_ar2_mvar, Xt_ar2_mvar)
    for t_ in range(t+1):
        kf._sequential_update(t_)
    Mt = kf.ft(kf.theta, kf.T)

    Bt = Mt['Bt'][t]
    Ft = Mt['Ft'][t]
    Qt = Mt['Qt'][t]
    
    expected_xi_t1_0 = Ft.dot(kf.xi_t[t][0]) + Bt.dot(kf.Xt[t])
    P_t_0 = kf.P_star_t[t][0] 
    P_t_t = P_t_0
    expected_P_t1_0 = Ft.dot(P_t_t).dot(Ft.T) + Qt
    np.testing.assert_array_almost_equal(expected_P_t1_0, 
            kf.P_star_t[t+1][0])
    np.testing.assert_array_almost_equal(expected_xi_t1_0, 
            kf.xi_t[t+1][0])


def test_sequential_update_mvar_missing_first(ft_ar2_mvar, theta_ar2_mvar, 
        Yt_ar2_mvar, Xt_ar2_mvar):
    """
    Test normal run in multi-variate case missing middle measurements
    """
    t = 3
    kf = Filter(ft_ar2_mvar, for_smoother=True)
    kf.init_attr(theta_ar2_mvar, Yt_ar2_mvar, Xt_ar2_mvar)
    for t_ in range(t+1):
        kf._sequential_update(t_)
    Mt = kf.ft(kf.theta, kf.T)

    Ht = Mt['Ht'][t][[1, 2]]
    Bt = Mt['Bt'][t]
    Dt = Mt['Dt'][t][[1, 2]]
    Ft = Mt['Ft'][t]
    Qt = Mt['Qt'][t]
    Rt = Mt['Rt'][t][[1, 2]][:,[1, 2]]
    Upsilon = Ht.dot(kf.P_star_t[t][0]).dot(Ht.T) + Rt 
    K = kf.P_star_t[t][0].dot(Ht.T).dot(linalg.pinvh(Upsilon))
    v = kf.Yt[t][[0, 1]] - Ht.dot(kf.xi_t[t][0]) - Dt.dot(kf.Xt[t])
    
    expected_xi_t_nt = kf.xi_t[t][0] + K.dot(v)
    P_t_0 = kf.P_star_t[t][0] 
    P_t_t = P_t_0 - P_t_0.dot(Ht.T).dot(linalg.pinvh(Upsilon)).dot(
            Ht).dot(P_t_0)
    expected_P_t_nt = P_t_t
    np.testing.assert_array_almost_equal(expected_P_t_nt, 
            kf.P_star_t[t][kf.n_t[t]])
    np.testing.assert_array_almost_equal(expected_xi_t_nt, 
            kf.xi_t[t][kf.n_t[t]])

