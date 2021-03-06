import pytest
import numpy as np
from scipy import linalg
import pandas as pd
from linkalman.core import Filter
from copy import deepcopy
from linkalman.core.utils import *


# Test _joseph_form
def test_joseph_form(ft_ar1, theta_ar1, Yt_1d):
    """
    Test normal run
    """

    kf = Filter(ft_ar1, Yt_1d, for_smoother=True)
    L = np.array([[2, 3], [4, 5]])
    P = np.array([[3, 4], [4, 5]])
    KRK = np.ones([2, 2])
    result = kf._joseph_form(L, P, KRK=KRK)
    expected_result = np.array([[106, 188], [188, 334]]) 

    np.testing.assert_array_equal(result, expected_result)


# Test init
def test_init_attr_input(ft_mvar, theta_mvar, Yt_mvar, Xt_mvar):
    """
    Test normal run
    """
    kf = Filter(ft_mvar, Yt_mvar, Xt_mvar, for_smoother=True)
    kf.init_attr(theta_mvar)
    assert len(kf.L_star_t) == len(Yt_mvar) and \
            len(kf.L_star_t[0]) == Yt_mvar[0].shape[0]


def test_init_attr_diffuse(ft_mvar, theta_mvar_diffuse, Yt_mvar, Xt_mvar):
    """
    Test if init_attr for diffuse
    """
    kf = Filter(ft_mvar, Yt_mvar, Xt_mvar, for_smoother=True)
    kf.init_attr(theta_mvar_diffuse)
    assert kf.q == 1 and \
            len(kf.L0_t[0]) == Yt_mvar[0].shape[0]


# Test _LDL
def test_LDL(ft_mvar, theta_mvar_diffuse, Yt_mvar, Xt_mvar):
    """
    Test normal run
    """
    kf = Filter(ft_mvar, Yt_mvar, Xt_mvar, for_smoother=True)
    kf.init_attr(theta_mvar_diffuse)
    
    Y_t, H_t, D_t, R_t, L_t, L_inv =  kf._LDL(0)
    assert kf.n_t[0] == 3
    
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
    np.testing.assert_array_equal(kf.partitioned_index[0], 
            expected_partitioned_index)


def test_LDL_first_missing(ft_mvar, theta_mvar_diffuse, Yt_mvar, Xt_mvar):
    """
    Test when first measurement is missing
    """
    kf = Filter(ft_mvar, Yt_mvar, Xt_mvar, for_smoother=True)
    kf.init_attr(theta_mvar_diffuse)
    
    Y_t, H_t, D_t, R_t, L_t, L_inv = kf._LDL(1)
    assert kf.n_t[1] == 2
    
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
    np.testing.assert_array_equal(kf.partitioned_index[1], 
            expected_partitioned_index)


def test_LDL_full_missing(ft_mvar, theta_mvar_diffuse, Yt_mvar, Xt_mvar):
    """
    Test when all measurements are missing
    """
    kf = Filter(ft_mvar, Yt_mvar, Xt_mvar, for_smoother=True)
    kf.init_attr(theta_mvar_diffuse)
    
    Y_t, H_t, D_t, R_t, L_t, L_inv = kf._LDL(2)
    assert kf.n_t[2] == 0
    
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
    np.testing.assert_array_equal(kf.partitioned_index[2], 
            expected_partitioned_index)


def test_LDL_middle_missing(ft_mvar, theta_mvar_diffuse, Yt_mvar, Xt_mvar):
    """
    Test when middle measurement is missing
    """
    kf = Filter(ft_mvar, Yt_mvar, Xt_mvar, for_smoother=True)
    kf.init_attr(theta_mvar_diffuse)
    
    Y_t, H_t, D_t, R_t, L_t, L_inv = kf._LDL(3)
    assert kf.n_t[3] == 2
    
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
    np.testing.assert_array_equal(kf.partitioned_index[3], 
            expected_partitioned_index)


# Test sequential_update
def test_sequential_update_uni(ft_rw_1, theta_rw, Yt_1d, Xt_1d):
    """
    Test normal run in univariate case
    """
    t = 0
    index = 1
    ob = index - 1
    kf = Filter(ft_rw_1, Yt_1d, Xt_1d, for_smoother=True)
    kf.init_attr(theta_rw)
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
    kf = Filter(ft_rw_1, Yt_1d, Xt_1d, for_smoother=True)
    kf.init_attr(theta_rw)
    for t_ in range(t+1):
        kf._sequential_update(t_)
    K = kf.P_star_t[t][ob] / (kf.P_star_t[t][ob] + kf.Rt[t][ob][ob])
    v = kf.Yt[t][ob] - kf.xi_t[t][ob] - kf.Dt[t][ob].dot(kf.Xt[t])
    expected_xi_t_11 = np.array([[np.nan]])
    expected_P_t_11 = np.zeros((1, 1)) * np.nan
    expected_P_t1_0 = kf.Ft[t].dot(kf.P_star_t[t][0]).dot(
            kf.Ft[t]) + kf.Qt[t]
    expected_xi_t1_0 = kf.Ft[t].dot(kf.xi_t[t][0]) + \
            kf.Bt[t].dot(kf.Xt[t])
    np.testing.assert_array_equal(expected_xi_t_11, kf.xi_t[t][1])
    np.testing.assert_array_equal(expected_P_t_11, kf.P_star_t[t][1])
    np.testing.assert_array_almost_equal(expected_P_t1_0, 
            kf.P_star_t[t+1][0])
    np.testing.assert_array_almost_equal(expected_xi_t1_0, 
            kf.xi_t[t+1][0])


def test_sequential_update_mvar_full_obs(ft_ar2_mvar_kw, theta_ar2_mvar, 
        Yt_ar2_mvar, Xt_ar2_mvar):
    """
    Test normal run in multi-variate case full measurements
    """
    t = 0
    kf = Filter(ft_ar2_mvar_kw, Yt_ar2_mvar, Xt_ar2_mvar, for_smoother=True)
    kf.init_attr(theta_ar2_mvar)
    kf._sequential_update(t)
    Mt = kf.ft(kf.theta, kf.T, x_0=Xt_ar2_mvar[0])

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
   

def test_sequential_update_mvar_missing_middle(ft_ar2_mvar_kw, theta_ar2_mvar, 
        Yt_ar2_mvar, Xt_ar2_mvar):
    """
    Test normal run in multi-variate case missing middle measurements
    """
    t = 1
    kf = Filter(ft_ar2_mvar_kw, Yt_ar2_mvar, Xt_ar2_mvar, for_smoother=True)
    kf.init_attr(theta_ar2_mvar)
    for t_ in range(t+1):
        kf._sequential_update(t_)
    Mt = kf.ft(kf.theta, kf.T, x_0=Xt_ar2_mvar[0])

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


def test_sequential_update_mvar_all_missing(ft_ar2_mvar_kw, theta_ar2_mvar, 
        Yt_ar2_mvar, Xt_ar2_mvar):
    """
    Test normal run in multi-variate case missing all measurements
    """
    t = 2
    kf = Filter(ft_ar2_mvar_kw, Yt_ar2_mvar, Xt_ar2_mvar, for_smoother=True)
    kf.init_attr(theta_ar2_mvar)
    for t_ in range(t+1):
        kf._sequential_update(t_)
    Mt = kf.ft(kf.theta, kf.T, x_0=Xt_ar2_mvar[0])
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


def test_sequential_update_mvar_missing_first(ft_ar2_mvar_kw, theta_ar2_mvar, 
        Yt_ar2_mvar, Xt_ar2_mvar):
    """
    Test normal run in multi-variate case missing middle measurements
    """
    t = 3
    kf = Filter(ft_ar2_mvar_kw, Yt_ar2_mvar, Xt_ar2_mvar, for_smoother=True)
    kf.init_attr(theta_ar2_mvar)
    for t_ in range(t+1):
        kf._sequential_update(t_)
    Mt = kf.ft(kf.theta, kf.T, x_0=Xt_ar2_mvar[0])

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


# Test sequential_update_diffuse
def test_sequential_update_diffuse_missing(ft_rw_1_diffuse, theta_rw, 
        Yt_1d_missing, Xt_1d):
    """
    Test first missing
    """
    t = 0
    kf = Filter(ft_rw_1_diffuse, Yt_1d_missing, Xt_1d, for_smoother=True)
    kf.init_attr(theta_rw)
    for t_ in range(t+1):
        kf._sequential_update_diffuse(t_)
    e_P_inf_t1_0 = np.array([[1]])
    e_xi_t1_0 = np.array([[0]])
    np.testing.assert_array_almost_equal(e_P_inf_t1_0, kf.P_inf_t[t+1][0])
    np.testing.assert_array_almost_equal(e_xi_t1_0, kf.xi_t[t+1][0])


def test_sequential_update_diffuse(ft_rw_1_diffuse, theta_rw,
        Yt_1d_missing, Xt_1d):
    """
    Test normal run
    """
    t = 1
    kf = Filter(ft_rw_1_diffuse, Yt_1d_missing, Xt_1d, for_smoother=True)
    kf.init_attr(theta_rw)
    for t_ in range(t+1):
        kf._sequential_update_diffuse(t_)
    e_P_inf_t1_0 = np.array([[0]])
    e_P_star_t1_0 = kf.Rt[0] + kf.Qt[0]
    e_xi_t1_0 = kf.Yt[1]
    np.testing.assert_array_almost_equal(e_P_inf_t1_0, kf.P_inf_t[t+1][0])
    np.testing.assert_array_almost_equal(e_xi_t1_0, kf.xi_t[t+1][0])
    np.testing.assert_array_almost_equal(e_P_star_t1_0, kf.P_star_t[t+1][0])


def test_sequential_update_diffuse_ll_1d(ft_ll_1d_diffuse,
        theta_ll_1d_diffuse, Yt_1d_full):
    """
    Test local linear models from chapter 5 of Koopman and Durbin (2012)
    """
    t = 3
    kf = Filter(ft_ll_1d_diffuse, Yt_1d_full, for_smoother=True)
    kf.init_attr(theta_ll_1d_diffuse)
    for t_ in range(t):
        kf._sequential_update_diffuse(t_)
    
    # Test period 0 result
    q1 = theta_ll_1d_diffuse[0] / theta_ll_1d_diffuse[2]
    q2 = theta_ll_1d_diffuse[1] / theta_ll_1d_diffuse[2]
    e_P_inf_t1_0 = np.ones([2, 2])
    e_P_star_t1_0 = np.array([[1 + q1, 0], [0, q2]]) * theta_ll_1d_diffuse[2] 
    e_xi_t1_0 = np.array([[Yt_1d_full[0][0]], [0]])
    
    np.testing.assert_array_almost_equal(e_P_inf_t1_0, kf.P_inf_t[1][0])
    np.testing.assert_array_almost_equal(e_xi_t1_0, kf.xi_t[1][0])
    np.testing.assert_array_almost_equal(e_P_star_t1_0, kf.P_star_t[1][0])


    # Test period 1 result
    e_P_inf_t1_0 = np.zeros([2, 2])
    e_P_star_t1_0 = np.array([[5 + 2 * q1 + q2, 3 + q1 + q2], 
                              [3 + q1 + q2, 2 + q1 + 2 * q2]]) * \
                                      theta_ll_1d_diffuse[2]
    y2 = Yt_1d_full[1][0][0]
    y1 = Yt_1d_full[0][0][0]
    e_xi_t1_0 = np.array([[2 * y2 - y1], [y2 - y1]])
    
    np.testing.assert_array_almost_equal(e_P_inf_t1_0, kf.P_inf_t[2][0])
    np.testing.assert_array_almost_equal(e_xi_t1_0, kf.xi_t[2][0])
    np.testing.assert_array_almost_equal(e_P_star_t1_0, kf.P_star_t[2][0])


    # Test period 2 result, should return same result as _sequential_update()
    P_inf_t1_0 = kf.P_inf_t[3][0].copy()
    P_star_t1_0 = kf.P_star_t[3][0].copy()
    xi_t1_0 = kf.xi_t[3][0].copy()

    kf._sequential_update(2)
    np.testing.assert_array_almost_equal(P_inf_t1_0, np.zeros([2, 2]))
    np.testing.assert_array_almost_equal(xi_t1_0, kf.xi_t[3][0])
    np.testing.assert_array_almost_equal(P_star_t1_0, kf.P_star_t[3][0])


def test_sequential_update_diffuse_ll_Upsilon_inf0(ft_ll_mvar_diffuse,
        theta_ll_mvar_diffuse, Yt_mvar_diffuse):
    """
    For ll model, only measurements across time can reduce rank of P_inf_t
    """
    t = 1
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse, for_smoother=True)
    kf.init_attr(theta_ll_mvar_diffuse)
    for t_ in range(t):
        kf._sequential_update_diffuse(t_)
    
    # Test period 0 result
    e_P_inf_t1_0 = np.ones([2, 2])
    expected_q = 1
    
    np.testing.assert_array_almost_equal(e_P_inf_t1_0, kf.P_inf_t[1][0])
    assert expected_q == kf.q
    
    # Test update when Upsilon_inf = 0
    index = 2
    ob = index - 1
    t = 0
    l_t, _, _ = linalg.ldl(kf.Rt[t])
    l_inv, _ = linalg.lapack.dtrtri(l_t, lower=True)
    R_t = l_inv.dot(kf.Rt[t])
    H_t = (l_inv.dot(kf.Ht[t]))[ob:index]
    D_t = (l_inv.dot(kf.Dt[t]))[ob:index]
    Upsilon = H_t.dot(kf.P_star_t[t][ob]).dot(H_t.T) + R_t[ob][ob]
    K = kf.P_star_t[t][ob].dot(H_t.T) / Upsilon
    v = l_inv.dot(kf.Yt[t])[ob] - H_t.dot(kf.xi_t[t][ob]) - D_t.dot(kf.Xt[t])
    expected_xi_t_11 = kf.xi_t[t][ob] + K * v
    expected_P_t_11 = kf.P_star_t[t][ob] - kf.P_star_t[t][ob].dot(
            (K.dot(H_t)).T)
    expected_P_t1_0 = kf.Ft[t].dot(expected_P_t_11).dot(
            kf.Ft[t].T) + kf.Qt[t]
    expected_xi_t1_0 = kf.Ft[t].dot(expected_xi_t_11) + \
            kf.Bt[t].dot(kf.Xt[t])
    np.testing.assert_array_almost_equal(expected_xi_t_11, 
            kf.xi_t[t][kf.n_t[t]])
    np.testing.assert_array_almost_equal(expected_P_t_11, 
            kf.P_star_t[t][kf.n_t[t]])
    np.testing.assert_array_almost_equal(expected_P_t1_0, 
            kf.P_star_t[t+1][0])
    np.testing.assert_array_almost_equal(expected_xi_t1_0, 
            kf.xi_t[t+1][0])


def test_sequential_update_diffuse_update_multiple_q(ft_q,
        theta_ll_mvar_diffuse, Yt_q):
    """
    For ll model, only measurements across time can reduce rank of P_inf_t
    """
    t = 1
    kf = Filter(ft_q, Yt_q, for_smoother=True)
    kf.init_attr(theta_ll_mvar_diffuse)
    kf._sequential_update_diffuse(0)
    assert kf.q == 0


def test_sequential_update_diffuse_ll_equivalent(ft_ll_mvar_diffuse,
        ft_ll_mvar_1d, Yt_mvar_diffuse_missing, Yt_mvar_1d, 
        theta_ll_mvar_diffuse):
    """
    Test in the case of misisng values such that at most 1 measurement present
    at time t, whether we get same result as 1d case
    """
    kf_mvar = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    kf_mvar.fit(theta_ll_mvar_diffuse)

    kf_1d = Filter(ft_ll_mvar_1d, Yt_mvar_1d, for_smoother=True)
    kf_1d.fit(theta_ll_mvar_diffuse)
    
    for t_ in range(kf_mvar.T-1):
        np.testing.assert_array_almost_equal(kf_1d.P_star_t[t_][0], 
                kf_mvar.P_star_t[t_][0])
        np.testing.assert_array_almost_equal(kf_1d.P_inf_t[t_][0], 
                kf_mvar.P_inf_t[t_][0])
        np.testing.assert_array_almost_equal(kf_1d.xi_t[t_][0], 
                kf_mvar.xi_t[t_][0])


# Test get_filtered_val
def test_get_filtered_val_not_filtered(ft_ll_mvar_1d, Yt_mvar_diffuse_missing):
    """
    Test error message when fit is not run
    """
    kf = Filter(ft_ll_mvar_1d, Yt_mvar_diffuse_missing, for_smoother=True)
    with pytest.raises(TypeError) as error:
        kf.get_filtered_val()
    expected_result = 'The Kalman filter object is not fitted yet'
    result = str(error.value)
    assert result == expected_result


def test_get_filtered_val_missing(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, 
        theta_ll_mvar_diffuse):
    """
    Test missing measurements handling
    """
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse)

    Yt_filtered, Yt_filtered_cov, _, _ = kf.get_filtered_val()
    np.testing.assert_array_equal(kf.Ht[2].dot(kf.xi_t[2][0]), Yt_filtered[2])


def test_get_filtered_val_all_xi(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    """
    Test df with all xi
    """
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse)

    Yt_filtered, Yt_filtered_cov, xi_t, P_t = kf.get_filtered_val()
    np.testing.assert_array_equal(xi_t[2], kf.xi_t[2][0])
    np.testing.assert_array_equal(P_t[2], np.nan * np.ones(P_t[2].shape))
    np.testing.assert_array_equal(P_t[3], kf.P_star_t[3][0])


def test_get_filtered_y_selected_xi(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    """
    Test df with selected xi
    """
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse)

    Yt_filtered, Yt_filtered_cov, xi_t, P_t = kf.get_filtered_val(xi_col=[1])
    np.testing.assert_array_equal(xi_t[2], kf.xi_t[2][0][[1]])
    np.testing.assert_array_equal(P_t[2], np.nan * np.ones(P_t[2].shape))
    np.testing.assert_array_equal(P_t[3], kf.P_star_t[3][0][[1]][:,[1]])


def test_get_filtered_y_no_xi(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    """
    Test df without xi
    """
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse)

    Yt_filtered, Yt_filtered_cov, xi_t, P_t = kf.get_filtered_val(is_xi=False)
    assert np.isnan(xi_t[-1])
    assert np.isnan(P_t[-1])


# Test get_filtered_state
def test_get_filtered_state_t(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    """
    get values at t < self.T
    """
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse)
    
    t = kf.T - 1
    result = kf.get_filtered_state(t)
    expected_result = {'P_star_t': kf.P_star_t[t][0],
            'P_inf_t': kf.P_inf_t[t][0],
            'xi_t': kf.xi_t[t][0],
            'q': 0}
    for i in ['P_star_t', 'P_inf_t', 'xi_t']:
        np.testing.assert_array_equal(result[i], expected_result[i])
    assert result['q'] == expected_result['q']


def test_get_filtered_state_T(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse)
    
    t = kf.T
    result = kf.get_filtered_state(t)
    expected_result = {'P_star_t': kf.P_T1,
            'P_inf_t':  np.zeros([2, 2]),
            'xi_t': kf.xi_T1,
            'q': 0}
    for i in ['P_star_t', 'P_inf_t', 'xi_t']:
        np.testing.assert_array_equal(result[i], expected_result[i])
    assert result['q'] == expected_result['q']


def test_get_filtered_state_diffuse(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse)
    
    t = 1
    with pytest.raises(ValueError) as error:
        result = kf.get_filtered_state(t)
    expected_result = 'Diffuse state at time 1'
    result = str(error.value)
    assert expected_result == result


def test_get_filtered_state_max(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse)
    
    t = 6
    with pytest.raises(ValueError) as error:
        result = kf.get_filtered_state(t)
    expected_result = 'Maximum t allowed is 4'
    result = str(error.value)
    assert expected_result == result


def test_get_filtered_state_not_fitted(ft_ll_mvar_diffuse, 
        Yt_mvar_diffuse_missing, theta_ll_mvar_diffuse):
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    
    t = 6
    with pytest.raises(ValueError) as error:
        result = kf.get_filtered_state(t)
    expected_result = 'Kalman filter is not fitted yet'
    result = str(error.value)
    assert expected_result == result


# Test override from init_state
def test_over_ride(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    """
    Force a diffuse kalman filter to become regular filter
    """
    init_val = {'P_star_t': np.zeros([2, 2]), 
            'xi_t': 100 * np.ones([2, 1]), 'q': 0}
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse, init_state=init_val)
    
    t = 0
    result = kf.get_filtered_state(t)
    expected_result = {'P_star_t': np.zeros([2, 2]),
            'P_inf_t':  np.zeros([2, 2]),
            'xi_t': 100 * np.ones([2, 1]),
            'q': 0}
    for i in ['P_star_t', 'P_inf_t', 'xi_t']:
        np.testing.assert_array_equal(result[i], expected_result[i])
    assert result['q'] == expected_result['q']


def test_over_ride_error_input(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    """
    Raises exception when init_state wrong input
    """
    init_val = {'P_star_t': np.zeros([2]), 
            'xi_t': 100 * np.ones([2, 1]), 'q': 0}
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing, for_smoother=True)
    with pytest.raises(ValueError) as error:
        kf.fit(theta_ll_mvar_diffuse, init_state=init_val)
    expected_result = 'User-specified P_star_t does not have 2 dimensions'
    result = str(error.value)
    assert expected_result == result
