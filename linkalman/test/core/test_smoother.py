import pytest
import numpy as np
from scipy import linalg
import pandas as pd
from linkalman.core import Filter, Smoother
from copy import deepcopy
from linkalman.core.utils import *


# Test init_attr_smoother
def test_init_attr_smoother(ft_rw_1, theta_rw, Yt_1d, Xt_1d):
    """
    Test normal run
    """
    kf = Filter(ft_rw_1, for_smoother=True)
    kf.fit(theta_rw, Yt_1d, Xt_1d)
    ks = Smoother()
    ks.init_attr_smoother(kf)

    e_r0 = np.zeros([1, 1])
    e_N0 = np.zeros([1, 1])
    
    r0 = ks.r0_t[kf.T-1]
    N0 = ks.N0_t[kf.T-1]
    np.testing.assert_array_equal(e_r0, r0)
    np.testing.assert_array_equal(e_N0, N0)
    assert ks.r1_t == None


def test_init_attr_smoother_not_fitted(ft_rw_1, theta_rw, Yt_1d, Xt_1d):
    """
    Test error message Kalman filter not fitted
    """
    kf = Filter(ft_rw_1, for_smoother=True)
    ks = Smoother()
    with pytest.raises(TypeError) as error:
        ks.init_attr_smoother(kf)
    msg = str(error.value)
    e_msg = 'The Kalman filter object is not fitted yet'

    assert msg == e_msg


def test_init_attr_smoother_not_for_smoother(ft_rw_1, theta_rw, Yt_1d, Xt_1d):
    """
    Test error message Kalman filter not for smoother
    """
    kf = Filter(ft_rw_1, for_smoother=False)
    kf.fit(theta_rw, Yt_1d, Xt_1d)
    ks = Smoother()
    with pytest.raises(TypeError) as error:
        ks.init_attr_smoother(kf)
    msg = str(error.value)
    e_msg = 'The Kalman filter object is not for smoothers'

    assert msg == e_msg


def test_init_attr_smoother_diffuse(ft_ll_mvar_diffuse, 
        theta_ll_mvar_diffuse, Yt_mvar_diffuse_missing):
    """
    Test initialization for diffuse smoother
    """
    kf = Filter(ft_ll_mvar_diffuse, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse, Yt_mvar_diffuse_missing)
    ks = Smoother()
    ks.init_attr_smoother(kf)
    e_r1 = np.zeros([2, 1])
    e_N1 = np.zeros([2, 2])
    np.testing.assert_array_equal(e_r1, ks.r1_t[ks.t_q-1])
    np.testing.assert_array_equal(e_N1, ks.N1_t[ks.t_q-1])
    assert ks.t_q == 3


# Test sequential_smooth
def test_sequential_smooth(ft_ar2_mvar, theta_ar2_mvar,
        Yt_ar2_mvar, Xt_ar2_mvar):
    """
    Test whether it gives the same result as using direct approach
    """
    kf = Filter(ft_ar2_mvar, for_smoother=True)
    kf.fit(theta_ar2_mvar, Yt_ar2_mvar, Xt_ar2_mvar)
    ks = Smoother()
    ks.fit(kf)

    # Test period 1
    t = 0
    Ft = ks.Ft[t]
    Ht = ks.Ht[t]
    xi_t = ks.xi_t[t][0]
    P_t = ks.P_star_t[t][0]
    Rt = ks.Rt[t]
    Dt = ks.Dt[t]
    Upsilon_t = Ht.dot(P_t).dot(Ht.T) + Rt
    Upsilon_t_inv = linalg.pinv(Upsilon_t)
    Kt = P_t.dot(Ht.T).dot(Upsilon_t_inv)
    vt = ks.Yt[t] - Ht.dot(xi_t) - Dt.dot(ks.Xt[t])
    Lt= Ft.dot(ks.I - Kt.dot(Ht))
    rt = ks.r0_t[t+1] 
    Nt = ks.N0_t[t+1]
    r1t = (Ht.T).dot(Upsilon_t_inv).dot(vt) + (Lt.T).dot(rt)
    N1t = (Ht.T).dot(Upsilon_t_inv).dot(Ht) + \
            (Lt.T).dot(Nt).dot(Lt)
    e_xi_t_T = xi_t + P_t.dot(r1t)
    e_P_t_T = P_t - P_t.dot(N1t).dot(P_t)
    e_P_cov_t_t1 = P_t.dot(Lt.T).dot(ks.I - Nt.dot(ks.P_star_t[t+1][0]))
    e_PL = ks.P_star_t[t][ks.n_t[t]].dot(Ft.T)
    np.testing.assert_array_almost_equal(e_xi_t_T, ks.xi_t_T[t])
    np.testing.assert_array_almost_equal(e_P_t_T, ks.P_t_T[t])
    np.testing.assert_array_almost_equal(e_PL, P_t.dot(Lt.T))
    np.testing.assert_array_almost_equal(e_P_cov_t_t1, ks.Pcov_t_t1[t])
    np.testing.assert_array_almost_equal(r1t, ks.r0_t[t])
    np.testing.assert_array_almost_equal(N1t, ks.N0_t[t])

    # Test period 2
    t = 1
    n_t = ks.n_t[t]
    Ft = ks.Ft[t]
    Ht = ks.Ht[t][:n_t]
    xi_t = ks.xi_t[t][0]
    P_t = ks.P_star_t[t][0]
    Rt = ks.Rt[t][:n_t][:,:n_t]
    Dt = ks.Dt[t][:n_t]
    Yt = ks.Yt[t][:n_t]
    Upsilon_t = Ht.dot(P_t).dot(Ht.T) + Rt
    Upsilon_t_inv = linalg.pinv(Upsilon_t)
    Kt = P_t.dot(Ht.T).dot(Upsilon_t_inv)
    vt = Yt - Ht.dot(xi_t) - Dt.dot(ks.Xt[t])
    Lt= Ft.dot(ks.I - Kt.dot(Ht))
    rt = ks.r0_t[t+1] 
    Nt = ks.N0_t[t+1]
    r1t = (Ht.T).dot(Upsilon_t_inv).dot(vt) + (Lt.T).dot(rt)
    N1t = (Ht.T).dot(Upsilon_t_inv).dot(Ht) + \
            (Lt.T).dot(Nt).dot(Lt)
    e_xi_t_T = xi_t + P_t.dot(r1t)
    e_P_t_T = P_t - P_t.dot(N1t).dot(P_t)
    
    e_P_cov_t_t1 = P_t.dot(Lt.T).dot(ks.I - Nt.dot(ks.P_star_t[t+1][0]))
    
    np.testing.assert_array_almost_equal(e_xi_t_T, ks.xi_t_T[t])
    np.testing.assert_array_almost_equal(e_P_t_T, ks.P_t_T[t])
    np.testing.assert_array_almost_equal(e_P_cov_t_t1, ks.Pcov_t_t1[t])
    
    np.testing.assert_array_almost_equal(r1t, ks.r0_t[t])
    np.testing.assert_array_almost_equal(N1t, ks.N0_t[t])


    # Test period 3
    t = 2
    n_t = ks.n_t[t]
    Ft = ks.Ft[t]
    xi_t = ks.xi_t[t][0]
    P_t = ks.P_star_t[t][0]
    rt = ks.r0_t[t+1] 
    Nt = ks.N0_t[t+1]
    r1t = (Ft.T).dot(rt)
    N1t = (Ft.T).dot(Nt).dot(Ft)
    e_xi_t_T = xi_t + P_t.dot(r1t)
    e_P_t_T = P_t - P_t.dot(N1t).dot(P_t)
    e_P_cov_t_t1 = P_t.dot(Ft.T).dot(ks.I - Nt.dot(ks.P_star_t[t+1][0]))
    
    np.testing.assert_array_almost_equal(e_xi_t_T, ks.xi_t_T[t])
    np.testing.assert_array_almost_equal(e_P_t_T, ks.P_t_T[t])
    np.testing.assert_array_almost_equal(e_P_cov_t_t1, ks.Pcov_t_t1[t])
    
    np.testing.assert_array_almost_equal(r1t, ks.r0_t[t])
    np.testing.assert_array_almost_equal(N1t, ks.N0_t[t])


    # Test period 4
    t = 3
    n_t = ks.n_t[t]
    Ft = ks.Ft[t]
    Ht = ks.Ht[t][:n_t]
    xi_t = ks.xi_t[t][0]
    P_t = ks.P_star_t[t][0]
    Rt = ks.Rt[t][:n_t][:,:n_t]
    Dt = ks.Dt[t][:n_t]
    Yt = ks.Yt[t][:n_t]
    Upsilon_t = Ht.dot(P_t).dot(Ht.T) + Rt
    Upsilon_t_inv = linalg.pinv(Upsilon_t)
    Kt = P_t.dot(Ht.T).dot(Upsilon_t_inv)
    vt = Yt - Ht.dot(xi_t) - Dt.dot(ks.Xt[t])
    Lt= Ft.dot(ks.I - Kt.dot(Ht))
    rt = np.zeros([2, 1])
    Nt = np.zeros([2, 2])
    r1t = (Ht.T).dot(Upsilon_t_inv).dot(vt) + (Lt.T).dot(rt)
    N1t = (Ht.T).dot(Upsilon_t_inv).dot(Ht) + \
            (Lt.T).dot(Nt).dot(Lt)
    e_xi_t_T = xi_t + P_t.dot(r1t)
    e_P_t_T = P_t - P_t.dot(N1t).dot(P_t)
    
    np.testing.assert_array_almost_equal(e_xi_t_T, ks.xi_t_T[t])
    np.testing.assert_array_almost_equal(e_P_t_T, ks.P_t_T[t])
    
    np.testing.assert_array_almost_equal(r1t, ks.r0_t[t])
    np.testing.assert_array_almost_equal(N1t, ks.N0_t[t])


# Test sequential_smooth_diffuse
def test_sequential_smooth_diffuse_missing(ft_ll_mvar_diffuse,
        theta_ll_mvar_diffuse, Yt_mvar_diffuse_smooth):
    """
    Test cases with diffuse initial conditions. I test period 1 
    in a seperate test, as the sequential smoother uses a different
    system of r and N. 
    """
    kf = Filter(ft_ll_mvar_diffuse, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse, Yt_mvar_diffuse_smooth)
    ks = Smoother()
    ks.fit(kf)
    
    # Test period 2
    t = 1
    Ft = ks.Ft[t]
    n_t = ks.n_t[t]
    xi_t = ks.xi_t[t][0]
    P_inf_t = ks.P_inf_t[t][0]
    P_star_t = ks.P_star_t[t][0]
    r0_t = ks.r0_t[t+1]
    r1_t = ks.r1_t[t+1]
    N0_t = ks.N0_t[t+1]
    N1_t = ks.N1_t[t+1]
    N2_t = ks.N2_t[t+1]

    r0_1t = (Ft.T).dot(r0_t)
    r1_1t = (Ft.T).dot(r1_t) 
    e_xi_T = xi_t + P_star_t.dot(r0_1t) + P_inf_t.dot(r1_1t)

    N0_1t = (Ft.T).dot(N0_t).dot(Ft)
    N1_1t = (Ft.T).dot(N1_t).dot(Ft)
    N2_1t = (Ft.T).dot(N2_t).dot(Ft)
    e_P_T = P_star_t - P_star_t.dot(N0_1t).dot(P_star_t) - P_star_t.dot(
            N1_1t).dot(P_inf_t) - P_inf_t.dot(N1_1t).dot(P_star_t) - \
            P_inf_t.dot(N2_1t).dot(P_inf_t)
    e_P_cov_t_t1 = (P_star_t.dot(Ft.T) + P_inf_t.dot(Ft.T)).dot(ks.I - N1_t.dot(
            ks.P_inf_t[t+1][0]) - N0_t.dot(ks.P_star_t[t+1][0])) - P_inf_t.dot(
            Ft.T).dot(N2_t.dot(ks.P_inf_t[t+1][0]) + N1_t.dot(ks.P_star_t[t+1][0]))
    
    np.testing.assert_array_almost_equal(e_xi_T, ks.xi_t_T[t])
    np.testing.assert_array_almost_equal(e_P_T, ks.P_t_T[t])
    np.testing.assert_array_almost_equal(e_P_cov_t_t1, ks.Pcov_t_t1[t])
    
    np.testing.assert_array_almost_equal(r0_1t, ks.r0_t[t])
    np.testing.assert_array_almost_equal(r1_1t, ks.r1_t[t])
    np.testing.assert_array_almost_equal(N0_1t, ks.N0_t[t])
    np.testing.assert_array_almost_equal(N1_1t, ks.N1_t[t])
    np.testing.assert_array_almost_equal(N2_1t, ks.N2_t[t])

    # Test period 3
    t = 2
    Ft = ks.Ft[t]
    n_t = ks.n_t[t]
    xi_t = ks.xi_t[t][0]
    Rt = ks.Rt[t][:n_t][:,:n_t]
    Ht = ks.Ht[t][:n_t]
    Dt = ks.Dt[t][:n_t]
    Yt = ks.Yt[t][:n_t]
    vt = Yt - Ht.dot(xi_t) - Dt.dot(ks.Xt[t])
    P_inf_t = ks.P_inf_t[t][0]
    P_star_t = ks.P_star_t[t][0]
    Upsilon_inf_t = Ht.dot(P_inf_t).dot(Ht.T)
    Upsilon_star_t = Ht.dot(P_star_t).dot(Ht.T) + Rt
    Upsilon_1 = linalg.pinv(Upsilon_inf_t)
    Upsilon_2 = -Upsilon_1.dot(Upsilon_star_t).dot(Upsilon_1)
    K0 = P_inf_t.dot(Ht.T).dot(Upsilon_1)
    K1 = P_star_t.dot(Ht.T).dot(Upsilon_1) + \
            P_inf_t.dot(Ht.T).dot(Upsilon_2)
    L0 = Ft.dot(ks.I - K0.dot(Ht))
    L1 = -Ft.dot(K1.dot(Ht))
    r0_t = ks.r0_t[t+1]
    r1_t = np.zeros([2, 1])
    N0_t = ks.N0_t[t+1]
    N1_t = np.zeros([2, 2])
    N2_t = np.zeros([2, 2])

    r0_1t = (L0.T).dot(r0_t)
    r1_1t = (Ht.T).dot(Upsilon_1).dot(vt) + \
            (L0.T).dot(r1_t) + (L1.T).dot(r0_t)
    e_xi_T = xi_t + P_star_t.dot(r0_1t) + P_inf_t.dot(r1_1t)

    N0_1t = (L0.T).dot(N0_t).dot(L0)
    N1_1t = (Ht.T).dot(Upsilon_1).dot(Ht) + (L0.T).dot(N1_t).dot(L0.T) + \
            (L1.T).dot(N0_t).dot(L0) + (L0.T).dot(N0_t).dot(L1)
    N2_1t = (Ht.T).dot(Upsilon_2).dot(Ht) + (L0.T).dot(N2_t).dot(L0) + \
            (L0.T).dot(N1_t).dot(L1) + (L1.T).dot(N1_t.T).dot(L0) + \
            (L1.T).dot(N0_t).dot(L1)
    e_P_T = P_star_t - P_star_t.dot(N0_1t).dot(P_star_t) - P_star_t.dot(
            N1_1t).dot(P_inf_t) - P_inf_t.dot(N1_1t).dot(P_star_t) - \
            P_inf_t.dot(N2_1t).dot(P_inf_t)
    e_P_cov_t_t1 = (P_star_t.dot(L0.T) + P_inf_t.dot(L1.T)).dot(ks.I - N1_t.dot(
            ks.P_inf_t[t+1][0]) - N0_t.dot(ks.P_star_t[t+1][0])) - P_inf_t.dot(
            L0.T).dot(N2_t.dot(ks.P_inf_t[t+1][0]) + N1_t.dot(ks.P_star_t[t+1][0]))
    
    np.testing.assert_array_almost_equal(e_xi_T, ks.xi_t_T[t])
    np.testing.assert_array_almost_equal(e_P_T, ks.P_t_T[t])
    np.testing.assert_array_almost_equal(e_P_cov_t_t1, ks.Pcov_t_t1[t])
    
    np.testing.assert_array_almost_equal(r0_1t, ks.r0_t[t])
    np.testing.assert_array_almost_equal(r1_1t, ks.r1_t[t])
    np.testing.assert_array_almost_equal(N0_1t, ks.N0_t[t])
    np.testing.assert_array_almost_equal(N1_1t, ks.N1_t[t])
    np.testing.assert_array_almost_equal(N2_1t, ks.N2_t[t])

    # Test period 4
    t = 3
    n_t = ks.n_t[t]
    Ft = ks.Ft[t]
    Ht = ks.Ht[t][:n_t]
    xi_t = ks.xi_t[t][0]
    P_t = ks.P_star_t[t][0]
    Rt = ks.Rt[t][:n_t][:,:n_t]
    Dt = ks.Dt[t][:n_t]
    Yt = ks.Yt[t][:n_t]
    Upsilon_t = Ht.dot(P_t).dot(Ht.T) + Rt
    Upsilon_t_inv = linalg.pinv(Upsilon_t)
    Kt = P_t.dot(Ht.T).dot(Upsilon_t_inv)
    vt = Yt - Ht.dot(xi_t) - Dt.dot(ks.Xt[t])
    Lt= Ft.dot(ks.I - Kt.dot(Ht))
    rt = np.zeros([2, 1])
    Nt = np.zeros([2, 2])
    r1t = (Ht.T).dot(Upsilon_t_inv).dot(vt) + (Lt.T).dot(rt)
    N1t = (Ht.T).dot(Upsilon_t_inv).dot(Ht) + \
            (Lt.T).dot(Nt).dot(Lt)
    e_xi_t_T = xi_t + P_t.dot(r1t)
    e_P_t_T = P_t - P_t.dot(N1t).dot(P_t)
    
    np.testing.assert_array_almost_equal(e_xi_t_T, ks.xi_t_T[t])
    np.testing.assert_array_almost_equal(e_P_t_T, ks.P_t_T[t])
    
    np.testing.assert_array_almost_equal(r1t, ks.r0_t[t])
    np.testing.assert_array_almost_equal(N1t, ks.N0_t[t])


def test_sequential_smooth_diffuse_vec(ft_ll_mvar_diffuse,
        theta_ll_mvar_diffuse, Yt_mvar_diffuse_smooth_vec):
    """
    Test cases with diffuse initial conditions. I test the case
    where there are multiple observations at time t, and only the 
    first measurement matters (in local linear models). 
    """
    kf = Filter(ft_ll_mvar_diffuse, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse, Yt_mvar_diffuse_smooth_vec)
    ks = Smoother()
    ks.fit(kf)

    # Test period 3
    t = 2
    n_t = ks.n_t[t]
    Ft = ks.Ft[t]
    Ht = ks.Ht[t][:n_t]
    xi_t = ks.xi_t[t][0]
    P_t = ks.P_star_t[t][0]
    Rt = ks.Rt[t][:n_t][:,:n_t]
    Dt = ks.Dt[t][:n_t]
    Yt = ks.Yt[t][:n_t]
    Upsilon_t = Ht.dot(P_t).dot(Ht.T) + Rt
    Upsilon_t_inv = linalg.pinv(Upsilon_t)
    Kt = P_t.dot(Ht.T).dot(Upsilon_t_inv)
    vt = Yt - Ht.dot(xi_t) - Dt.dot(ks.Xt[t])
    Lt= Ft.dot(ks.I - Kt.dot(Ht))
    rt = np.zeros([2, 1])
    Nt = np.zeros([2, 2])
    r1t = (Ht.T).dot(Upsilon_t_inv).dot(vt) + (Lt.T).dot(rt)
    N1t = (Ht.T).dot(Upsilon_t_inv).dot(Ht) + \
            (Lt.T).dot(Nt).dot(Lt)
    e_xi_t_T = xi_t + P_t.dot(r1t)
    e_P_t_T = P_t - P_t.dot(N1t).dot(P_t)
    
    np.testing.assert_array_almost_equal(e_xi_t_T, ks.xi_t_T[t])
    np.testing.assert_array_almost_equal(e_P_t_T, ks.P_t_T[t])
    
    np.testing.assert_array_almost_equal(r1t, ks.r0_t[t])
    np.testing.assert_array_almost_equal(N1t, ks.N0_t[t])
    
    # Test period 2 (refer to Koopman 1997)
    t = 1
    Ft = ks.Ft[t]
    n_t = ks.n_t[t]
    xi_t = ks.xi_t[t][0]
    Rt = ks.Rt[t][:n_t][:,:n_t]
    Ht = ks.Ht[t][:n_t]
    Dt = ks.Dt[t][:n_t]
    Yt = ks.Yt[t][:n_t]
    vt = Yt - Ht.dot(xi_t) - Dt.dot(ks.Xt[t])
    P_inf_t = ks.P_inf_t[t][0]
    P_star_t = ks.P_star_t[t][0]
    r_star_t = ks.r0_t[t+1]
    N_star_t = ks.N0_t[t+1]
    Upsilon_inf_t = Ht.dot(P_inf_t).dot(Ht.T)
    Upsilon_star_t = Ht.dot(P_star_t).dot(Ht.T) + Rt

    K_ = np.array([[0, 0.5], [0.5, -0.25]])
    star = K_.dot(Upsilon_star_t).dot(K_)
    L_ = np.eye(2)
    L_[0][1] = -star[1][0]/star[1][1]

    L_star_L = L_.dot(star).dot(L_.T)
    L_[1][1] = 1/np.sqrt(L_star_L[1][1])
    J = (L_.dot(K_)).T
    J1 = J[:, 0:1]
    J2 = J[:, 1:2]
    
    F_star_neg = J2.dot(J2.T) 
    F_inf_neg = J1.dot(J1.T)
    F2 = -F_inf_neg.dot(Upsilon_star_t).dot(F_inf_neg)
    M_star = Ft.dot(P_star_t).dot(Ht.T)
    M_inf =  Ft.dot(P_inf_t).dot(Ht.T)
    K_star = M_star.dot(F_star_neg) + M_inf.dot(F_inf_neg)
    K_inf = M_star.dot(F_inf_neg) - M_inf.dot(F_inf_neg).dot(
            Upsilon_star_t).dot(F_inf_neg)
    L_star = Ft - K_star.dot(Ht)
    L_inf = -K_inf.dot(Ht)
    
    r_inf_1t = (Ht.T).dot(F_inf_neg).dot(vt) + (L_inf.T).dot(r_star_t)
    r_star_1t = (Ht.T).dot(F_star_neg).dot(vt) + (L_star.T).dot(r_star_t)
    N_star_1t = (Ht.T).dot(F_star_neg).dot(Ht) + (L_star.T).dot(N_star_t).dot(L_star)
    N_1_1t = (Ht.T).dot(F_inf_neg).dot(Ht) + (L_inf.T).dot(N_star_t).dot(
            L_star) + (L_star.T).dot(N_star_t).dot(L_inf) 
    N_2_1t = (Ht.T).dot(F2).dot(Ht) + (L_inf.T).dot(N_star_t).dot(L_inf)
    e_xi_t_T = xi_t + P_star_t.dot(r_star_1t) + P_inf_t.dot(r_inf_1t)
    e_P_t_T = P_star_t - P_star_t.dot(N_star_1t).dot(P_star_t) - \
            P_star_t.dot(N_1_1t).dot(P_inf_t) - P_inf_t.dot(N_1_1t).dot(
            P_star_t) - P_inf_t.dot(N_2_1t).dot(P_inf_t)
    e_Pcov = (P_star_t.dot(L_star.T) + P_inf_t.dot(L_inf.T)).dot(ks.I - \
            N_star_t.dot(ks.P_star_t[t+1][0]))

    np.testing.assert_array_almost_equal(e_xi_t_T, ks.xi_t_T[t])
    np.testing.assert_array_almost_equal(e_P_t_T, ks.P_t_T[t])
    np.testing.assert_array_almost_equal(e_Pcov, ks.Pcov_t_t1[t])


# Test get_smoothed_val
def test_get_smoothed_val_not_smoothed(ft_ll_mvar_1d):
    """
    Test error message when fit is not run
    """
    kf = Filter(ft_ll_mvar_1d, for_smoother=True)
    ks = Smoother()
    with pytest.raises(TypeError) as error:
        ks.get_smoothed_val()
    expected_result = 'The Kalman smoother object is not fitted yet'
    result = str(error.value)
    assert result == expected_result


def test_get_smoothed_val_all_xi(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    """
    Test df with all xi
    """
    kf = Filter(ft_ll_mvar_diffuse, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse, Yt_mvar_diffuse_missing)
    ks = Smoother()
    ks.fit(kf)

    y_t_T, yP_t_T, xi_T, P_T = ks.get_smoothed_val()
    np.testing.assert_array_equal(xi_T[2], ks.xi_t_T[2])
    np.testing.assert_array_equal(P_T[2], ks.P_t_T[2])
    np.testing.assert_array_equal(P_T[3], ks.P_t_T[3])
   
    Mt = ft_ll_mvar_diffuse(theta_ll_mvar_diffuse, 4)
    Rt = Mt['Rt'][0]
    Dt = Mt['Dt'][0]
    Ht = Mt['Ht'][0]

    # Test smoothed y
    R2_0 = np.array([[0.5 - 0.4 / 0.6 * 0.4]])
    B0 = 0.4 / 0.6
    delta_H0 = Ht[0:1] - B0 * Ht[1:]
    eps0 = B0 * (Yt_mvar_diffuse_missing[0][1] - 
            Dt[1:].dot(ks.Xt[0]) - Ht[1:].dot(ks.xi_t_T[0]))
    y0 = Ht[0:1].dot(ks.xi_t_T[0]) + Dt[0:1].dot(ks.Xt[0]) + eps0

    yP_0 = (delta_H0.dot(ks.P_t_T[0]).dot(delta_H0.T) + R2_0).item()

    R2_2 = np.array([[0.6 - 0.4 / 0.5 * 0.4]])
    B2 = 0.4 / 0.5
    delta_H2 = Ht[1] - B2 * Ht[0]
    eps2 = B2 * (Yt_mvar_diffuse_missing[2][0] - 
            Dt[:1].dot(ks.Xt[2]) - Ht[:1].dot(ks.xi_t_T[2]))
    y2 = Ht[1:].dot(ks.xi_t_T[2]) + Dt[1:].dot(ks.Xt[2]) + eps2
    yP_2 = (delta_H2.dot(ks.P_t_T[2]).dot(delta_H2.T) + R2_2).item()
    expected_y_t_T = [np.array([[y0], [2]]),
            Ht.dot(ks.xi_t_T[1] + Dt.dot(ks.Xt[1])),
            np.array([[2.5], [y2]]), 
            np.array([[3], [5]])]
    expected_Pcov_T = [np.array([[yP_0, 0], [0, 0]]),
            ks.Ht[1].dot(ks.P_t_T[1]).dot(ks.Ht[1].T) + ks.Rt[1],
            np.array([[0, 0], [0, yP_2]]),
            np.zeros([2, 2])]
    for t in range(4):
        np.testing.assert_array_almost_equal(yP_t_T[t], expected_Pcov_T[t])
        np.testing.assert_array_almost_equal(y_t_T[t], expected_y_t_T[t])


def test_get_smoothed_y_selected_xi(ft_ll_mvar_diffuse, Yt_mvar_diffuse_missing,
        theta_ll_mvar_diffuse):
    """
    Test df with selected xi
    """
    kf = Filter(ft_ll_mvar_diffuse, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse, Yt_mvar_diffuse_missing)
    ks = Smoother()
    ks.fit(kf)

    _, _, xi_T, P_T = ks.get_smoothed_val(xi_col=[1])
    np.testing.assert_array_equal(xi_T[2], ks.xi_t_T[2][[1]])
    np.testing.assert_array_equal(P_T[2], ks.P_t_T[2][[1]][:,[1]])
