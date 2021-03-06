import pytest
import numpy as np
from scipy import linalg
import pandas as pd
from linkalman.core import Filter, Smoother
from copy import deepcopy
from linkalman.core.utils import *
from linkalman.models import BaseConstantModel as BCM


# Test _E_delta2
def test_E_delta2(ft_ll_mvar_diffuse,
        theta_ll_mvar_diffuse, Yt_mvar_diffuse_smooth):
    """
    Test normal run 
    """
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_smooth, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse)
    ks = Smoother()
    ks.fit(kf)

    t = 2
    theta_test = [i + 0.1 for i in theta_ll_mvar_diffuse]
    Mt = ft_ll_mvar_diffuse(theta_test, ks.T)
    delta2 = ks._E_delta2(Mt, t)

    nt = ks.xi_t_T[t] - Mt['Ft'][t-1].dot(ks.xi_t_T[t-1]) - \
            Mt['Bt'][t-1].dot(ks.Xt[t-1])
    e_delta2 = nt.dot(nt.T) + ks.P_t_T[t] + Mt['Ft'][t-1].dot(
            ks.P_t_T[t-1]).dot(Mt['Ft'][t-1].T) - Mt['Ft'][t-1].dot(
            ks.Pcov_t_t1[t-1]) - (ks.Pcov_t_t1[t-1].T).dot(Mt['Ft'][t-1].T)
    np.testing.assert_array_almost_equal(e_delta2, delta2)

    
# Test _E_chi2
def test_E_chi2(ft_ll_mvar_diffuse,
        theta_ll_mvar_diffuse, Yt_mvar_diffuse_smooth):
    """
    Test normal run 
    """
    kf = Filter(ft_ll_mvar_diffuse, Yt_mvar_diffuse_smooth, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse)
    ks = Smoother()
    ks.fit(kf)

    theta_test = theta_ll_mvar_diffuse + 1
    Mt = ft_ll_mvar_diffuse(theta_test, ks.T)


    # Test first missing
    t = 2
    chi2 = ks._E_chi2(Mt, t)
    Ht = Mt['Ht'][t][1]
    Dt = Mt['Dt'][t][1]
    e_chi = ks.Yt[t][0] - Ht.dot(ks.xi_t_T[t]) - \
            Dt.dot(ks.Xt[t])
    e_chi2 = e_chi.dot(e_chi.T) + Ht.dot(ks.P_t_T[t]).dot(Ht.T)
    np.testing.assert_array_equal(e_chi2, chi2)

    # # Test all present
    t = 3
    chi2 = ks._E_chi2(Mt, t)
    Ht = Mt['Ht'][t]
    Dt = Mt['Dt'][t]
    e_chi = ks.Yt[t] - Ht.dot(ks.xi_t_T[t]) - \
            Dt.dot(ks.Xt[t])
    e_chi2 = e_chi.dot(e_chi.T) + Ht.dot(ks.P_t_T[t]).dot(Ht.T)
    np.testing.assert_array_equal(e_chi2, chi2)


# Test Filter.get_LL
def test_get_LL_explosive_root(f_arma32):
    """
    Test how to handle explosive root in Filter
    """
    # Initialize the model
    x = 1  # used to calculate stationary mean
    model = BCM()
    model.set_f(f_arma32)
    theta_intend = np.array([0.1, 0.2, 0.25])
    theta_test = np.array([0.59358299, 0.91708496, 0.54634604])
    T = 250
    my_ft = lambda theta,T, **kwargs: ft(theta, f_arma32, T, **kwargs)
    x_col = ['const']
    Xt = pd.DataFrame({x_col[0]: x * np.ones(T)})

    # Build simulated data
    df, y_col, xi_col = model.simulated_data(
            input_theta=theta_intend, Xt=Xt)
    Xt = df_to_tensor(df, x_col)
    Yt = df_to_tensor(df, y_col)
    kf = Filter(my_ft, Yt, Xt)
    kf.fit(theta_test)
    result = kf.get_LL()
    assert not np.isnan(result)


# Test Smoother.G
def test_G_explosive_root(f_arma32):
    """
    Test how to handle explosive root in Smoother
    """
    # Initialize the model
    x = 1  # used to calculate stationary mean
    model = BCM()
    model.set_f(f_arma32)
    theta_intend = np.array([0.1, 0.2, 0.25])
    theta_test = np.array([0.59358299, 0.91708496, 0.54634604])
    T = 250
    my_ft = lambda theta,T, **kwargs: ft(theta, f_arma32, T, **kwargs)
    x_col = ['const']
    Xt = pd.DataFrame({x_col[0]: x * np.ones(T)})

    # Build simulated data
    df, y_col, xi_col = model.simulated_data(
            input_theta=theta_intend, Xt=Xt)
    Xt = df_to_tensor(df, x_col)
    Yt = df_to_tensor(df, y_col)
    kf = Filter(my_ft, Yt, Xt, for_smoother=True)
    kf.fit(theta_test)
    ks = Smoother()
    ks.fit(kf)
    result = ks.G(theta_test)
    assert not np.isnan(result)
