import pytest
import numpy as np
from scipy import linalg
import pandas as pd
from linkalman.core import Filter, Smoother
from copy import deepcopy
from linkalman.core.utils import *


# Test _E_delta2
def test_E_delta2(ft_ll_mvar_diffuse,
        theta_ll_mvar_diffuse, Yt_mvar_diffuse_smooth):
    """
    Test normal run 
    """
    kf = Filter(ft_ll_mvar_diffuse, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse, Yt_mvar_diffuse_smooth)
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
    kf = Filter(ft_ll_mvar_diffuse, for_smoother=True)
    kf.fit(theta_ll_mvar_diffuse, Yt_mvar_diffuse_smooth)
    ks = Smoother()
    ks.fit(kf)

    theta_test = [i + 0.1 for i in theta_ll_mvar_diffuse]
    Mt = ft_ll_mvar_diffuse(theta_test, ks.T)

    # Test all missing
    t = 1

    # Test first missing
    t = 2
    chi2 = ks._E_chi2(Mt, t)

    # Test all present
    t = 3
    et = Yt_mvar_diffuse_smooth[t] - ks.Ht

