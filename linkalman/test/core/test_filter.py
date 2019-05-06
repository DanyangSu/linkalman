import pytest
import numpy as np
from scipy import linalg
import pandas as pd
from linkalman.core import Filter
from copy import deepcopy


# Test autocomplete x
def test_gen_xt(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Xt_ = None
    kf = Filter(Mt_)
    kf.gen_Xt(Xt_)
    expected_result = np.zeros((1, 1))
    result = kf.Xt[0]
    # np.testing.assert_equal(expected_result, result)


# Test if y_t is 1D or 2D array
def test_ckeck_consistence_2D(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    kf = Filter(Mt_)
    kf.Yt = [np.ones(3)]
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'y_t has the wrong dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Ft and xi_t
def test_check_consistence_Ft_xi_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Ft': [np.ones((5, 4))]})
    kf = Filter(Mt_)
    kf.Yt = deepcopy(Yt)
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'Ft and xi_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Ht and xi_t
def test_ckeck_consistence_Ht_xi_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Ht': [np.ones((5, 4))]})
    kf = Filter(Mt_)
    kf.Yt = deepcopy(Yt)
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'Ht and xi_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Ht and y_t
def test_ckeck_consistence_Ht_y_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Ht': [np.ones((5, 4))]})
    kf = Filter(Mt_)
    kf.Yt = deepcopy(Yt)
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'Ht and xi_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Bt and xi_t
def test_ckeck_consistence_Bt_xi_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Bt': [np.ones((5, 4))]})
    kf = Filter(Mt_)
    kf.Yt = deepcopy(Yt)
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'Bt and xi_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Bt and x_t
def test_ckeck_consistence_Bt_x_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Bt': [np.ones((3, 4))]})
    kf = Filter(Mt_)
    kf.Yt = deepcopy(Yt)
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'Bt and x_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Dt and y_t
def test_ckeck_consistence_Dt_y_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Dt': [np.ones((5, 2))]})
    kf = Filter(Mt_)
    kf.Yt = deepcopy(Yt)
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'Dt and y_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Dt and x_t
def test_ckeck_consistence_Dt_x_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Dt': [np.ones((4, 4))]})
    kf = Filter(Mt_)
    kf.Yt = deepcopy(Yt)
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'Dt and x_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Qt and xi_t
def test_ckeck_consistence_Qt_xi_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Qt': [np.ones((5, 5))]})
    kf = Filter(Mt_)
    kf.Yt = deepcopy(Yt)
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'Qt and xi_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Rt and y_t
def test_ckeck_consistence_Rt_y_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Rt': [np.ones((3, 3))]})
    kf = Filter(Mt_)
    kf.Yt = deepcopy(Yt)
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'Rt and y_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test if y_t is a vector
def test_ckeck_consistence_y_vector(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    kf = Filter(Mt_)
    kf.Yt = [np.ones((4, 2))]
    kf.gen_Xt(Xt)
    with pytest.raises(ValueError) as error:
        kf.check_consistence()
    expected_result = 'y_t must be a vector'
    result = str(error.value)
    assert result == expected_result
