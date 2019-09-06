import pytest
import numpy as np
from scipy import linalg
import pandas as pd
from linkalman.core.utils import *
from copy import deepcopy


# Test if Mt contains all keys
def test_ckeck_consistence_missing_keys(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.pop('Ft')
    Mt_.pop('xi_1_0')
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt[0], Xt[0])
    expected_result1 = 'Mt does not contain all required keys. ' + \
            "The missing keys are ['xi_1_0', 'Ft']"
    expected_result2 = 'Mt does not contain all required keys. ' + \
            "The missing keys are ['Ft', 'xi_1_0']"
    result = str(error.value)
    assert result in [expected_result1, expected_result2]


# Test if y_t is 1D or 2D array
def test_ckeck_consistence_2D(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Yt_ = [np.ones(3)]
    with pytest.raises(ValueError) as error:
        check_consistence(Mt, Yt_[0], Xt[0])
    expected_result = 'y_t has the wrong dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Ft and xi_t
def test_check_consistence_Ft_xi_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Ft': [np.ones((5, 4))]})
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt[0], Xt[0])
    expected_result = 'Ft and xi_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Ht and xi_t
def test_ckeck_consistence_Ht_xi_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Ht': [np.ones((5, 4))]})
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt[0], Xt[0])
    expected_result = 'Ht and xi_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Ht and y_t
def test_ckeck_consistence_Ht_y_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Ht': [np.ones((5, 4))]})
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt[0], Xt[0])
    expected_result = 'Ht and xi_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Bt and xi_t
def test_ckeck_consistence_Bt_xi_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Bt': [np.ones((5, 4))]})
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt[0], Xt[0])
    expected_result = 'Bt and xi_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Bt and x_t
def test_ckeck_consistence_Bt_x_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Bt': [np.ones((3, 4))]})
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt[0], Xt[0])
    expected_result = 'Bt and x_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Dt and y_t
def test_ckeck_consistence_Dt_y_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Dt': [np.ones((5, 2))]})
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt[0], Xt[0])
    expected_result = 'Dt and y_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Dt and x_t
def test_ckeck_consistence_Dt_x_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Dt': [np.ones((4, 4))]})
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt[0], Xt[0])
    expected_result = 'Dt and x_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Qt and xi_t
def test_ckeck_consistence_Qt_xi_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Qt': [np.ones((5, 5))]})
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt[0], Xt[0])
    expected_result = 'Qt and xi_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test consistence Rt and y_t
def test_ckeck_consistence_Rt_y_t(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Mt_.update({'Rt': [np.ones((3, 3))]})
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt[0], Xt[0])
    expected_result = 'Rt and y_t do not match in dimensions'
    result = str(error.value)
    assert result == expected_result


# Test if y_t is a vector
def test_ckeck_consistence_y_vector(Mt, Yt, Xt):
    Mt_ = deepcopy(Mt)
    Yt_ = np.ones([4,2])
    with pytest.raises(ValueError) as error:
        check_consistence(Mt_, Yt_, Xt[0])
    expected_result = 'y_t must be a vector'
    result = str(error.value)
    assert result == expected_result
