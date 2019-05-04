import pytest
import pandas as pd
import numpy as np
from linkalman.core.utils import Constant_M as CM, M_wrap
from copy import deepcopy


# Test getitem
def test_getitem():
    """
    Test if fetch correct item
    """
    M = np.array([[5, 3], [3, 4]])
    T = 10
    Mt = CM(M, T)
    Mt_wrap = M_wrap(Mt)
    expected_result = Mt[0]
    result = Mt_wrap[0] 
    np.testing.assert_array_equal(expected_result, result)


# Test setitem for wrapped class
def test_wrapped_class():
    """
    Test whether correctly fetch the wrapped object,
    and modify its values.
    """
    M = np.array([[5, 3], [3, 4]])
    T = 10
    Mt = CM(M, T)
    Mt_wrap = M_wrap(Mt)
    Mt_wrap[0] = np.array([[2, 4], [3, 6]])
    expected_result = np.array([[2, 4], [3, 6]])
    result = Mt_wrap[0]
    np.testing.assert_array_equal(expected_result, result)


def test_wrapped_class_other_index():
    """
    Test whether update in one index affect
    other index
    """
    M = np.array([[5, 3], [3, 4]])
    T = 10
    Mt = CM(M, T)
    Mt_wrap = M_wrap(Mt)
    Mt_wrap[0] = np.array([[2, 4], [3, 6]])
    expected_result = np.array([[5, 3], [3, 4]])
    result = Mt_wrap[1]
    np.testing.assert_array_equal(expected_result, result)


def test_wrapped_class_partial_update():
    """
    Test whether correctly fetch the wrapped object,
    and modify its values.
    """
    M = np.array([[5, 3], [3, 4]])
    T = 10
    Mt = CM(M, T)
    Mt_wrap = M_wrap(Mt)
    Mt_wrap[0][1, :] = 0
    expected_result = np.array([[5, 3], [0, 0]])
    result = Mt_wrap[0]
    np.testing.assert_array_equal(expected_result, result)


def test_wrapped_class_partial_update_other_index():
    """
    Test whether partially updating array affect other arrays
    """
    M = np.array([[5, 3], [3, 4]])
    T = 10
    Mt = CM(M, T)
    Mt_wrap = M_wrap(Mt)
    Mt_wrap[0][1, :] = 0
    expected_result = np.array([[5, 3], [3, 4]])
    result = Mt_wrap[1]
    np.testing.assert_array_equal(expected_result, result)


# Test pdet
def test_pdet():
    """
    Test pdet
    """
    M = np.array([[5, 3], [3, 4]])
    T = 10
    Mt = CM(M, T)
    Mt_wrap = M_wrap(Mt)
    expected_result = 11.0
    result = Mt_wrap.pdet(1)
    np.testing.assert_array_almost_equal(expected_result, result)
    

def test_pdet_not_full_rank():
    """
    Test pdet if not full rank
    """
    M = np.array([[5, 0], [0, 0]])
    T = 10
    Mt = CM(M, T)
    Mt_wrap = M_wrap(Mt)
    expected_result = 5
    result = Mt_wrap.pdet(1)
    np.testing.assert_array_almost_equal(expected_result, result)


def test_pdet_0():
    """
    Test pdet if 0
    """
    M = np.array([[0, 0], [0, 0]])
    T = 10
    Mt = CM(M, T)
    Mt_wrap = M_wrap(Mt)
    expected_result = 1 
    result = Mt_wrap.pdet(1)
    np.testing.assert_array_almost_equal(expected_result, result)

