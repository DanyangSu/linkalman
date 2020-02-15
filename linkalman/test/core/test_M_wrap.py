import pytest
import pandas as pd
import numpy as np
from linkalman.core.utils import M_wrap
from copy import deepcopy

# Test getitem
def test_getitem():
    """
    Test if fetch correct item
    """
    M = np.array([[5, 3], [3, 4]])
    T = 10
    reset = np.ones(T, dtype=bool)
    M = np.expand_dims(M,0)
    Mt = np.repeat(M, 5, 0)
    Mt_wrap = M_wrap(Mt, reset)
    expected_result = Mt[0]
    result = Mt_wrap[0] 
    np.testing.assert_array_equal(expected_result, result)


# Test setitem for wrapped class
def test_wrapped_class(Mt):
    """
    Test whether correctly fetch the wrapped object,
    and modify its values.
    """
    reset = np.ones(Mt['Ft'].shape[0], dtype=bool)
    Mt_wrap = M_wrap(Mt['Ft'], reset)
    Mt_wrap[0] = 2 * np.ones((3, 3))
    expected_result = 2 * np.ones((3, 3))
    result = Mt_wrap[0]
    np.testing.assert_array_equal(expected_result, result)


def test_wrapped_class_other_index(Mt):
    """
    Test whether update in one index affect
    other index
    """
    reset = np.ones(Mt['Ft'].shape[0], dtype=bool)
    Mt_wrap = M_wrap(Mt['Ft'], reset)
    Mt_wrap[0] = np.zeros((3, 3))
    expected_result = np.ones((3, 3))
    result = Mt_wrap[1]
    np.testing.assert_array_equal(expected_result, result)


def test_wrapped_class_partial_update(Mt):
    """
    Test whether correctly fetch the wrapped object,
    and modify its values.
    """
    reset = np.ones(Mt['Ft'].shape[0], dtype=bool)
    Mt_wrap = M_wrap(Mt['Ft'], reset)
    Mt_wrap[0][1, :] = 0
    expected_result = np.array([[1, 1, 1], [0, 0, 0], [1, 1 ,1]])
    result = Mt_wrap[0]
    np.testing.assert_array_equal(expected_result, result)


def test_wrapped_class_partial_update_other_index(Mt):
    """
    Test whether partially updating array affect other arrays
    """
    reset = np.ones(Mt['Ft'].shape[0], dtype=bool)
    Mt_wrap = M_wrap(Mt['Ft'], reset)
    Mt_wrap[0][1, :] = 0
    expected_result = np.ones((3, 3))
    result = Mt_wrap[1]
    np.testing.assert_array_equal(expected_result, result)


# Test pdet
def test_pdet(Mt):
    """
    Test pdet
    """
    reset = np.ones(Mt['Ft'].shape[0], dtype=bool)
    Mt_wrap = M_wrap(Mt['Ft'], reset)
    Mt_wrap[1][1][1] = 0
    Mt_wrap[1][2][2] = 3
    Mt_wrap.refresh()
    expected_result = -2.0
    result = Mt_wrap.pdet(1)
    np.testing.assert_array_almost_equal(expected_result, result)
    

def test_pdet_not_full_rank(Mt):
    """
    Test pdet if not full rank
    """
    reset = np.ones(Mt['Ft'].shape[0], dtype=bool)
    Mt_wrap = M_wrap(Mt['Ft'], reset)
    Mt_wrap[0][0][2] = 0
    Mt_wrap[0][2][0] = 0
    expected_result = -1
    result = Mt_wrap.pdet(0)
    np.testing.assert_array_almost_equal(expected_result, result)


def test_pdet_0(Mt):
    """
    Test pdet if 0
    """
    reset = np.ones(Mt['Ft'].shape[0], dtype=bool)
    Mt['Ft'][0] = np.zeros((3, 3))
    Mt_wrap = M_wrap(Mt['Ft'], reset)
    expected_result = 1 
    result = Mt_wrap.pdet(0)
    np.testing.assert_array_almost_equal(expected_result, result)


# Test refresh
def test_refresh_init(Mt):
    """
    Test if correctly refresh initial values
    """
    reset = np.ones(Mt['Ft'].shape[0], dtype=bool)
    Mt_wrap = M_wrap(Mt['Ft'], reset)
    Mt_wrap[0][0][2] = 0
    Mt_wrap[0][2][0] = 0
    expected_result = -1
    Mt_wrap.refresh()
    result = Mt_wrap.pdet(0)
    np.testing.assert_array_almost_equal(expected_result, result)

