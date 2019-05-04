import pytest
import pandas as pd
import numpy as np
from linkalman.core.utils import Constant_M
from copy import deepcopy


# Test __init__
def test_wrong_M_type():
    """
    Test if raise exception if M is not a numpy array
    """
    M = 2.1
    T = 5
    with pytest.raises(TypeError):
        Mt = Constant_M(M, T)


# Test __getitem__
def test_getitem():
    """
    Test getitem
    """
    M = np.array([2, 3])
    T = 5
    Mt = Constant_M(M, T)
    assert(np.array_equal(Mt[0], M))


# Test __setitem__
def test_setitem():
    """
    Test setitem
    """
    M = np.array([2, 3])
    T = 5
    Mt = Constant_M(M, T)
    M_modify = np.array([4, 5])
    Mt[1] = M_modify
    assert(np.array_equal(Mt[1], M_modify))


def test_setitem_other_index():
    """
    Test if setitem affect other index
    """
    M = np.array([2, 3])
    T = 5
    Mt = Constant_M(M, T)
    M_modify = np.array([4, 5])
    Mt[1] = M_modify
    assert(np.array_equal(Mt[2], np.array([2, 3])))


def test_setitem_partial_update():
    """
    Test whether correctly modify indexed array
    """
    M = np.array([[5, 3], [3, 4]])
    T = 10
    Mt = Constant_M(M, T)
    Mt[0][1, :] = 0
    expected_result = np.array([[5, 3], [0, 0]])
    result = Mt[0]
    np.testing.assert_array_equal(expected_result, result)


def test_setitem_partial_update_other_index():
    """
    Test whether partially updating array affect other arrays
    """
    M = np.array([[5, 3], [3, 4]])
    T = 10
    Mt = Constant_M(M, T)
    Mt[0][1, :] = 0
    expected_result = np.array([[5, 3], [3, 4]])
    result = Mt[1]
    np.testing.assert_array_equal(expected_result, result)


def test_setitem_comprehensive():
    """
    Test multiple operations involving setitem
    """
    M = np.array([[5, 3], [3, 4]])
    T = 10
    Mt = Constant_M(M, T)
    expected_result_1 = np.array([[5, 2], [1, 5]])  # full update
    expected_result_2 = np.array([[5, 3], [0, 0]])  # partial update
    expected_result_default = np.array([[5, 3], [3, 4]])
    expected_result = [expected_result_2, expected_result_default, 
            expected_result_default, expected_result_1,
            expected_result_1, expected_result_default,
            expected_result_default, expected_result_default,
            expected_result_1]

    # Partial Update
    Mt[0][1, :] = 0

    # Full update 
    Mt[3] = deepcopy(expected_result_1)

    # Update twice
    Mt[4] = deepcopy(expected_result_2)
    Mt[4] = deepcopy(expected_result_1)

    # Update twice, second time same as M
    Mt[5] = deepcopy(expected_result_1)
    Mt[5] = deepcopy(expected_result_default)

    # Partially update twice, second time same as M
    Mt[6][1, :] = 0
    Mt[6][1, :] = np.array([3, 4])

    # Partially update then full update, second time same as M
    Mt[7][1, :] = 0
    Mt[7] = deepcopy(expected_result_default)

    # Partially update then full update
    Mt[8][1, :] = 0
    Mt[8] = deepcopy(expected_result_1)
    match = True
    for i in range(9):
        match = match and np.array_equal(Mt[i], expected_result[i])
    assert(match)
