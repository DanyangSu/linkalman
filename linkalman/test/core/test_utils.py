import pytest
import numpy as np
from scipy import linalg
from linkalman.core.utils import inv, M_wrap
from linkalman.models import Constant_M as CM

# Test inv
def test_inv():
    """
    Test normal behavior of inv
    """
    mat = np.array([[2, 1], [1, 4]])
    expected_result =np.array([[1/2 + 1/14, -1/7], [-1/7, 2/7]])
    result = inv(mat)
    np.testing.assert_array_almost_equal(result, expected_result)
    
def test_inv_0():
    """
    Test pseudo-inverse of zero matrix
    """
    mat = np.zeros([3, 3])
    expected_result =np.zeros([3, 3])
    result = inv(mat)
    np.testing.assert_array_almost_equal(result, expected_result)

def test_inv_not_full_rank():
    """
    Test pseudo-inverse if not full rank
    """
    mat = np.array([[2, 0], [0, 0]])
    expected_result = np.array([[0.5, 0], [0, 0]])
    result = inv(mat)
    np.testing.assert_array_almost_equal(result, expected_result)

def test_inv_not_PSD():
    """
    Test result if input matrix not PSD
    """
    mat = np.array([[2, 4], [3, 1]])
    expected_result = linalg.pinv(mat)
    result = inv(mat)
    np.testing.assert_array_almost_equal(result, expected_result)

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

