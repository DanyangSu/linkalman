import pytest
import numpy as np
from scipy import linalg
from linkalman.core.utils import mask_nan, inv, M_wrap
from linkalman.models import Constant_M as CM

# Test mask_nan
def test_mask_nan():
    """
    Test when input is a matrix with column size > 1
    """
    mat = np.ones((4,4))
    is_nan = np.array([True, False, True, False])
    expected_result = np.array([[0, 0, 0, 0], 
                            [0, 1, 0, 1], 
                            [0, 0, 0, 0], 
                            [0, 1, 0, 1]])
    result = mask_nan(is_nan, mat)
    np.testing.assert_array_equal(expected_result, result)

def test_mask_nan_vector():
    """
    Test when input is a matrix with column size == 1
    """
    mat = np.ones((4, 1))
    is_nan = np.array([True, False, True, False])
    expected_result = np.array([[0], [1], [0], [1]])
    result = mask_nan(is_nan, mat)
    np.testing.assert_array_equal(expected_result, result)

def test_mask_nan_wrong_dim_input():
    """
    Test if raise exception if wrong dim input 
    """
    mat = np.ones((4, 1))
    is_nan = np.array([True, False, True, False])
    with pytest.raises(ValueError):
        result = mask_nan(is_nan, mat, 'Col')

def test_mask_nan_row_only():
    """
    Test if only change row if dim=='row'
    """
    mat = np.ones((4,4))
    is_nan = np.array([True, False, True, False])
    expected_result = np.array([[0, 0, 0, 0], 
                            [1, 1, 1, 1], 
                            [0, 0, 0, 0], 
                            [1, 1, 1, 1]])
    result = mask_nan(is_nan, mat, dim='row')
    np.testing.assert_array_equal(expected_result, result)

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


    
