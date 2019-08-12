import pytest
import numpy as np
from scipy import linalg
import pandas as pd
from linkalman.core.utils import *


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


# Test ft
def test_ft():
    """
    Test regular behavior
    """
    def f(theta):
        array = np.array([[theta[0]]])
        F = B = H = D = Q = R = xi_1_0 = P_1_0 = array
        return {'F': F,
                'B': B,
                'H': H,
                'D': D,
                'Q': Q,
                'R': R}

    theta = [0.1]
    T = 2
    Mt = ft(theta, f, T)
    assert(np.array_equal(Mt['Bt'][1], np.array([[theta[0]]])))
   

def test_ft_missing_keys():
    """
    Test if raise exception if f(theta) is missing some keys
    """
    f = lambda theta: {'F': np.array([[theta[0], theta[1]]])}
    theta = [2, 3]
    T = 2

    with pytest.raises(ValueError):
        Mt = ft(theta, f, T)


def test_ft_auto_complete():
    """
    Test if autocomplete if missing B or D
    """
    def f(theta):
        array = np.array([[theta[0]]])
        F = B = H = D = Q = R = array
        return {'F': F,
                'H': H,
                'Q': Q,
                'R': R}

    theta = [2]
    T = 2
    Mt = ft(theta, f, T)
    np.testing.assert_equal(Mt['Dt'][0], np.zeros((1, 1)))


# Test gen_PSD
def test_gen_PSD():
    """
    Test normal behavior 
    """
    theta = [0,2,0,4,5,0]
    dim = 3
    expected_results = np.array([[1, 2, 4],
                                [2, 5, 13],
                                [4, 13, 42]])
    results = gen_PSD(theta, dim)
    assert(np.array_equal(expected_results, results))


def test_gen_PSD_wrong_theta_size():
    """
    Test if raise exception when theta wrong size
    """
    theta = [1,2,3,4,5,6]
    dim = 2
    with pytest.raises(ValueError):
        PSD = gen_PSD(theta, dim)

    
# Test df_to_list
def test_df_to_list():
    """
    Test normal behaviors
    """
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 3., 4.]})
    expected_result = [np.array([[1.], [2.]]), np.array([[2.], [3.]]), np.array([[3.], [4.]])]
    result = df_to_list(df)
    outcome = True

    for i in range(len(expected_result)):
        outcome = outcome and np.array_equal(expected_result[i], result[i])
    assert(outcome)


def test_df_to_list_NaN():
    """
    Test partially missing observations
    """
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., np.nan, 4.]})
    expected_result = [np.array([[1.], [2.]]), np.array([[2.], [np.nan]]), np.array([[3.], [4.]])]
    result = df_to_list(df)
    
    for i in range(len(expected_result)):
        np.testing.assert_array_equal(expected_result[i], result[i])


def test_df_to_list_all_NaN():
    """
    Test 2 fully missing observations 
    """
    df = pd.DataFrame({'a': [1., np.nan, 3.], 'b': [2., np.nan, 4.]})
    expected_result = [np.array([[1.], [2.]]), np.array([[np.nan], [np.nan]]), np.array([[3.], [4.]])]
    result = df_to_list(df)

    for i in range(len(expected_result)):
        np.testing.assert_array_equal(expected_result[i], result[i])


def test_df_to_list_string():
    """
    Test str input exceptions
    """
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': [1, 'str2', 'str3']})

    with pytest.raises(TypeError):
        df_to_list(df)


def test_df_to_list_not_df():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    with pytest.raises(TypeError):
        df_to_list(df['a'])
    

# Test list_to_df
def test_list_to_df():
    """
    Test normal behaviors
    """
    input_array = [np.array([[1.], [2.]]), np.array([[2.], [3.]]), np.array([[3.], [4.]])]
    col = ['a', 'b']
    expected_result = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 3., 4.]})
    result = list_to_df(input_array, col)
    assert(expected_result.equals(result))


def test_list_to_df_NaN():
    """
    Test partially missing observations
    """
    input_array = [np.array([[1.], [2.]]), np.array([[2.], [np.nan]]), np.array([[3.], [4.]])]
    col = ['a', 'b']
    expected_result = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., np.nan, 4.]})
    result = list_to_df(input_array, col)
    assert(expected_result.equals(result))


def test_list_to_df_all_NaN():
    """
    Test 2 fully missing observations
    """
    input_array = [np.array([[1.], [2.]]), np.array([[np.nan], [np.nan]]), np.array([[3.], [4.]])]
    col = ['a', 'b']
    expected_result = pd.DataFrame({'a': [1., np.nan, 3.], 'b': [2., np.nan, 4.]})
    result = list_to_df(input_array, col)
    assert(expected_result.equals(result))


def test_list_to_df_col_not_list():
    """
    Test if raise exception if col is not a list
    """
    input_array = [np.array([[1.]]), np.array([[np.nan]]), np.array([[3]])]
    col = 'string'
    with pytest.raises(TypeError):
        list_to_df(input_array, col)


# Test creat_col
def test_create_col():
    """
    Test append column suffix
    """
    col = ['a', 'b']
    suffix = '_test'
    expected_result = ['a_test', 'b_test']
    result = create_col(col, suffix=suffix)
    assert(expected_result == result)


def test_create_col_default():
    """
    Test default suffix
    """
    col = ['a', 'b']
    expected_result = ['a_pred', 'b_pred']
    result = create_col(col)
    assert(expected_result == result)


# Test noise
def test_noise_1d():
    """
    test 1d noise
    """
    epsilon = noise(1, np.array([[1]]))
    result = epsilon.shape
    expected_result = (1, 1)
    assert(result == expected_result)


def test_noise_2d():
    """
    test 2d noise
    """
    epsilon = noise(2, np.array([[3, 2], [2, 4]]))
    result = epsilon.shape
    expected_result = (2, 1)
    assert(result == expected_result)


# Test simulated_data
def test_simulated_data_type_error():
    """
    Test if raise exception when both Xt and T are None
    """
    Mt = {'Ft': [np.array([[1.0]])],
            'Ht': [np.array([[1.0]])],
            'Qt': [np.array([[1.0]])],
            'Bt': [np.array([[1.0]])],
            'Dt': [np.array([[1.0]])]}
    with pytest.raises(ValueError):
        df = simulated_data(Mt)


# Test clean_matrix
def test_clean_matrix():
    """
    Test whether it detect small and large values
    """
    M = np.array([1e-10, np.inf])
    M_clean = clean_matrix(M)
    result = M_clean[0] * M_clean[1]
    expected_result = 0
    assert(result == expected_result)


# Test get_ergodic
def test_get_ergodic_unit_roots():
    """
    Test whether return diffuse prior if unit roots
    """
    F = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0.8]])
    Q = 0.36 * np.eye(3)
    result = get_ergodic(F, Q)
    expected_result = np.array([[np.nan, 0, 0],
        [0, np.nan, 0], [0, 0, 1]])
    np.testing.assert_array_almost_equal(expected_result, result)
