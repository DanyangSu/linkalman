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

    
# Test df_to_tensor
def test_df_to_tensor(df1):
    """
    Test normal behaviors
    """
    expected_result = np.array([[[1.], [4.]], [[2.], [5.]], [[3.], [6.]]])
    col_list = ['a', 'b']
    result = df_to_tensor(df1, col_list)
    
    np.testing.assert_array_equal(expected_result, result)


def test_df_to_tensor_nan(df1):
    """
    Test if return None when not specify col_list
    """
    expected_result = None
    result = df_to_tensor(df1)

    assert result == expected_result


def test_df_to_tensor_NaN():
    """
    Test partially missing observations
    """
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., np.nan, 4.], 'c': [1, 2, 3]})
    expected_result = np.array([[[1.], [2.]], [[2.], [np.nan]], [[3.], [4.]]])
    col_list = ['c', 'b']
    result = df_to_tensor(df, col_list)
    
    np.testing.assert_array_equal(expected_result, result)


def test_df_to_tensor_all_NaN():
    """
    Test 2 fully missing observations 
    """
    df = pd.DataFrame({'a': [1., np.nan, 3.], 'b': [2., np.nan, 4.]})
    col_list = ['a', 'b']
    expected_result = np.array([[[1.], [2.]], [[np.nan], [np.nan]], [[3.], [4.]]])
    result = df_to_tensor(df, col_list)

    np.testing.assert_array_equal(expected_result, result)


def test_df_to_tensor_string():
    """
    Test str input exceptions
    """
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': [1, 'str2', 'str3']})
    col_list = ['a', 'b']
    with pytest.raises(TypeError):
        df_to_tensor(df, col_list)


def test_df_to_list_not_df(df1):
    with pytest.raises(TypeError):
        df_to_tensor(df1['a'], ['a'])
    

# Test tensor_to_df
def test_tensor_to_df():
    """
    Test normal behaviors
    """
    input_array = np.array([[[1.], [2.]], [[2.], [3.]], [[3.], [4.]]])
    col = ['a', 'b']
    expected_result = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 3., 4.]})
    result = tensor_to_df(input_array, col)
    assert(expected_result.equals(result))


def test_tensor_to_df_NaN():
    """
    Test partially missing observations
    """
    input_array = np.array([[[1.], [2.]], [[2.], [np.nan]], [[3.], [4.]]])
    col = ['a', 'b']
    expected_result = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., np.nan, 4.]})
    result = tensor_to_df(input_array, col)
    assert(expected_result.equals(result))


def test_tensor_to_df_all_NaN():
    """
    Test 2 fully missing observations
    """
    input_array = np.array([[[1.], [2.]], [[np.nan], [np.nan]], [[3.], [4.]]])
    col = ['a', 'b']
    expected_result = pd.DataFrame({'a': [1., np.nan, 3.], 'b': [2., np.nan, 4.]})
    result = tensor_to_df(input_array, col)
    assert(expected_result.equals(result))


def test_list_to_df_col_not_list():
    """
    Test if raise exception if col is not a list
    """
    input_array = np.array([[[1.]], [[np.nan]], [[3]]])
    col = 'string'
    with pytest.raises(TypeError):
        tensor_to_df(input_array, col)


# Test get_reset
def test_get_reset():
    """
    Test if generate correct reset array
    """
    tensor = build_tensor(np.array([[1, 2], [3, 4]]), 10)
    tensor[1][1][1] = 0
    tensor[2][1][1] = 0
    tensor[6][0][0] = 8

    expected_result = np.zeros(10)
    expected_result[0] = 1
    expected_result[1] = 1
    expected_result[3] = 1
    expected_result[6] = 1
    expected_result[7] = 1

    result = get_reset(tensor)
    np.testing.assert_array_equal(expected_result, result)


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
    Mt = ft(theta, f, T, None)
    assert(np.array_equal(Mt['Bt'][1], np.array([[theta[0]]])))
   

def test_ft_missing_keys():
    """
    Test if raise exception if f(theta) is missing some keys
    """
    f = lambda theta: {'F': np.array([[theta[0], theta[1]]])}
    theta = [2, 3]
    T = 2

    with pytest.raises(ValueError):
        Mt = ft(theta, f, T, None)


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
    Mt = ft(theta, f, T, np.array([[1]]))
    np.testing.assert_equal(Mt['Dt'][0], np.zeros((1, 1)))


def test_ft_Q_symmetric():
    """
    Test if Q is not symmetric
    """
    def f(theta):
        array = np.array([[theta[0]]])
        F = B = H = D = R = array
        Q = np.array([[1, 2],[3, 4]])
        return {'F': F,
                'H': H,
                'Q': Q,
                'R': R}

    theta = [2]
    T = 2

    with pytest.raises(ValueError) as error:
        Mt = ft(theta, f, T)
    expected_result = 'Q is not symmetric'
    result = str(error.value)
    assert result == expected_result


def test_ft_Q_symmetric():
    """
    Test if Q is not symmetric
    """
    def f(theta):
        array = np.array([[theta[0]]])
        F = B = H = D = R = array
        Q = np.array([[1, 2],[2, 1]])
        return {'F': F,
                'H': H,
                'Q': Q,
                'R': R}

    theta = [2]
    T = 2

    with pytest.raises(ValueError) as error:
        Mt = ft(theta, f, T, None)
    expected_result = 'Q is not semi-PSD'
    result = str(error.value)
    assert result == expected_result


# Test build_tensor
def test_build_tensor():
    """
    Test normal behavior of build_tensor
    """
    arr = np.array([[1, 2], [3, 4]])
    tensor = build_tensor(arr, 10)
    assert tensor.shape == (10, 2, 2)
    np.testing.assert_array_equal(arr, tensor[5])


def test_build_tensor_1d():
    arr = np.array([1])
    with pytest.raises(TypeError) as error:
        build_tensor(arr, 5)
    expected_result = 'Input must be 2-d arrays'
    result = str(error.value)
    assert result == expected_result


# Test simulated_data
def test_simulated_data_type_error():
    """
    Test if raise exception when both Xt and T are None
    """
    def ft(theta, T):
        Mt = {'Ft': [np.array([[theta]])],
                'Ht': [np.array([[theta]])],
                'Qt': [np.array([[theta]])],
                'Bt': [np.array([[theta]])],
                'Rt': [np.array([[theta]])],
                'Dt': [np.array([[theta]])],
                'xi_1_0': np.array([[theta]]),
                'P_1_0': np.array([[theta]])}
        return Mt

    theta = 1.0

    with pytest.raises(ValueError):
        df = simulated_data(ft, theta)


def test_simulated_data_init_state_wrong_dim():
    """
    Test if init_value is properly checked
    """
    def ft(theta, T):
        Mt = {'Ft': [np.array([[theta]])],
                'Ht': [np.array([[theta]])],
                'Qt': [np.array([[theta]])],
                'Bt': [np.array([[theta]])],
                'Rt': [np.array([[theta]])],
                'Dt': [np.array([[theta]])],
                'xi_1_0': np.array([[theta]]),
                'P_1_0': np.array([[theta]])}
        return Mt

    theta = 1.0
    T = 4
    # init_state = {'xi_t': np.zeros([3, 1]),
    #         'P_star_t': np.zeros([3, 2])}
    init_state = {'xi_t': np.zeros([1, 1]),
            'P_star_t': np.zeros([3, 2])}
    with pytest.raises(ValueError) as error:
        df = simulated_data(ft, theta, T=T, init_state=init_state)
    result = str(error.value)
    expected_result = 'User-specified P_star_t has wrong dimensions'
    assert result == expected_result


def test_simulated_data_init_state_value():
    """
    Test if init_value overrides
    """
    def ft(theta, T):
        Mt = {'Ft': [np.array([[theta]])],
                'Ht': [np.array([[theta]])],
                'Qt': [np.array([[theta]])],
                'Bt': [np.array([[theta]])],
                'Rt': [np.array([[theta]])],
                'Dt': [np.array([[theta]])],
                'xi_1_0': np.array([[theta]]),
                'P_1_0': np.array([[theta]])}
        return Mt

    theta = 1.0
    T = 1
    init_state = {'xi_t': 100 * np.ones([1, 1]),
            'P_star_t': np.zeros([1, 1])}
    df, _, _ = simulated_data(ft, theta, T=T, init_state=init_state)
    result = np.array([df.loc[0, ['xi_0']]]).T
    expected_result = 100 * np.ones([1, 1])
    np.testing.assert_array_equal(result, expected_result)


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
def test_get_ergodic_unit_roots_var():
    """
    Test whether return diffuse prior if unit roots
    """
    F = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0.8]])
    Q = 0.36 * np.eye(3)
    result, _ = get_ergodic(F, Q)
    expected_result = np.array([[np.nan, 0, 0],
        [0, np.nan, 0], [0, 0, 1]])
    np.testing.assert_array_almost_equal(expected_result, result)


def test_get_ergodic_unit_roots_mean():
    """
    Test whether return diffuse prior mean if unit roots
    """
    F = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0.8]])
    Q = 0.36 * np.eye(3)
    _, result = get_ergodic(F, Q, B=np.ones([3, 1]), x_0=np.array([[0.3]]))
    expected_result = np.array([[0, 0, 1.5]]).T
    np.testing.assert_array_almost_equal(expected_result, result)


def test_get_ergodic_unit_roots_mean_ar2():
    """
    Test whether return diffuse prior mean if unit roots and ar2
    """
    F = np.array([[0.5, 0.5, 0, 0], 
                  [1, 0, 0, 0], 
                  [0, 0, 0.3, 0.2],
                  [0, 0, 1, 0]])
    Q = 0.36 * np.eye(4)
    Q[3][3] = 0
    _, result = get_ergodic(F, Q, B=np.ones([4, 1]), x_0=np.array([[0.3]]))
    F = np.array([[0.3, 0.2], [1, 0]])
    ergodic_mean = linalg.pinv(np.eye(2) - F).dot(np.array([[0.3, 0.3]]).T)
    expected_result = np.zeros([4, 1])
    expected_result[2] = ergodic_mean[0]
    expected_result[3] = ergodic_mean[1]
    np.testing.assert_array_almost_equal(expected_result, result)


def test_get_ergodic_unit_roots_var_ar2_Q33_0():
    """
    Test whether return diffuse prior mean if unit roots and ar2
    In addition, I set Q[3][3] = 0. I test the result by evaluating 
    the lyapunov equation
    """
    F = np.array([[0.5, 0.5, 0, 0], 
                  [1, 0, 0, 0], 
                  [0, 0, 0.3, 0.2],
                  [0, 0, 1, 0]])
    Q = 0.36 * np.eye(4)
    Q[3][3] = 0
    cov_, _ = get_ergodic(F, Q, B=np.ones([4, 1]), x_0=np.array([[0.3]]))
    ergo_var = cov_[2:4, 2:4]
    ergo_F = F[2:4, 2:4]
    ergo_Q = Q[2:4, 2:4]
    result = ergo_F.dot(ergo_var).dot(ergo_F.T) + ergo_Q
    expected_result = ergo_var
    np.testing.assert_array_almost_equal(expected_result, result)


def test_get_ergodic_unit_roots_var_ar2_Q33_0_diffuse():
    """
    Test whether return diffuse prior mean if unit roots and ar2
    In addition, I set Q[3][3] = 0. Here I check whether it returns 
    the correct diffuse part.
    """
    F = np.array([[0.5, 0.5, 0.1, 0], 
                  [1, 0, 0, 0], 
                  [0, 0, 0.3, 0.2],
                  [0, 0, 1, 0]])
    Q = 0.36 * np.eye(4)
    Q[3][3] = 0
    cov_, _ = get_ergodic(F, Q, B=np.ones([4, 1]), x_0=np.array([[0.3]]))
    diffuse_var = cov_[0:2, 0:2]
    result = diffuse_var
    expected_result = np.array([[np.nan, 0], [0, np.nan]])
    np.testing.assert_array_almost_equal(expected_result, result)


def test_get_ergodic_explosive_roots_mean():
    """
    Test cases where we have explosive roots
    """
    F = np.array([[0.8, 0.6, 0.2, 0], 
                  [1, 0, 0, 0], 
                  [0, 0, 0.3, 0.2],
                  [0, 0, 1, 0]])
    Q = 0.36 * np.eye(4)
    Q[1][1] = 0
    Q[3][3] = 0
    B = np.array([1, 0, 1, 0]).reshape(-1, 1)
    cov_, mean = get_ergodic(F, Q, B=B, x_0=np.array([[0.3]]))
    result = mean
    expected_result = np.array([0, 0, 0.6, 0.6]).reshape(-1, 1)
    np.testing.assert_array_almost_equal(expected_result, result)


def test_get_ergodic_explosive_roots_var():
    """
    Test cases where we have explosive roots
    """
    F = np.array([[0.8, 0.6, 0.2, 0], 
                  [1, 0, 0, 0], 
                  [0, 0, 0.3, 0.2],
                  [0, 0, 1, 0]])
    Q = 0.36 * np.eye(4)
    Q[1][1] = 0
    Q[3][3] = 0
    B = np.array([1, 0, 1, 0]).reshape(-1, 1)
    cov_, mean_ = get_ergodic(F, Q, B=B, x_0=np.array([[0.3]]))
    result = cov_
    vec_Q = np.array([Q[0][0], Q[1][0], Q[1][1]]).reshape(-1, 1)
    M = np.array([[0.91, -0.12, -0.04],
                  [-0.3, 0.8, 0],
                  [-1, 0, 1]])
    vec_var = linalg.pinv(M).dot(vec_Q)
    ergo_var = np.array([[vec_var[0][0], vec_var[1][0]],
                         [vec_var[1][0], vec_var[2][0]]])
    expected_result = np.zeros([4, 4])
    expected_result[0][0] = np.nan
    expected_result[1][1] = np.nan
    expected_result[2:4, 2:4] = ergo_var
    expected_mean = np.array([0, 0, 0.6, 0.6]).reshape(-1, 1)
    np.testing.assert_array_almost_equal(expected_result, result)
    np.testing.assert_array_almost_equal(expected_mean, mean_)


def test_get_ergodic_explosive_roots_var_force_diffuse():
    """
    Test cases where we have explosive roots
    """
    F = np.array([[0.8, 0.6, 0.2, 0], 
                  [1, 0, 0, 0], 
                  [0, 0, 0.3, 0.2],
                  [0, 0, 1, 0]])
    Q = 0.36 * np.eye(4)
    Q[1][1] = 0
    Q[3][3] = 0
    force_diffuse = [False, False, False, True]
    cov_, mean_ = get_ergodic(F, Q, B=np.ones([4, 1]), x_0=np.array([[0.3]]),
            force_diffuse=force_diffuse)
    result = cov_
    vec_Q = np.array([Q[0][0], Q[1][0], Q[1][1]]).reshape(-1, 1)
    M = np.array([[0.91, -0.12, -0.04],
                  [-0.3, 0.8, 0],
                  [-1, 0, 1]])
    expected_result = np.diag([np.nan] * 4)
    expected_mean = np.zeros([4, 1])
    np.testing.assert_array_almost_equal(expected_result, result)
    np.testing.assert_array_almost_equal(expected_mean, mean_)


# Test get_init_mat
def test_get_init_mat():
    """
    Test normal run
    """
    P_1_0 = np.eye(4)
    P_1_0[1][1] = np.nan
    P_1_0[3][3] = np.nan

    num_diff, A, Pi, P_star = get_init_mat(P_1_0)
    expected_num_diff = 2
    expected_A = np.eye(4)
    expected_A[0][0] = 0
    expected_A[2][2] = 0
    expected_Pi = np.eye(4) - expected_A
    expected_P_star = expected_Pi
    np.testing.assert_array_almost_equal(expected_A, A)
    np.testing.assert_array_almost_equal(expected_Pi, Pi)
    np.testing.assert_array_almost_equal(expected_P_star, P_star)
    assert num_diff == expected_num_diff


# Test clean_matrix
def test_clean_matrix():
    """
    Test normal run
    """
    M = np.array([1e-12, 1e14])
    expected_result = np.array([0, inf_val])
    result = clean_matrix(M)
    np.testing.assert_array_almost_equal(expected_result, result)


# Test permute
def test_permute_row(perm_vector):
    """
    Test row permutation
    """
    new_index = np.array([1, 3, 0, 2])
    expected_result = np.array([[2], [4], [1], [3]])
    result = permute(perm_vector, new_index)
    np.testing.assert_array_equal(expected_result, result)


def test_permute_col(perm_mat):
    """
    Test column permutation
    """
    new_index = np.array([1, 3, 0, 2])
    expected_result = np.array([[2, 4, 1, 3],
                                [5, 7, 2, 6],
                                [6, 9, 3, 8],
                                [7, 0, 4, 9]])
    result = permute(perm_mat, new_index, axis='col')
    np.testing.assert_array_equal(expected_result, result)


def test_permute_both(perm_mat):
    """
    Test permutation for row and column
    """
    new_index = np.array([1, 3, 0, 2])
    expected_result = np.array([[5, 7, 2, 6],
                                [7, 0, 4, 9], 
                                [2, 4, 1, 3],
                                [6, 9, 3, 8],])
    result = permute(perm_mat, new_index, axis='both')
    np.testing.assert_array_equal(expected_result, result)


# Test revert_permute
def revert_permute():
    """
    Test normal run
    """
    index = np.array([2, 3, 1, 0])
    expected_result = np.array([3, 2, 0, 1])
    result = revert_permute(index)
    np.testing.assert_array_equal(expected_result, result)


# Test partition_index
def test_partition_index():
    """
    Test normal run
    """
    is_missing = np.array([True, False, False, True, False, True, False])
    expected_result = np.array([1, 2, 4, 6, 0, 3, 5])
    result = partition_index(is_missing)
    np.testing.assert_array_equal(expected_result, result)


def test_partition_index_all_missing():
    """
    Test all missing
    """
    is_missing = np.array([True] * 7)
    expected_result = np.array(list(range(7)))
    result = partition_index(is_missing)
    np.testing.assert_array_equal(expected_result, result)


# Test gen_Xt
def test_gen_Xt_None():
    """
    Test if Xt is None
    """
    B = np.array([[1,2],[3,4]])
    T = 1
    result = gen_Xt(B=B, T=T)[0]
    expected_result = np.zeros([2,1])
    np.testing.assert_array_equal(expected_result, result)
    

def test_gen_Xt_no_BT():
    """
    Test if No Xt and no B or T
    """
    with pytest.raises(ValueError) as error:
        Xt = gen_Xt()
    expected_result = 'B and T must not be None'
    result = str(error.value)
    assert result == expected_result

    
# Test LL_correct
def test_LL_correct():
    """
    Test cases when index is sorted
    """
    Ht = [np.array([[1, 2], [3, 4]])] * 2
    Ft = [np.array([[3, 4], [5, 6]])] * 2
    n_t = [1, 2]
    A = np.eye(2)
    A[0][0] = 0 

    result = LL_correct(Ht, Ft, n_t, A)
    expected_result = np.array([[0, 0], [0, 1556]])
    np.testing.assert_array_equal(expected_result, result)


def test_LL_correct_not_sorted():
    """
    Test cases when index is not sorted
    """
    Ht = [np.array([[1, 2], [3, 4]])] * 2
    Ft = [np.array([[3, 4], [5, 6]])] * 2
    n_t = [1, 2]
    index = [[1, 0], [1, 0]]
    A = np.eye(2)

    result = LL_correct(Ht, Ft, n_t, A, index=index)
    expected_result = np.array([[1019, 1264], [1264, 1568]])
    np.testing.assert_array_equal(expected_result, result)


# Test preallocate
def test_preallocate_dim1():
    """
    Test for 1d list of None
    """
    dim1 = 3
    result = preallocate(dim1, 1, 1)
    expected_result = np.zeros((3, 1, 1))
    np.testing.assert_array_equal(result, expected_result)


# Test get_explosive_diffuse
def test_get_explosive_diffuse4():
    """
    Test F with four strongly connected components
    """

    test_F = np.array([[2, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0.1, 0.2, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0.1, 0, 2, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0.01, 0, 0, 0, 0, 0, 0.1, 0.2],
                        [0, 0, 0, 0, 0, 0, 1, 0]])
    result = get_explosive_diffuse(test_F)
    expected_result = np.array([True, True, False, False, True, True, False, False])
    np.testing.assert_array_equal(result, expected_result)


def test_get_explosive_diffuse3():
    """
    Test F with four strongly connected components
    """

    test_F = np.array([[2, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0.1, 0.2, 0, 0.01, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0.1, 0, 2, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0.01, 0, 0, 0, 0, 0, 0.1, 0.2],
                        [0, 0, 0, 0, 0, 0, 1, 0]])
    result = get_explosive_diffuse(test_F)
    expected_result = np.array([True, True, True, True, True, True, False, False])
    np.testing.assert_array_equal(result, expected_result)


# Test get_nearest_PSD
def test_get_nearest_PSD_PSD():
    """
    Test result when input is PSD
    """
    X = np.array([[3, 2], [2, 4]])
    X_PSD = get_nearest_PSD(X)
    
    np.testing.assert_array_almost_equal(X, X_PSD)


def test_get_nearest_PSD_not_PSD():
    """
    Test result when input is not PSD
    """
    X = np.array([[-2]])
    X_PSD = get_nearest_PSD(X)
    expected_X = np.zeros([1, 1])
    np.testing.assert_array_equal(expected_X, X_PSD)


# Test validate_wrapper
def test_validate_wrapper_string():
    """
    Test error message if input is some random object
    """
    class wrapper():
        def __init__(m_tensor, reset):
            pass
    with pytest.raises(AttributeError) as error:
        validate_wrapper(wrapper)
    expected_result = 'The wrapper does not contain all required methods.'
    assert expected_result == str(error.value)


def test_validate_wrapper_wrong_input():
    """
    Test error message if the wrapper object has required arguments
    """
    class wrapper():
        def __init__(self, m_tensor, rese):
            pass
    with pytest.raises(AttributeError) as error:
        validate_wrapper(wrapper)
    expected_result = """The wrapper object must contain 'm_tensor' and 'reset'. """
    assert expected_result == str(error.value)


def test_validate_wrapper_incomplete_obj():
    """
    Test error message if input does not have all required methods
    """
    class incomplete_M(object):
        def __init__(self, m_tensor, reset):
            pass

        def pinvh(self):
            pass

        def pdet(self):
            pass

        def Ldl(self):  # intentionally use capital L
            pass 

    with pytest.raises(AttributeError) as error:
        validate_wrapper(incomplete_M)
    expected_result = 'The wrapper does not contain all required methods.'
    assert expected_result == str(error.value)
