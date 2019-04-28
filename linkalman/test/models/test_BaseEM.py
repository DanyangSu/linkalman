import pytest
from linkalman.models import BaseEM
import pandas as pd
import numpy as np

# Test _df_to_list
def test_df_to_list():
    """
    Test normal behaviors
    """
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 3., 4.]})
    expected_result = [np.array([[1.], [2.]]), np.array([[2.], [3.]]), np.array([[3.], [4.]])]
    result = BaseEM._df_to_list(df)
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
    result = BaseEM._df_to_list(df)
    
    for i in range(len(expected_result)):
        np.testing.assert_array_equal(expected_result[i], result[i])

def test_df_to_list_all_NaN():
    """
    Test 2 fully missing observations 
    """
    df = pd.DataFrame({'a': [1., np.nan, 3.], 'b': [2., np.nan, 4.]})
    expected_result = [np.array([[1.], [2.]]), np.array([[np.nan], [np.nan]]), np.array([[3.], [4.]])]
    result = BaseEM._df_to_list(df)

    for i in range(len(expected_result)):
        np.testing.assert_array_equal(expected_result[i], result[i])

def test_df_to_list_string():
    """
    Test str input exceptions
    """
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': [1, 'str2', 'str3']})

    with pytest.raises(TypeError):
        BaseEM._df_to_list(df)

def test_df_to_list_not_df():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    with pytest.raises(TypeError):
        BaseEM._df_to_list(df['a'])
    
# Test _list_to_df
def test_list_to_df():
    """
    Test normal behaviors
    """
    input_array = [np.array([[1.], [2.]]), np.array([[2.], [3.]]), np.array([[3.], [4.]])]
    col = ['a', 'b']
    expected_result = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 3., 4.]})
    result = BaseEM._list_to_df(input_array, col)
    assert(expected_result.equals(result))

def test_list_to_df_NaN():
    """
    Test partially missing observations
    """
    input_array = [np.array([[1.], [2.]]), np.array([[2.], [np.nan]]), np.array([[3.], [4.]])]
    col = ['a', 'b']
    expected_result = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., np.nan, 4.]})
    result = BaseEM._list_to_df(input_array, col)
    assert(expected_result.equals(result))

def test_list_to_df_all_NaN():
    """
    Test 2 fully missing observations
    """
    input_array = [np.array([[1.], [2.]]), np.array([[np.nan], [np.nan]]), np.array([[3.], [4.]])]
    col = ['a', 'b']
    expected_result = pd.DataFrame({'a': [1., np.nan, 3.], 'b': [2., np.nan, 4.]})
    result = BaseEM._list_to_df(input_array, col)
    assert(expected_result.equals(result))

def test_list_to_df_col_not_list():
    """
    Test if raise exception if col is not a list
    """
    input_array = [np.array([[1.]]), np.array([[np.nan]]), np.array([[3]])]
    col = 'string'
    with pytest.raises(TypeError):
        BaseEM._list_to_df(input_array, col)

# Test __init__
def test_init_not_function():
    """
    Test if raise exception when Ft is not callable
    """
    F = 2
    with pytest.raises(TypeError):
        A = BaseEM(F)
