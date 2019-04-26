import pytest
from linkalman.models import Base
import pandas as pd
import numpy as np

# Test _df_to_list
def test_df_to_list():
    """
    Test normal behaviors
    """
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 3., 4.]})
    expected_result = [np.array([[1.], [2.]]), np.array([[2.], [3.]]), np.array([[3.], [4.]])]
    result = Base._df_to_list(df)
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
    result = Base._df_to_list(df)
    
    for i in range(len(expected_result)):
        np.testing.assert_array_equal(expected_result[i], result[i])

def test_df_to_list_same_two_NaN():
    """
    Test 2 fully missing observations 
    """
    df = pd.DataFrame({'a': [1., np.nan, 3.], 'b': [2., np.nan, 4.]})
    expected_result = [np.array([[1.], [2.]]), np.array([[np.nan], [np.nan]]), np.array([[3.], [4.]])]
    result = Base._df_to_list(df)

    for i in range(len(expected_result)):
        np.testing.assert_array_equal(expected_result[i], result[i])

def test_df_to_list_string():
    """
    Test str input exceptions
    """
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': [1, 'str2', 'str3']})

    with pytest.raises(TypeError):
        Base._df_to_list(df)
    

# Test _list_to_df
