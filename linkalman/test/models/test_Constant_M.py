import pytest
import pandas as pd
import numpy as np
from linkalman.models import Constant_M

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
    assert(np.array_equal(Mt[2], M))


