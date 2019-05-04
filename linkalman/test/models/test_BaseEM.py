import pytest
from linkalman.models import BaseEM
import pandas as pd
import numpy as np


# Test __init__
def test_init_not_function():
    """
    Test if raise exception when Ft is not callable
    """
    F = 2
    with pytest.raises(TypeError):
        A = BaseEM(F)

