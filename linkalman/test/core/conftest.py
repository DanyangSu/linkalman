import pytest
import numpy as np


# Generate input data
@pytest.fixture()
def Mt():
    Mt = {'Ft': [np.ones((3, 3))],
            'Bt': [np.ones((3, 2))],
            'Ht': [np.ones((4, 3))],
            'Dt': [np.ones((4, 2))],
            'Qt': [np.ones((3, 3))],
            'Rt': [np.ones((4, 4))],
            'xi_1_0': np.ones((3, 1)),
            'P_1_0': np.ones((3, 3))}
    return Mt


@pytest.fixture()
def Yt():
    Yt = [np.ones((4, 1))]
    return Yt


@pytest.fixture()
def Xt():
    Xt = [np.ones((2, 1))]
    return Xt

