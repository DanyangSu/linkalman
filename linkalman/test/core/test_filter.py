# import pytest
# import numpy as np
# from scipy import linalg
# import pandas as pd
# from linkalman.core import Filter
# from copy import deepcopy


# # Test autocomplete x
# def test_gen_xt(Mt, Yt, Xt):
#     Mt_ = deepcopy(Mt)
#     Xt_ = None
#     kf = Filter(Mt_)
#     kf.gen_Xt(Xt_)
#     expected_result = np.zeros((1, 1))
#     result = kf.Xt[0]
#     # np.testing.assert_equal(expected_result, result)


