import numpy as np
import pandas as pd
import linkalman
from linkalman.models import BaseConstantModel as BCM
from linkalman.models import BaseEM
from linkalman.core import Filter, Smoother

class SimpleEM(BCM):
    """
    Solve HMM with 1-D xi and y. The dimension of B and D are 
    determined by x. The model is suitable for y and x with 
    low dimensions.

    This example illustrates the basic usage of linkalman. 
    If you are working with HMM with Mt=M, you simply need to 
    specify a mapping theta->M. You can use BaseEM.simulated_data
    to genereate a simulated data and verify the model performance.
    If the goal is to estimate theta, you need to make sure that
    theta in the model is properly identified.
    """

    def __init__(self):
        super().__init__()

    def get_f(self, theta):
        """
        Create mapping M=f(theta, **kwargs).This function 
        should only accept theta as the only argument. The
        idea is that the Mt is fully characterized by theta.
        """
        dim_x = 2
        F = np.array([[theta[0]]])
        Q = np.array([[np.exp(theta[1])]])
        H = np.array([[theta[2]]])
        R = np.array([[np.exp(theta[3])]])
        xi_1_0 = np.array([[theta[4]]])
        P_1_0 = np.array([[np.exp(theta[5])]])
        B = np.array([[theta[i] for i in range(6, 6 + dim_x)]])
        D = np.array([[theta[i] for i in range(6 + dim_x, 6 + dim_x * 2)]])
        
        M = {'F': F, 'Q': Q, 'H': H, 'B': B, 'D': D, 
                'R': R, 'xi_1_0': xi_1_0, 'P_1_0': P_1_0}
        return M

# Generate Xt
T = 1000
x_col = ['x_0', 'x_1']
val = np.random.multivariate_normal(np.zeros(2), 
        np.array([[0.3, 0.3], [0.3, 0.4]]), T)
Xt = pd.DataFrame(data=val, columns=x_col)
simple_ts = SimpleEM()
theta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.7, 0.8, 0.9, 1]
Mt = simple_ts.F_theta(theta, simple_ts.f, T)

# Generate data
df = BaseEM.simulated_data(Xt, Mt)
Yt = BaseEM._df_to_list(df[['y_0']])
Xt = BaseEM._df_to_list(df[['x_0', 'x_1']])
# Fit data
kf = Filter(Mt)
kf(Xt, Yt)
ks = Smoother()
ks(kf)


