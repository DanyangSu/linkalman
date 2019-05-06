import numpy as np
import pandas as pd
import linkalman
from linkalman.core import Filter, Smoother
from linkalman.core.utils import simulated_data, ft, df_to_list, list_to_df, gen_PSD, create_col


# Define Mt
def get_f(theta):
    """
    Solve HMM with 1-D xi and 2-D y. Xt is None.

    This example illustrates the basic usage of Filter and Smoother in
    the presence of complete or incomplete missing measurements.
    If you are working with HMM with Mt=M, you simply need to 
    specify a mapping theta->M. You can use simulated_data
    to genereate a simulated data and verify the model performance.
    If the goal is to estimate theta, you need to make sure that
    theta in the model is properly identified.
    """
    F = np.array(theta[0]).reshape(-1, 1)
    Q = np.array([[np.exp(theta[1])]])
    H = np.array(theta[2:4]).reshape(-1, 1)
    R = gen_PSD(np.exp(theta[4:7]), 2)
    xi_1_0 = np.array([[theta[7]]])
    P_1_0 = np.array([[np.exp(theta[8])]])
    
    M = {'F': F, 'Q': Q, 'H': H, 'R': R, 
            'xi_1_0': xi_1_0, 'P_1_0': P_1_0}
    return M

T = 1000
theta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
f = lambda x: get_f(x)
Mt = ft(theta, f, T)

# Generate data
df, y_col, xi_col = simulated_data(Mt, T=T)

# Create some missingness in the data
df.loc[df.index % 5 == 0, y_col[0]] = np.nan
df.loc[df.index % 10 == 0, y_col[1]] = np.nan
Yt = df_to_list(df[y_col])


# Fit filered data
kf = Filter(Mt)
kf(Yt)
y_col_filter = create_col(y_col, suffix='_filtered')
Yt_filtered, _ = kf.get_filtered_y()
df_Yt_filtered = list_to_df(Yt_filtered, y_col_filter)

# Fit smoothed data
ks = Smoother()
ks(kf)
y_col_smoother = create_col(y_col, suffix='_smoothed')
Yt_smoothed, _ = ks.get_smoothed_y()
df_Yt_smoothed = list_to_df(Yt_smoothed, y_col_smoother)

# Produce final output
df_output = pd.concat([df, df_Yt_filtered, df_Yt_smoothed], axis=1) 


