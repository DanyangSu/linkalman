import numpy as np
import pandas as pd
import linkalman
import scipy
from linkalman.core import Filter, Smoother
from linkalman.core.utils import simulated_data, ft, df_to_list, list_to_df, gen_PSD, get_ergodic, create_col, clean_matrix


# Define Mt
def get_f(theta):
    """.
    This example illustrates the basic usage of Filter and Smoother in
    the presence of complete or incomplete missing measurements.
    If you are working with HMM with Mt=M, you simply need to 
    specify a mapping theta->M. You can use simulated_data
    to genereate a simulated data and verify the model performance.
    If the goal is to estimate theta, you need to make sure that
    theta in the model is properly identified. This example implements
    the Application Section in the user manual
    """

    # Define theta
    rho_delta = np.exp(theta[0]) / (np.exp(theta[0]) + 1)
    Delta_delta = theta[1]
    rho = np.exp(theta[2]) / (np.exp(theta[2]) + 1)
    Lambda = np.exp(theta[3]) / (np.exp(theta[3]) + 1) * np.pi
    b = theta[4]
    d = theta[5]
    D1 = theta[6]
    D2 = theta[7]
    pi = theta[8]
    sigma_mu = np.exp(theta[9])
    sigma_delta = np.exp(theta[10])
    sigma_gamma = np.exp(theta[11])
    sigma_psi = np.exp(theta[12])
    cov_w = theta[13:16]

    # Generate F
    F = np.zeros([10, 10])
    F[0][0:2] = 1
    F[2][2:8] = -1
    for i in range(3,8):
        F[i][i-1] = 1

    rcos = rho * np.cos(Lambda)
    rsin = rho * np.sin(Lambda)
    F_cycle = np.array([[rcos, rsin], [-rsin, rcos]])
    F[8:10, 8:10] = F_cycle

    # Generate Q
    Q = np.zeros([10, 10])
    Q[0][0] = sigma_mu
    Q[1][1] = sigma_delta
    Q[2][2] = sigma_gamma
    Q[8][8] = sigma_psi
    Q[9][9] = sigma_psi

    # Generate R
    R = gen_PSD(cov_w, 2)

    # Generate B
    B = np.zeros([10, 3])
    B[0][1] = b
    B[1][0] = (1-rho_delta)*Delta_delta

    # Generate D
    D = np.array([[0, 0, D1],[d, 0, D2]])

    # Generate H
    H = np.zeros([2, 10])
    for i in [0, 2, 8]:
        H[0][i] = 1
        H[1][i] = pi

    # Collect system matrices
    M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'B': B, 'D': D}

    return M


# Gemerate fake data
T = 1000
theta = np.random.rand(16)
f = lambda x: get_f(x)
Mt = ft(theta, f, T)

# Generate data
x1 = np.ones(T)
x2 = np.zeros(T)
x2[300] = 1
x3 = np.random.rand(T)
Xt = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3}) 
df, y_col, xi_col = simulated_data(Mt, Xt, T=T)

# Create some missingness in the data
df.loc[(df.index % 5 == 0) & 
        (df.index % 7 == 0), y_col[0]] = np.nan
df.loc[df.index % 7 != 0, y_col[1]] = np.nan
Y_t = df_to_list(df[y_col])
X_t = df_to_list(df[Xt.columns])

# Fit filered data
kf = Filter(Mt)
kf(Y_t, X_t)
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


