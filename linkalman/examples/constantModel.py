import numpy as np
import pandas as pd
import linkalman
import scipy
from linkalman.models import BaseConstantModel as BCM
from linkalman.core.utils import simulated_data, ft, df_to_list, list_to_df, \
        gen_PSD, get_ergodic, create_col, clean_matrix
from linkalman.core import Filter
import nlopt

def my_f(theta):
    """.
    This example illustrates the basic usage of Filter and Smoother in
    the presence of complete or incomplete missing measurements.
    If you are working with BSTS with Mt=M, you simply need to 
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
    F[1][1] = rho_delta
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


############################################################
# You can skip this part if you already have a dataset, and
# load that dataset instead.

# Gemerate fake data
theta = np.random.rand(16)
theta = 0.2 * np.ones(16)
T = 1000
cutoff_t = np.floor(T * 0.7).astype(int)

# Use this function to convert f with more parameters
my_ft = lambda theta, t: ft(theta, my_f, t)

# Generate data
x1 = np.ones(T)
x2 = np.zeros(T)
x2[T//2] = 1
x3 = np.random.rand(T)
x_col = ['x1', 'x2','x3']
Xt = pd.DataFrame({x_col[0]: x1, x_col[1]: x2, x_col[2]: x3}) 
df, y_col, xi_col = simulated_data(my_ft, theta, Xt)

# Create someissingness in the data
df.loc[df.index % 7 != 0, y_col[1]] = np.nan

# Create a training set
df_train = df.iloc[0:cutoff_t].copy()
df_test = df.copy()  # Keep the full data for forward prediction

############################################################


# Get true log likelihood
Mt = my_ft(theta, cutoff_t)
kf = Filter(Mt)
Yt = df_to_list(df_train[y_col])
Xt = df_to_list(df_train[x_col])
kf(Yt, Xt)
print('The true log likelihood is: {}'.format(kf.get_LL()))

# Build a suitable solver: 
# f: solver(theta, obj, **kwargs) -> theta_opt

def my_solver(param, obj, **kwargs):
    # conform to the nlopt format
    def nlopt_obj(x, grad):
        output = obj(x)
        print('{}-{}'.format(output,list(x)))
        return output

    opt = nlopt.opt(nlopt.LN_BOBYQA, param.shape[0])
    opt.set_max_objective(nlopt_obj)
    opt.set_xtol_rel(1e-4)
    opt.verbose = 1
    theta_opt = opt.optimize(param)
    return theta_opt

# Initialize the model
model = BCM(method='LLY')
model.get_f(my_f)
model.get_solver(my_solver) 

# Fit data:
theta_init = np.random.rand(16)
x_col = ['x1', 'x2', 'x3']
model.fit(df_train, theta_init, y_col=y_col, x_col=x_col)

print(theta)
print(model.theta_opt)

# # Make predictions:
# df_pred = model.predict(df)

