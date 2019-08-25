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
    """
    AR(2) model
    """

    # Define theta
    phi_1 = 1 / (np.exp(theta[0])+1)
    phi_2 = 1 / (np.exp(theta[1])+1)
    sigma = np.exp(theta[2]) 

    # Generate F
    F = np.array([[phi_1, phi_2], [1, 0]])

    # Generate Q
    Q = np.array([[sigma, 0], [0, 0]])
    
    # Generate R
    R = np.array([[0]])

    # Generate H
    H = np.array([[1, 0]])

    # Collect system matrices
    M = {'F': F, 'Q': Q, 'H': H, 'R': R}

    return M


############################################################
# You can skip this part if you already have a dataset, and
# load that dataset instead.

# Gemerate fake data
theta = np.array([-0.2, -0.2, -4])
T = 1000
cutoff_t = np.floor(T * 0.7).astype(int)

# Use this function to convert f with more parameters
my_ft = lambda theta, t: ft(theta, my_f, t)

# Generate data
df, y_col, xi_col = simulated_data(my_ft, theta, T=T)

# Create someissingness in the data
# df.loc[df.index % 7 == 0, y_col[0]] = np.nan

# Create a training set
df_train = df.iloc[0:cutoff_t].copy()
df_test = df.copy()  # Keep the full data for forward prediction

############################################################


# Get true log likelihood
Mt = my_ft(theta, cutoff_t)
kf = Filter(Mt)
Yt = df_to_list(df_train, y_col)
kf(Yt)
print('The true log likelihood is: {}'.format(kf.get_LL()))

# Build a suitable solver: 
# f: solver(theta, obj, **kwargs) -> theta_opt

def my_solver(param, obj, **kwargs):
    # conform to the nlopt format
    def nlopt_obj(x, grad):
        output = obj(x)
        print(output)
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
theta_init = np.random.rand(3)
model.fit(df_train, theta_init, y_col=y_col)

print(theta)
print(model.theta_opt)

# Make predictions:
df_pred = model.predict(df)

df_pred.to_csv('/Users/dsu/pred.csv')
df.to_csv('/Users/dsu/actuals.csv')
