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
    AR(1) model. Note that it is important to get the correct
    specification. It's important to note that a constant 
    factor is needed for estimating more accurate theta, 
    becaseu it will absorb some biasness.In general, 
    however, MLE is biased, so the focus should be on prediction
    fit, not parameter estimation. Here is an example where I 
    drop c. The formula here for Ar(1) is:
    y_t = Fy_{t-1} + epsilon_{t-1}
    The performance without constant c is generally poorer than
    with c.
    """

    # Define theta
    phi_1 = 1 / (np.exp(theta[0])+1)
    sigma = np.exp(theta[1]) 

    # Generate F
    F = np.array([[phi_1]])

    # Generate Q
    Q = np.array([[sigma]])
    
    # Generate R
    R = np.array([[0]])

    # Generate H
    H = np.array([[1]])

    # Collect system matrices
    M = {'F': F, 'Q': Q, 'H': H, 'R': R}

    return M


############################################################
# You can skip this part if you already have a dataset, and
# load that dataset instead.

# Gemerate fake data
theta = np.array([-0.2, -0.1])
num_params = len(theta)
T = 1000
cutoff_t = np.floor(T * 0.7).astype(int)
offset_t = np.floor(T * 0.9).astype(int)  

# Use this function to convert f with more parameters
my_ft = lambda theta, t: ft(theta, my_f, t)

# Generate data
df, y_col, xi_col = simulated_data(my_ft, theta, T=T)

# Create a training set
df_train = df.iloc[0:cutoff_t].copy()
df_test = df.copy()  # Keep the full data for forward prediction

# Create an offset:
df_test['y_0_vis'] = df_test.y_0.copy()  # fully visible y
df_test.loc[df.index >= offset_t, ['y_0']] = np.nan

############################################################


# Get true log likelihood
kf = Filter(my_ft)
Yt = df_to_list(df_train, y_col)
kf(theta, Yt)
true_LL = kf.get_LL()
print('The true log likelihood is: {}'.format(true_LL))

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
theta_init = np.random.rand(num_params)
model.fit(df_train, theta_init, y_col=y_col)

print(theta)
print(model.theta_opt)

# Make predictions:
df_pred = model.predict(df_test)

# Make predictions on true theta
df_pred_act = model.predict(df_test, theta)

df_pred_act.to_csv('/Users/dsu/pred_true.csv')
df_pred.to_csv('/Users/dsu/pred.csv')
df_test.to_csv('/Users/dsu/actuals.csv')

