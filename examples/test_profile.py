#!/usr/bin/env python
# coding: utf-8

# # Model Specification
# This example solve a standard AR(1) process but with multiple noise measurements. If there are many parameters, we need more data for proper estimation.

# In[11]:



import numpy as np
import pandas as pd
import linkalman
import scipy
from linkalman.models import BaseConstantModel as BCM
from linkalman.core.utils import gen_PSD
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy


# # Unrestricted Parametrization of Covariance Matrices
# Sometime we want to let a covariance matrix (e.g. `R`) to be fully parametrized without restriction (e.g. PSD of `R`). Here I use `linkalman.core.utils.gen_PSD` to achieve this. It uses Cholesky decomposition with strictly non-negative diagonal values to achieve unique and restriction-free parametrizaion.   

# In[12]:


def my_f(theta):
    """
    AR(1) model. Introduce noise and 
    """
    # Define theta
    f = 1 / (1 + np.exp(theta[3]))
    sigma = np.exp(theta[4]) 
    # Generate F
    F = np.array([[f]])
    # Generate Q
    Q = np.array([[sigma]]) 
    # Generate R, set to 0 to be consistent with AR(1) process
    R = gen_PSD(theta[0:3], 2)  # need three parameters to define a 2-by-2 R
    # Generate H
    H = np.array([[1], [theta[5]]])  # enforce one coefficient to be 1 to make the system more identifiable.
    # Generate D
    D = np.array([[theta[6]], [theta[7]]])
    # Collect system matrices
    M = {'F': F, 'Q': Q, 'H': H, 'R': R, 'D': D}

    return M


# In[13]:


def my_solver(param, obj_func, verbose=False, **kwargs):
    """
    Simple solver for LLY
    """
    obj_ = lambda x: -obj_func(x)
    def disp_f(x):
        print('theta is {}. Function value is: {}.'.format(x, obj_func(x)))
    callbackf = None
    if verbose:
        callbackf = disp_f
    res = minimize(obj_, param, callback=callbackf, **kwargs)
    theta_opt = np.array(res.x)
    fval_opt = res.fun
    return theta_opt, fval_opt


# In[14]:


# Initialize the model
x = 1  # used to calculate stationary mean
model = BCM()
model.set_f(my_f, x_0=x * np.ones([1, 1]))
model.set_solver(my_solver, method='nelder-mead', 
        options={'xatol': 1e-8, 'maxfev': 200}, verbose=False) 


# # Generate Synthetic Data
# Same as the standard setup, but I cross off some measurements during training period and see how `linkalman` handles them. I generate some partial missing data for each of the measurements.

# In[15]:


# Some initial parameters
theta = np.array([0.1, 0.3, 0.1, -0.5, -0.1, 2, 4, 5])
T = 3000  
train_split_ratio = 0.7
forecast_cutoff_ratio = 0.8  

missing_range_1st = [0.3, 0.4]  # range of missing for the first measurement
missing_range_2nd_end = 0.5  # end ratio of missing for the second measurement

# Split train data
train_split_t = np.floor(T * train_split_ratio).astype(int)

# Generate missing data for forcasting
forecast_t = np.floor(T * forecast_cutoff_ratio).astype(int)

# If we want AR(1) with non-zero stationary mean, we should proivde a constant 
x_col = ['const']
Xt = pd.DataFrame({x_col[0]: x * np.ones(T)})  # use x to ensure constant model

# Build simulated data
df, y_col, xi_col = model.simulated_data(input_theta=theta, Xt=Xt)

# Store fully visible y for comparison later
df['y_0_vis'] = df.y_0.copy()  
df['y_1_vis'] = df.y_1.copy()

# Insert some missingness
missing_start1_t = np.floor(T * missing_range_1st[0]).astype(int)
missing_end1_t = np.floor(T * missing_range_1st[1]).astype(int)
missing_start2_t = missing_end1_t 
missing_end2_t = np.floor(T * missing_range_2nd_end).astype(int)

is_missing1 = (df.index >= missing_start1_t) & (df.index < missing_end1_t)
is_missing2 = (df.index >= missing_end1_t) & (df.index < missing_end2_t)
df.loc[is_missing1, ['y_0']] = np.nan
df.loc[is_missing2, ['y_1']] = np.nan

# Splits models into three groups
is_train = df.index < train_split_t
is_test = (~is_train) & (df.index < forecast_t)
is_forecast = ~(is_train | is_test)

# Create a training and test data
df_train = df.loc[is_train].copy()
df_test = df.copy()  

# Create an offset
df_test.loc[is_forecast, ['y_0', 'y_1']] = np.nan


# # Fit and Predict

# In[18]:


# Fit data using LLY:
start_time = datetime.datetime.now()
theta_init = np.random.rand(len(theta))
model.fit(df_train, theta_init, y_col=y_col, x_col=x_col, method='LLY')
end_time = datetime.datetime.now()
print('Runtime: {} seconds.'.format((end_time - start_time).seconds))


# # Check Model Performance
# Here I check filtered and smoothed estimates for both `y_0` and `y_1`

# In[19]:

