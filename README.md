# linkalman

`linkalman` is a python package that solves linear structural time series models with Gaussian noises. Compared with some other popular Kalman filter packages written in python, linkalman has a combination of several advantages:

  - Account for partially and fully incomplete measurements 
  - Flexible and convenient model structure
  - Robust and efficient implementation
  - Proper implementation for unknown priors
  - Built-in numerical and EM algorithm
  - Open-source with a comprehensive user manual 
  - Modular design with intuitive model specification

### Installation
`linkalman` requires the following packages to run:

  - numpy
  - pandas
  - networkx
  - scipy
 
To install `linkalman`, simply use the standard `pip` command:

```sh
$ pip install linkalman
```
### Example
Here I will provide a simple example using `linkalman`. See [here](https://github.com/DanyangSu/linkalman/tree/master/examples/jupyter_notebooks) for more examples, and [user's manual](https://github.com/DanyangSu/linkalman/blob/master/doc/manual.pdf) for technical details.

```python
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from linkalman.models import BaseConstantModel as BCM
import matplotlib.pyplot as plt


# Get data
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv')
df['x'] = 1
df.set_index('Date', inplace=True)
```

First we define the system dynamics of a Bayesian Structural Time Series (BSTS) model. Here I define a Stochastic linear trend model to extract the trend information from the time series (referring to the example section of [user's manual](https://github.com/DanyangSu/linkalman/blob/master/doc/manual.pdf) for details)
```python
def my_f(theta):
    sig1 = np.exp(theta[0])
    sig2 = np.exp(theta[1])
    sig3 = np.exp(theta[2])

    F = np.array([[1, 1], [0, 1]])
    Q = np.array([[sig1, 0], [0, sig2]]) 
    R = np.array([[sig3]])
    H = np.array([[1, 0]])
    # Collect system matrices
    M = {'F': F, 'Q': Q, 'H': H, 'R': R}

    return M 
```
Next we define a solver or optimizer, you can choose any solver you prefer. Here I just use `scipy.optimize.minimize`.
```python
def my_solver(param, obj_func, verbose=False, **kwargs):
    obj_ = lambda x: -obj_func(x)
    res = minimize(obj_, param, **kwargs)
    theta_opt = np.array(res.x)
    fval_opt = res.fun
    return theta_opt, fval_opt
```
Now we can fit the data. First we initialize the model and feed the system dynamics (`my_f`) and solver (`my_solver`). You may also pass the keyworded arguments to for `my_f` and `my_solver`.
```python
model = BCM()
model.set_f(my_f)
model.set_solver(my_solver, method='nelder-mead', 
        options={'xatol': 1e-8, 'disp': True, 'maxiter': 10000})
theta_init = np.random.rand(3)
model.fit(df, theta_init, y_col=['Births'], x_col=['x'], 
              method='LLY')
df_LLY = model.predict(df)
```
That is it! If you want to do additional work, you can do the following to plot a confidence interval around your predictions.
```python
df_LLY['kf_ub'] = df_LLY.Births_filtered + 1.96 * np.sqrt(df_LLY.Births_fvar)
df_LLY['kf_lb'] = df_LLY.Births_filtered - 1.96 * np.sqrt(df_LLY.Births_fvar)
df_LLY = df_LLY[df_LLY.index > '1959-01-01']
df_LLY.index = pd.to_datetime(df_LLY.index)

# Define plot function
def simple_plot(df, col_est, col_actual, col_ub, col_lb, label_est,
                label_actual, title, figsize=(12, 8)):
    ax = plt.figure(figsize=figsize)
    plt.plot(df.index, df[col_est], 'r', label=label_est)
    plt.scatter(df_LLY.index, df[col_actual], s=20, c='b', 
                marker='o', label=label_actual)
    plt.fill_between(df.index, df[col_ub], df[col_lb], color='g', alpha=0.2)
    ax.legend(loc='right', fontsize=9)
    plt.title(title, fontsize=22)
    plt.show()
simple_plot(df_LLY, 'Births_filtered', 'Births', 'kf_ub', 'kf_lb',  
           'Prediction', 'Births', 'Filtered Births Data')
```

### License

3-Clause BSD

