import numpy as np
import pandas as pd
from typing import List, Any, Callable
from collections.abc import Sequence
from ..core import EM
from ..core import Smoother
from ..core.utils import df_to_list, list_to_df, simulated_data, ft, Constant_M
from copy import deepcopy
from numpy.random import multivariate_normal

__all__ = ['BaseEM', 'BaseConstantModel']


class BaseEM(object):
    """
    BaseEM is the core of models using EM algorithms. It directly interacts 
    with the EM engine, and can be inherited by both constant M models and 
    more complicated nonconstant Mt models.
    """
    
    def __init__(self, Ft: Callable) -> None:
        """
        Initialize Base EM model. 

        Parameters:
        ----------
        Ft : function that takes theta and returns Mt
        """
        
        # Raise exception if Ft no callable
        if not isinstance(Ft, Callable):
            raise TypeError('Ft must be a function')

        self.Ft = Ft
        self.theta_opt = None
        self.x_col = None
        self.y_col = None


    def fit(self, df: pd.DataFrame, theta_init: List[float], 
            y_col: List[str], x_col: List[str]=None) -> None:
        """
        Fit the model using EM algorithm. Produces optimal 
        theta. 

        Parameters:
        ----------
        df : input data to be fitted
        theta_init : initial theta. self.Ft(theta_init) produce 
            Mt for first iteration of EM algorithm.
        y_col : list of columns in df that belong to Yt
        x_col : list of columns in df that belong to Xt. May be None
        """
        # Initialize
        em = EM(self.Ft)
        self.x_col = x_col
        self.y_col = y_col

        # Convert dataframe to lists
        Yt = self._df_to_list(df[y_col])

        # If x_col is given, convert dataframe to lists
        if x_col is not None:
            Xt = self._df_to_list(df[x_col])
        else:
            Xt = None

        # Run EM solver
        self.theta_opt = em.fit(theta_init, Yt, Xt)


    def predict(self, df: pd.DataFrame, Ft: Callable) -> Smoother: 
        """
        Predict time series. df_extended should contain 
        both training and test data.

        Parameters:
        ----------
        df : df to be predicted. Use np.nan for missing Yt
        Ft : should be consistent with self.Ft for t <= T

        Returns:
        ----------
        ks : Contains filtered/smoothed y_t, xi_t, and P_t
        """
        # Generate system matrices for prediction
        Mt = Ft(self.theta_opt)
        Xt = self._df_to_list(df_extended[self.x_col])
        Yt = self._df_to_list(df_extended[self.y_col])

        # Run E-step and get filtered/smoothed y_t, xi_t, and P_t
        ks = EM.E_step(Mt, Xt, Yt)
        return ks
        

class BaseConstantModel(object):
    """
    Any HMM with constant system matrices may inherit this class.
    The child class should provide get_f function.
    """

    def __init__(self) -> None:
        """
        Initialize self.f 
        """
        self.f = lambda theta: self.get_f(theta)
        self.mod = None  # placeholder for BaseEM object


    def get_f(self, theta: List[float]) -> None:
        """
        Mapping from theta to M. Provided by children classes.
        Must be the form of get_f(theta). If defined, it should
        return system matrix M

        Parameters:
        ----------
        theta : placeholder for system parameter
        """
        raise NotImplementedError


    def fit(self, df: pd.DataFrame, theta_init: List[float], 
            x_col: List[str], y_col: List[str]) -> None:
        """
        Invoke BaseEM.fit to fit the data.

        Parameters:
        ----------
        df : data to be fitted
        theta_init : initial theta, generate first Mt
        x_col : columns in df that belong to Xt
        y_col : columns in df that belong to Yt
        """
        # Raise exception if x_col or y_col is not list
        if not isinstance(x_col, list):
            raise TypeError('x_col must be a list.')
        if not isinstance(y_col, list):
            raise TypeError('y_col must be a list.')

        # Collect dimensions of Xt and Yt
        T = df.shape[0]

        # Create F
        F = lambda theta: self.F_theta(theta, self.f, T)

        # Fit model using ConstantEM
        ConstEM = BaseEM(F)
        ConstEM.fit(df, theta_init, x_col, y_col)
        self.mod = ConstEM


    def predict(self, df: pd.DataFrame) -> Smoother:
        """
        Predict fitted values from Kalman Filter / Kalman Smoother
        Ft is extended to fit the size of the input df.

        Parameters:
        ----------
        df : input dataframe. Must contain both the training set and
            the prediction set.

        Returns:
        ----------
        ks : Contains fitted y_t, xi_t, and P_t
        """
        # Update Ft
        T = df.shape[0]
        Ft = lambda theta: self.F_theta(theta, self.f, T)

        # Generate a smoother object that stores fitted values
        ks = self.mod.predict(df, Ft)
        return ks
