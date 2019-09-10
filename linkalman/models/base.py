import numpy as np
import pandas as pd
from typing import List, Any, Callable, Dict
from collections.abc import Sequence
from ..core import Filter, Smoother
from ..core.utils import df_to_list, list_to_df, simulated_data, \
        get_diag, ft, Constant_M, create_col
from copy import deepcopy
from numpy.random import multivariate_normal
from inspect import signature

__all__ = ['BaseOpt', 'BaseConstantModel']


class BaseOpt(object):
    """
    BaseOpt is the core of model solvers. It directly interacts 
    with the Kalman filters/smoothers, and can be inherited by 
    both constant M models and more complicated nonconstant Mt models.
    """
    
    def __init__(self, method: str='mix') -> None:
        """
        Initialize base model. 

        Parameters:
        ----------
        method : method to fit. Default "mix" if not specified
        """
        self.ft = None
        self.theta_opt = None
        self.x_col = None
        self.y_col = None
        self.solver = None
        self.method = method
        if self.method not in ['mix','EM','LLY']:
            raise ValueError('method must be "mix", "EM", or "LLY".')


    def set_f(self, Ft: Callable) -> None:
        """
        Mapping from theta to M. Ft must be the form: 
        f: Ft(theta, T) -> [M_1, M_2,...,M_T]. 

        Parameters:
        ----------
        Ft : theta -> Mt
        """
        # Check Ft
        sig = signature(Ft)
        if len(sig.parameters) != 2:
            raise TypeError('Ft must have two positional arguments')

        self.ft = Ft


    def set_solver(self, solver: Any) -> None:
        """
        Get solver object for the model. The solver must be 
        solver(theta, obj, **kwargs) where theta is the paramter,
        obj is the objective function (e.g. likelihood), and **kwargs
        are kwargs for the solver object. The solver should return 
        optimal theta

        Parameters:
        ----------
        solver : a solver object
        """
        self.solver = solver


    def fit(self, df: pd.DataFrame, theta_init: np.ndarray,
            y_col: List[str], x_col: List[str]=None, **kwarg) -> None:
        """
        Fit the model and returns optimal theta. 

        Parameters:
        ----------
        df : input data to be fitted
        theta_init : initial theta. self.Ft(theta_init) produce 
            Mt for first iteration of EM algorithm.
        y_col : list of columns in df that belong to Yt
        x_col : list of columns in df that belong to Xt. May be None
        kwarg : options for optimizers
        """
        # Raise exception if x_col or y_col is not list
        if x_col is not None:
            if not isinstance(x_col, list):
                raise TypeError('x_col must be a list.')

        if not isinstance(y_col, list):
            raise TypeError('y_col must be a list.')

        if self.ft is None:
            raise ValueError('Need ft')

        if self.solver is None:
            raise ValueError('Need solver')
        
        # Raise exception if Ft no callable
        if not isinstance(self.ft, Callable):
            raise TypeError('ft must be a function')

        # Preprocess data inputs
        self.x_col = x_col
        self.y_col = y_col

        # If x_col is given, convert dataframe to lists
        Xt = df_to_list(df, x_col)
        Yt = df_to_list(df, y_col)

        # Run solver
        if self.method == 'LLY': 
            obj = lambda theta: self.get_LLY(theta, Yt, Xt)
            self.theta_opt = self.solver(theta_init, obj, **kwarg)

        elif self.method == 'EM':
            dist = 1
            self.theta_i = deepcopy(theta_init)
            self.G_i = np.inf
            while dist > self.EM_threshold:
                obj = lambda theta: self.get_LLEM(theta, Yt, Xt)
                theta_opt, G_opt = self.solver(self.theta_i, obj, **kwarg)
                dist = np.abs(self.G_i - G_opt)
                self.G_i = G_opt
                self.theta_i = deepcopy(theta_opt)
            self.theta_opt = deepcopy(theta_opt)

        else:

            # Cold start with EM
            for EM_iter in range(self.num_iter):
                obj = lambda theta: self.get_LLEM(theta, Yt, Xt)
                theta_opt, G_opt = self.solver(self.theta_i, obj, **kwarg)
                self.G_i = G_opt
                self.theta_i = deepcopy(theta_opt)

            # Warm start with LLY
            obj = lambda theta: self.get_LLY(theta, Yt, Xt)
            self.theta_opt = self.solver(theta_i, obj, **kwarg)


    def get_LLEM(self, theta: List[float], Yt: List[np.ndarray],
            Xt: List[np.ndarray]=None) -> float:
        raise ValueError('need add')


    def get_LLY(self, theta: List[float], Yt: List[np.ndarray], 
            Xt: List[np.ndarray]=None) -> float:
        """
        Wrapper for calculating LLY. Used as the objective 
        function for optimizers.

        Parameters:
        ----------
        theta : paratmers
        Yt : list of measurements
        Xt : list of regressors. May be None

        Returns:
        ----------
        lly : log likelihood from Kalman filters
        """
        kf = Filter(self.ft)
        kf(theta, Yt, Xt)
        return kf.get_LL()


    def predict(self, df: pd.DataFrame, theta: np.ndarray=None) -> pd.DataFrame: 
        """
        Predict time series. df should contain both training and 
        test data. If Yt is not available for some or all test data,
        use np.nan as placeholders. Accept user-supplied theta as well

        Parameters:
        ----------
        df : df to be predicted. Use np.nan for missing Yt
        theta : override theta_opt using user-supplied theta

        Returns:
        ----------
        df_fs : Contains filtered/smoothed y_t, xi_t, and P_t
        """
        # Generate system matrices for prediction
        Xt = df_to_list(df, self.x_col)
        Yt = df_to_list(df, self.y_col)
        
        # Generate filtered predictions
        kf = Filter(self.ft)

        # Override theta_opt if theta is not None
        if theta is not None:
            kf(theta, Yt, Xt)
        else:
            kf(self.theta_opt, Yt, Xt)

        y_col_filter = create_col(self.y_col, suffix='_filtered')
        y_filter_var = create_col(self.y_col, suffix='_fvar')
        Yt_filtered, Yt_P = kf.get_filtered_y()
        Yt_P_diag = get_diag(Yt_P)
        df_Yt_filtered = list_to_df(Yt_filtered, y_col_filter)
        df_Yt_fvar = list_to_df(Yt_P_diag, y_filter_var)

        # Generate smoothed predictions
        ks = Smoother()
        ks(kf)
        y_col_smooth = create_col(self.y_col, suffix='_smoothed')
        y_smooth_var = create_col(self.y_col, suffix='_svar')
        Yt_smoothed, Yt_P_smooth = ks.get_smoothed_y()
        Yt_P_smooth_diag = get_diag(Yt_P_smooth)
        df_Yt_smoothed = list_to_df(Yt_smoothed, y_col_smooth)
        df_Yt_svar = list_to_df(Yt_P_smooth_diag, y_smooth_var)
        df_fs = pd.concat([df_Yt_filtered, df_Yt_fvar,
            df_Yt_smoothed, df_Yt_svar], axis=1)

        return df_fs
        

class BaseConstantModel(BaseOpt):
    """
    Any BSTS model with constant system matrices may inherit this class.
    The child class should provide get_f function. It inherits from BaseOpt
    and has a customized get_f
    """
    def get_f(self, f: Callable) -> None:
        """
        Mapping from theta to M. Provided by children classes.
        Must be the form of get_f(theta). If defined, it should
        return system matrix M

        Parameters:
        ----------
        f : theta -> M
        """
        # Raise exception if Ft no callable
        self.ft = lambda theta, T: ft(theta, f, T)


