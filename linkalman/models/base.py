import numpy as np
import pandas as pd
from typing import List, Any, Callable, Dict, Tuple
from ..core import Filter, Smoother
from ..core.utils import df_to_list, list_to_df, \
        simulated_data, get_diag, ft, create_col
import warnings
warnings.simplefilter('default')

__all__ = ['BaseOpt', 'BaseConstantModel']


class BaseOpt(object):
    """
    BaseOpt is the core of model solvers. It directly interacts 
    with the Kalman filters/smoothers, and can be inherited by 
    both constant M models and more complicated non-constant Mt models.
    """
    
    def __init__(self) -> None:
        """
        Initialize base model. 
        """
        self.ft = None
        self.ft_kwargs = None
        self.solver_kwargs = None
        self.theta_opt = None
        self.fval_opt = None
        self.x_col = None
        self.y_col = None
        self.solver = None

        # Store fitted Kalman smoother 
        self.ks_fitted = None


    def set_f(self, Ft: Callable, reset: bool=True, 
            **ft_kwargs) -> None:
        """
        Mapping from theta to M. Ft must be the form: 
        f: Ft(theta, T) -> [M_1, M_2,...,M_T]. 

        Parameters:
        ----------
        Ft : theta -> Mt
        ft_kwargs : kwargs for ft
        reset : if true, reset fitted values
        """
        self.ft = Ft
        self.ft_kwargs = ft_kwargs
        if reset:
            self.theta_opt = None
            self.fval_opt = None
            self.ks_fitted = None


    def set_solver(self, solver: Any, **solver_kwargs) -> None:
        """
        Get solver object for the model. The solver must be 
        solver(theta, obj, **kwargs) where theta is the paramter,
        obj is the objective function (e.g. likelihood), and **kwargs
        are kwargs for the solver object. The solver should return 
        optimal theta and optimal LL evaluation

        Parameters:
        ----------
        solver : a solver object
        kwargs : kwargs for solver
        """
        self.solver = solver
        self.solver_kwargs = solver_kwargs


    def fit(self, df: pd.DataFrame, theta_init: np.ndarray,
            y_col: List[str], x_col: List[str]=None, 
            method: str='LLY', EM_threshold: float=1e-3, 
            num_EM_iter: int=np.inf, post_min_iter: int=100, 
            EM_stopping_rate: float=0.01, init_state: Dict=None) -> None:
        """
        Fit the model and returns optimal theta. 

        Parameters:
        ----------
        df : input data to be fitted
        theta_init : initial theta. self.Ft(theta_init) produce 
            Mt for first iteration of EM algorithm.
        y_col : list of columns in df that belong to Yt
        x_col : list of columns in df that belong to Xt. May be None
        method : EM or LLY
        EM_threshold : stopping criteria
        num_EM_iter : number of iterations for EM algorithms
        post_min_iter : after a new minimum is set, terminate iteration 
            after the number of runs determined by this argument
        EM_stopping_rate : weight placed on counter, higher rate means 
            update theta_opt slower. EM_stopping_rate = 0 means full 
            weight on theta_opt after each iteration
        init_state : user-specified initial state values
        """
        # Raise exception if method is not correctly specified
        if method not in ['EM','LLY']:
            raise ValueError('method must be "EM", or "LLY".')

        # Raise exception if EM_stopping_rate is negative
        if EM_stopping_rate < 0:
            raise ValueError('EM_stopping_rate must be non-negative')

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

        # Pre-process data inputs
        self.x_col = x_col
        self.y_col = y_col

        # If x_col is given, convert dataframe to lists
        Xt = df_to_list(df, x_col)
        Yt = df_to_list(df, y_col)

        # Run solver
        if method == 'LLY': 
            obj = lambda theta: self.get_LLY(theta, Yt, Xt)
            self.theta_opt, self.fval_opt = self.solver(
                    theta_init, obj, **self.solver_kwargs)

        # Note that fval_opt in EM and in LLY are not comparable
        elif method == 'EM':
            dist = np.inf
            theta_i = theta_init
            counter = 0
            max_G = -np.inf
            clock = 0

            while (dist > EM_threshold) and (counter < num_EM_iter):
                kf = Filter(self.ft, for_smoother=True, **self.ft_kwargs)
                kf.fit(theta_i, Yt, Xt)
                ks = Smoother()
                ks.fit(kf)
                obj = ks.G
                theta_opt, G_opt = self.solver(theta_i, obj, 
                        **self.solver_kwargs)
                dist = np.abs(max_G - G_opt)
                
                # Record minimal G and reset clock
                if max_G < G_opt:
                    max_G = G_opt
                    clock = 0

                # Giving increasingly more weight on theta_i
                alpha = 1 / (1 + counter * EM_stopping_rate)
                theta_i = alpha * theta_opt + (1 - alpha) * theta_i
                counter += 1
                clock += 1

                if clock > post_min_iter:
                    warnings.warn('Premature termination of EM')
                    break

            self.theta_opt = theta_i
            self.fval_opt = ks.G(theta_i)

        # Generate fitted smoother
        kf = Filter(self.ft, **self.ft_kwargs, for_smoother=True)
        kf.fit(self.theta_opt, Yt, Xt, init_state=init_state)
        ks = Smoother()
        ks.fit(kf)
        self.ks_fitted = ks


    def get_LLY(self, theta: np.ndarray, Yt: List[np.ndarray], 
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
        kf = Filter(self.ft, **self.ft_kwargs)
        kf.fit(theta, Yt, Xt)
        return kf.get_LL()


    def predict_t(self, df: pd.DataFrame, theta: np.ndarray=None, 
            is_xi: bool=True, xi_col: List[int]=None, 
            init_state: Dict=None, t_index: int=0) -> \
            Tuple[pd.DataFrame, Smoother]: 
        """
        Wrapper for BaseOpt.predict. It set the index to t_index. The 
        value of t_index must be between BaseOpt.ks_fitted.t_q and 
        BaseOpt.ks_fitted.T

        Parameters:
        ----------
        df : df to be predicted. Use np.nan for missing Yt
        theta : override theta_opt using user-supplied theta
        is_xi : whether output xi
        xi_col : index of xi to be included
        init_state : user-specified initial state values
        t_index : starting time index of df related to training set.
            t_index == -1 refers to the BaseOpt.ks_fitted.T

        Returns:
        ----------
        df_fitted : Contains filtered/smoothed y_t, xi_t, and P_t
        """
        if init_state is None:
            t = self.ks_fitted.T if t_index == -1 else t_index
            init_state = self.ks_fitted.get_filtered_state(t)
        
        df_fitted = self.predict(df, theta=theta, is_xi=is_xi, 
                xi_col=xi_col, init_state=init_state)
        
        return df_fitted


    def predict(self, df: pd.DataFrame, theta: np.ndarray=None, 
            is_xi: bool=True, xi_col: List[int]=None, 
            init_state: Dict=None, fmean_suffix: str='_filtered',
            fvar_suffix: str='_fvar', smean_suffix: str='_smoothed',
            svar_suffix: str='_svar') -> pd.DataFrame: 
        """
        Predict time series. df should contain both training and 
        test data. If Yt is not available for some or all test data,
        use np.nan as placeholders. Accept user-supplied theta as well

        Parameters:
        ----------
        df : df to be predicted. Use np.nan for missing Yt
        theta : override theta_opt using user-supplied theta
        is_xi : whether output xi
        xi_col : index of xi to be included
        init_state : user-specified initial state values
        fmean_suffix : suffix to filtered mean variable
        fvar_suffix : suffix to filtered var variable
        smean_suffix : suffix to smoothed mean variable
        svar_suffix : suffix to smoothed var variable

        Returns:
        ----------
        df_fitted : Contains filtered/smoothed y_t, xi_t, and P_t
        """
        # Generate system matrices for prediction
        Xt = df_to_list(df, self.x_col)
        Yt = df_to_list(df, self.y_col)
        
        # Generate filtered predictions
        kf = Filter(self.ft, for_smoother=True, **self.ft_kwargs)

        # Override theta_opt if theta is not None
        if theta is not None:
            kf.fit(theta, Yt, Xt, init_state=init_state)
        elif self.theta_opt is not None:
            kf.fit(self.theta_opt, Yt, Xt, init_state=init_state)
        else:
            raise ValueError('Model is not fitted')
        
        # Fit smoother
        ks = Smoother()
        ks.fit(kf)
        
        # Get filtered Yt
        y_col_filter = create_col(self.y_col, suffix=fmean_suffix)
        y_filter_var = create_col(self.y_col, suffix=fvar_suffix)
        Yt_filtered, Yt_P, xi_t, P_t = kf.get_filtered_val(
                is_xi=is_xi, xi_col=xi_col)
        Yt_P_diag = get_diag(Yt_P)
        df_Yt_filtered = list_to_df(Yt_filtered, y_col_filter)
        df_Yt_fvar = list_to_df(Yt_P_diag, y_filter_var)
            
        # Get smoothed Yt
        y_col_smoother = create_col(self.y_col, suffix=smean_suffix)
        y_smoother_var = create_col(self.y_col, suffix=svar_suffix)
        Yt_smoothed, Yt_S, xi_T, P_T = ks.get_smoothed_val(
                is_xi=is_xi, xi_col=xi_col)
        Yt_S_diag = get_diag(Yt_S)
        df_Yt_smoothed = list_to_df(Yt_smoothed, y_col_smoother)
        df_Yt_svar = list_to_df(Yt_S_diag, y_smoother_var)

        # Generate xi values if needed
        if is_xi:
            if xi_col is None:
                xi_col = list(range(kf.xi_length))

            xi_col_f = ['xi_{}_filtered'.format(i) for i in xi_col]
            P_col_f = ['P_{}_filtered'.format(i) for i in xi_col]
            xi_col_s = ['xi_{}_smoothed'.format(i) for i in xi_col]
            P_col_s = ['P_{}_smoothed'.format(i) for i in xi_col]

            df_xi_t = list_to_df(xi_t, xi_col_f)
            P_t_diag = get_diag(P_t)
            df_P_t = list_to_df(P_t_diag, P_col_f)

            P_T_diag = get_diag(P_T)
            df_xi_T = list_to_df(xi_T, xi_col_s)
            df_P_T = list_to_df(P_T_diag, P_col_s)
            df_fs = pd.concat([df_Yt_filtered, df_Yt_fvar,
                               df_Yt_smoothed, df_Yt_svar, df_xi_t,
                               df_P_t, df_xi_T, df_P_T], axis=1)

        else:
            df_fs = pd.concat([df_Yt_filtered, df_Yt_fvar, 
                    df_Yt_smoothed, df_Yt_svar], axis=1)

        df_fs.set_index(df.index, inplace=True)

        # Warning if new columns are in the original df.col
        if len(set(df_fs.columns).intersection(set(df.columns))) > 0:
            warnings.warn('Input df and predicted df contain ' + \
                    'overlapping column names')

        df_fitted = pd.concat([df, df_fs], axis=1)
        return df_fitted


    def simulated_data(self, input_theta: np.ndarray=None, 
            Xt: pd.DataFrame=None, T: int=None, init_state: Dict=None) \
            -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Calls utils.simulated_data

        Parameters:
        ----------
        input_theta : parameters of ft, if None, use self.theta_opt
        Xt : optional input deterministic values
        T : length of the dataframe
        init_state : user-specified initial state values

        Returns:
        ----------
        df : output dataframe
        y_col : column names of y_t
        xi_col : column names of xi_t
        """
        if input_theta is not None:
            theta_ = input_theta
        elif self.theta_opt is not None:
            theta_ = self.theta_opt
        else:
            raise ValueError('Model needs theta')
        
        if self.ft is None:
            raise ValueError('Model needs ft')

        df, y_col, xi_col = simulated_data(self.ft, theta_, Xt=Xt, 
                T=T, init_state=init_state, **self.ft_kwargs)
        return df, y_col, xi_col


class BaseConstantModel(BaseOpt):
    """
    Any BSTS model with constant system matrices may inherit this class.
    The child class should provide get_f function. It inherits from BaseOpt
    and has a customized get_f
    """

    def set_f(self, f: Callable, reset: bool=True, **ft_kwargs) -> None:
        """
        Mapping from theta to M. Provided by children classes.
        Must be the form of get_f(theta). If defined, it should
        return system matrix M

        Parameters:
        ----------
        f : theta -> M
        ft_kwargs : arguments for ft
        reset : if true, reset fitted values
        """
        ft_ = lambda theta, T, **ft_kwargs: \
                ft(theta, f, T, **ft_kwargs)
        super().set_f(ft_, reset=reset, **ft_kwargs)


