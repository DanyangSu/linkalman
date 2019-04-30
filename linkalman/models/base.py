import numpy as np
import pandas as pd
from typing import List, Any, Callable
from pandas.api.types import is_numeric_dtype
from collections.abc import Sequence
from ..core import EM
from ..core import Smoother
from copy import deepcopy
from numpy.random import multivariate_normal

__all__ = ['BaseEM', 'BaseConstantModel', 'Constant_M']

class Constant_M(Sequence):
    """
    If the sequence of system matrix is mostly constant over time 
    (with the exception of occassional deviation), using Constant_M 
    saves memory space. It mimics the behavior of a regular list but 
    use one single baseline M and stores any deviation.

    Example:
    ----------
    M = np.array([[5, 3],[3, 4]])  # baseline matrix
    T = 100  # intended length of the list
    
    # Mt has similar behavior as [copy.deepcopy(M) for _ in range(T)]
    Mt = Constant_M(M, T)  
    
    # Modify Mt[i] for i doesn't affect Mt[j] for j!=i
    Mt[2] = np.array([[5, 2], [2, 5]])
    print(Mt[2])  # np.array([[5, 2], [2, 5]])
    print(Mt[1])  # np.array([[5, 3], [3, 4]])
    """

    def __init__(self, M: np.ndarray, length: int) -> None:
        """
        Generate a list of M.

        Parameters:
        ----------
        M : input system matrix 
        length : length of the list
        """
        # Raise if M is not np.array type
        if not isinstance(M, np.ndarray):
            raise TypeError('M must be a numpy array')

        self.M = deepcopy(M)
        self._M = deepcopy(M)  # benchmark M
        self.index = None
        self.Mt = {}
        self.length = length

    @property
    def is_M_modified(self) -> bool:
        """
        Determine whether self.M is modified or not

        Returns:
        ----------
        Boolean that returns True if self.M is different from self._M
        """
        return not np.array_equal(self.M, self._M)

    def finger_print(self) -> None:
        """
        Update self.Mt and restore self.M if self.M is modified. This 
        happens when elements of a numpy array is modified and 
        __setitem__ is not invoked. The modified value will be kept 
        in self.M. This function will record the change before restoring
        self.M to its original state.
        """
        self.Mt.update({self.index: deepcopy(self.M)})
        self.M = deepcopy(self._M)


    def __setitem__(self, index: int, val: np.ndarray) -> None:
        """
        If val differ from self.M, store val and index
        in self.Mt. 

        Parameters:
        ----------
        index : index number of the list between 0 and self.T
        val : value to replace M at index. 
        """
        # Check if already partially updated. If it is, update self.Mt
        if self.index == index and self.is_M_modified:
            self.finger_print()

        # Only update if val differs from current value or self._M
        if not np.array_equal(self.Mt.get(index, self._M), val):
            self.Mt.update({index: deepcopy(val)}) 

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Search through self.Mt dictionary, return 
        self.Mt[index] if self.Mt[index] is set, 
        else returns default self.M. In addition,
        if a modification on self.M is detected, 
        update self.Mt and restore self.M

        Parameters:
        ----------
        index : index number of the list

        Returns:
        ----------
        Mt_index : Constant_M[index]
        """
        # Restore self.M and update self.Mt
        if self.is_M_modified:
            self.finger_print()

        Mt_index = self.Mt.get(index, self.M)
        self.index = index
        return Mt_index

    def __len__(self) -> int:
        """
        Set length of the list, required for an object inherited 
        from Sequence.

        Returns:
        ----------
        self.length : length of the list
        """
        return self.length

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
            x_col: List[str], y_col: List[str]) -> None:
        """
        Fit the model using EM algorithm. Produces optimal 
        theta. 

        Parameters:
        ----------
        df : input data to be fitted
        theta_init : initial theta. self.Ft(theta_init) produce 
            Mt for first iteration of EM algorithm.
        x_col : list of columns in df that belong to Xt
        y_col : list of columns in df that belong to Yt
        """
        # Initialize
        em = EM(self.Ft)
        self.x_col = x_col
        self.y_col = y_col
        # Convert dataframe to lists
        Xt = self._df_to_list(df[x_col])
        Yt = self._df_to_list(df[y_col])

        # Run EM solver
        self.theta_opt = em.fit(theta_init, Xt, Yt)

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
        
    @staticmethod
    def _df_to_list(df: pd.DataFrame) -> List[np.ndarray]:
        """
        Convert pandas dataframe to list of arrays.
        
        Parameters:
        ----------
        df : must be numeric

        Returns:
        ----------
        L : len(L) == df.shape[0], L[0].shape[0] == df.shape[1]
        """
        # Raise exception if df is not a dataframe
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a dataframe')

        # Check datatypes, must be numeric
        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                raise TypeError('Input dataframe must be numeric')

        # Convert df to list row-wise
        L = []
        for i in range(df.shape[0]):
            L.append(np.array([df.loc[i,:]]).T)
        return L
    
    @staticmethod
    def _list_to_df(L: List[np.ndarray], col: List[str]) -> pd.DataFrame:
        """
        Convert list of arrays to a dataframe. Reverse operation of _df_to_list.

        Parameters:
        ----------
        L : len(L) == df.shape[0], L[0].shape[0] == df.shape[1]
        col : list of column names. len(col) must equal to L[0].shape[0]

        Returns:
        ----------
        df: output dataframe
        """
        if not isinstance(col, list):
            raise TypeError('col must be a list of strings')
        df_val = np.concatenate([i.T for i in L])
        df = pd.DataFrame(data=df_val, columns=col)
        return df
    
    @staticmethod
    def create_col(col: List[str], suffix: str='_pred') -> List[str]:
        """
        Create column names for predictions. Default suffix is '_pred'

        Parameters:
        ----------
        col : column list of a dataframe
        suffix : string to be appended to each column name in col

        Returns:
        ----------
        col_new : modified column names
        """
        col_new = []
        for i in col:
            col_new.append(i + suffix)
        return col_new

    @staticmethod
    def noise(y_dim: int, Sigma: np.ndarray) -> np.ndarray:
        """
        Generate n-by-1 Gaussian noise 

        Parameters: 
        ----------
        y_dim : dimension of yt
        Sigma : cov matrix of Gaussian noise

        Returns:
        ----------
        epsilon : noise of the system
        """
        epsilon = multivariate_normal(np.zeros(y_dim), Sigma).reshape(-1, 1)
        return epsilon

    @staticmethod
    def simulated_data(Xt: pd.DataFrame, Mt: np.ndarray) -> pd.DataFrame:
        """
        Generate simulated data from a given HMM system. 

        Parameters: 
        ----------
        Xt : input Xt. If one does not want Xt, can use 0and set Bt Dt as 0
        Mt : system matrices

        Returns:
        ----------
        df : output dataframe that contains Xi_t, Y_t and X_t
        """
        T = Xt.shape[0]
        xi_dim = Mt['xi_1_0'].shape[0]
        y_dim = Mt['Ht'][0].shape[0]
        x_dim = Xt.shape[1]
        Y_t = []
        X_t = BaseEM._df_to_list(Xt)
        Xi_t = [Mt['xi_1_0'] + BaseEM.noise(xi_dim, Mt['P_1_0'])]

        # Iterate through time steps
        for t in range(T):
            # Generate Y_t
            y_t = Mt['Ht'][t].dot(Xi_t[t]) + Mt['Dt'][t].dot(X_t[t]) + \
                    BaseEM.noise(y_dim, Mt['Rt'][t])
            Y_t.append(y_t)

            # Genereate Xi_t
            if t < T - 1:
                xi_t1 = Mt['Ft'][t].dot(Xi_t[t]) + Mt['Bt'][t].dot(X_t[t]) + \
                        BaseEM.noise(xi_dim, Mt['Qt'][t])
                Xi_t.append(xi_t1)

        # Generate df
        y_col = ['y_{}'.format(i) for i in range(y_dim)]
        xi_col = ['xi_{}'.format(i) for i in range(xi_dim)]
        df_Y = BaseEM._list_to_df(Y_t, y_col)
        df_Xi = BaseEM._list_to_df(Xi_t, xi_col)
        df = pd.concat([df_Xi, df_Y, Xt], axis=1)
        return df

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

    def fit(self, df: pd.DataFrame, x_col: List[str], y_col: List[str], 
            **kwargs: Any) -> None:
        """
        Invoke BaseEM.fit to fit the data.

        Parameters:
        ----------
        df : data to be fitted
        x_col : columns in df that belong to Xt
        y_col : columns in df that belong to Yt
        kwargs : kwargs for get_f 
        """
        # Raise exception if x_col or y_col is not list
        if not isinstance(x_col, list):
            raise TypeError('x_col must be a list.')
        if not isinstance(y_col, list):
            raise TypeError('y_col must be a list.')

        # Collect dimensions of Xt and Yt
        x_dim = len(x_col)
        y_dim = len(y_col)
        T = df.shape[0]

        # Create F
        kwargs.update({'x_dim': x_dim, 'y_dim': y_dim})
        self.f = lambda theta: self.get_f(theta, **kwargs)
        F = lambda theta: self.F_theta(theta, self.f, T)

        # Fit model using ConstantEM
        ConstEM = BaseEM(F)
        ConstEM.fit(df, theta, x_col, y_col)
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

    @staticmethod
    def gen_PSD(theta: List[float], dim: int) -> np.ndarray:
        """
        Generate covariance matrix from theta. Requirement:
        len(theta) = (dim**2 + dim) / 2

        Parameters:
        ----------
        theta : parameters used in generating lower triangle matrix
        dim : dimension of the matrix. 

        Returns:
        PSD : PSD matrix
        """
        # Raise exception if theta has wrong size
        if len(theta) != (dim ** 2 + dim)/2:
            raise ValueError('theta has wrong length')

        L = np.zeros([dim, dim])
        idx = np.tril_indices(dim, k=0)
        L[idx] = theta
        PSD = L.dot(L.T)
        return PSD
    
    @staticmethod
    def F_theta(theta: List[float], f: Callable, T: int) -> dict:
        """
        Duplicate arrays in M = f(theta) and generate list of Mt
        Output of f(theta) must contain all the required keys.

        Parameters:
        ----------
        theta : input of f(theta). Underlying paramters to be optimized
        f : obtained from get_f. Mapping theta to M
        T : length of Mt. "Duplicate" M for T times.

        Returns:
        ----------
        Mt : system matrices to feed into the EM algorithm. Should contain
            all the required keywords. 
        """ 
        M = f(theta)

        # Check validity of M
        required_keys = set(['F', 'B', 'H', 'D', 'Q', 'R', 'xi_1_0', 'P_1_0'])
        if set(M.keys()) != required_keys:
            raise ValueError('f does not have right outputs')

        Ft = Constant_M(M['F'], T)
        Bt = Constant_M(M['B'], T)
        Ht = Constant_M(M['H'], T)
        Dt = Constant_M(M['D'], T)
        Qt = Constant_M(M['Q'], T)
        Rt = Constant_M(M['R'], T)
        xi_1_0 = M['xi_1_0']
        P_1_0 = M['P_1_0']

        # Raise exception if xi_1_0 or P_1_0 is not numpy arrays
        if not isinstance(xi_1_0, np.ndarray):
            raise TypeError('xi_1_0 must be a numpy array')
        if not isinstance(P_1_0, np.ndarray):
            raise TypeError('P_1_0 must be a numpy array')

        Mt = {'Ft': Ft, 
                'Bt': Bt, 
                'Ht': Ht, 
                'Dt': Dt, 
                'Qt': Qt, 
                'Rt': Rt, 
                'xi_1_0': xi_1_0, 
                'P_1_0': P_1_0}
        return Mt
