import numpy as np
import pandas as pd
from typing import List, Any, Callable
from pandas.api.types import is_numeric_dtype
from collections.abc import Sequence
from ..core import EM
from ..core import Smoother

__all__ = ['BaseEM', 'BaseConstantModel', 'F_theta', 'create_col', 'Constant_M']

def F_theta(theta: List[float], f: Callable, T: int) -> dict:
    """
    Duplicate arrays in M = f(theta) and generate list of Mt
    Output of f(theta) must contain all the required keys.
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

    return {'Ft': Ft, 
            'Bt': Bt, 
            'Ht': Ht, 
            'Dt': Dt, 
            'Qt': Qt, 
            'Rt': Rt, 
            'xi_1_0': xi_1_0, 
            'P_1_0': P_1_0}

def create_col(col: List[str], suffix: str='_pred') -> List[str]:
    """
    Create column names for filter predictions. Default suffix is '_pred'

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

        self.M = M
        self.Mt = {}
        self.length = length

    def __setitem__(self, index: int, val: np.ndarray) -> None:
        """
        If val differ from self.M, store val and index
        in self.Mt. 

        Parameters:
        ----------
        index : index number of the list between 0 and self.T
        val : value to replace M at index. 
        """
        # Only update if val differs from self.M
        if not np.array_equal(self.M, val):
            self.Mt.update({index: val}) 

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Search through self.Mt dictionary, return 
        self.Mt[index] if self.Mt[index] is set, 
        else returns default self.M

        Parameters:
        ----------
        index : index number of the list

        Returns:
        ----------
        Mt_index : Constant_M[index]
        """
        Mt_index = self.Mt.get(index, self.M)
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

    def predict(self, df: pd.DataFrame, Ft: Callable=None) -> Smoother: 
        """
        Predict time series. df_extended should contain 
        both training and test data.

        Parameters:
        ----------
        df : df to be predicted. Use np.nan for missing Yt
        Ft : must be a function of theta, update self.Ft if not
            None. Ft should be consistent with self.Ft for t <= T

        Returns:
        ----------
        ks : Contains filtered/smoothed y_t, xi_t, and P_t
        """
        # Generate system matrices for prediction
        self.Ft = Ft if Ft != None else self.Ft 
        Mt = self.Ft(self.theta_opt)
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
    
class BaseConstantModel(object):

    def __init__(self) -> None:
        raise NotImplementedError
    
    def get_f(self) -> None:
        raise NotImplementedError

    def fit(self, df: pd.DataFrame, x_col: List[str], y_col: List[str], 
            **kwargs) -> None:
        """
        Fit a time-series model. For specification design, refer to theory.pdf
        """
        x_dim = len(x_col)
        y_dim = len(y_col)
        T = df.shape[0]

        # Create f
        kwargs.update({'x_dim': x_dim, 'y_dim': y_dim})
        f = lambda theta: self.get_f(theta, **kwargs)

        # Fit model using ConstantEM
        ConstEM = ConstantEM(f, T)
        ConstEM.fit(df, theta, x_col, y_col)
        self.mod = ConstEM

    def predict(self, df):
        """
        Predict filtered yt
        """
        return self.mod.predict(df)

    @staticmethod
    def gen_PSD(theta, dim):
        """
        Generate covariance matrix from theta. Requirement:
        len(theta) = (dim**2 + dim) / 2
        """
        L = np.zeros([dim, dim])

        # Fill diagonal values
        for i in range(dim):
            L[i][i] = np.exp(theta[i])

        # Fill lower off-diagonal values
        theta_off = theta[dim:]
        idx = np.tril_indices(dim, k=-1)
        L[idx] = theta_off
        return L.dot(L.T)
