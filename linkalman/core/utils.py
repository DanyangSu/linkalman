from collections.abc import Sequence
import numpy as np
import pandas as pd
from scipy import linalg
from typing import List, Any, Callable, Dict, Tuple
from pandas.api.types import is_numeric_dtype
from numpy.random import multivariate_normal
from copy import deepcopy

__all__ = ['mask_nan', 'inv', 'df_to_list', 'list_to_df', 'create_col', 
        'noise', 'simulated_data', 'gen_PSD', 'ft', 'M_wrap']


def mask_nan(is_nan: np.ndarray, mat: np.ndarray, dim: str='both') -> np.ndarray:
    """
    Takes the list of NaN indices and mask the ros and columns 
    of a matrix with 0 if index has NaN value.

    Example:
    ----------
    mat_masked = mask_nan(np.array([False, True, False]), 
        np.array([[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]]))
    print(mat_masked)  # np.array([[1, 0, 3], [0, 0, 0], [7, 0, 9]])

    Parameters:
    ----------
    mask_nan : indicates if the row and column should be masked
    mat : matrix to be transformed

    Returns:
    ----------
    mat_masked : masked mat
    """
    if dim not in ['col', 'row', 'both']:
        raise ValueError('dim must be either "col", "row", or "both"')
    mat_masked = mat.copy()

    # Perform row operation if dim=='row' and 'both' and not a vector
    if dim != 'col' and mat_masked.shape[0] > 1: 
        mat_masked[is_nan] = 0

    # Perform column operation if dim=='col' or 'both' and not a vector
    if dim != 'row' and mat_masked.shape[1] > 1:
        mat_masked[:, is_nan] = 0
    return mat_masked


def inv(h_array: np.ndarray) -> np.ndarray:
    """
    Calculate pinvh of PSD array. Note pinvh performs poorly
    if input matrix is far from being Hermitian, so use pinv2
    instead in this case.

    Parameters:
    ----------
    h_array : input matrix, assume to be Hermitian
    
    Returns:
    ----------
    h_inv : pseudo inverse of h_array. 
    """
    if np.allclose(h_array, h_array.T):
        h_inv = linalg.pinvh(h_array)
    else:
        h_inv = linalg.pinv2(h_array)
    return h_inv


def df_to_list(df: pd.DataFrame) -> List[np.ndarray]:
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


def list_to_df(L: List[np.ndarray], col: List[str]) -> pd.DataFrame:
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


def simulated_data(Mt: Dict, Xt: pd.DataFrame=None, T: int=None) -> \
        Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Generate simulated data from a given HMM system. Xt and T
    cannot be both set to None.

    Parameters: 
    ----------
    Mt : system matrices
    Xt : input Xt. Optional and can be set to None
    T : length of the time series.

    Returns:
    ----------
    df : output dataframe that contains Xi_t, Y_t and X_t
    y_col : column names of y_t
    xi_col : column names of xi_t
    """
    xi_dim = Mt['xi_1_0'].shape[0]
    y_dim = Mt['Ht'][0].shape[0]
    x_dim = Mt['Dt'][0].shape[1]
    Y_t = []
    
    # Set Xt to Constant_M(np.zeros((1, 1))) if set as None
    if Xt is None:
        if T is None:
            raise ValueError('When Xt = None, T must be assigned')
        X_t = Constant_M(np.zeros((x_dim, 1)), T)
    else:
        T = Xt.shape[0]

    # Initialize Xi_t
    Xi_t = [Mt['xi_1_0'] + noise(xi_dim, Mt['P_1_0'])]

    # Iterate through time steps
    for t in range(T):
        # Generate Y_t
        y_t = Mt['Ht'][t].dot(Xi_t[t]) + Mt['Dt'][t].dot(X_t[t]) + \
                noise(y_dim, Mt['Rt'][t])
        Y_t.append(y_t)

        # Genereate Xi_t
        if t < T - 1:
            xi_t1 = Mt['Ft'][t].dot(Xi_t[t]) + Mt['Bt'][t].dot(X_t[t]) + \
                    noise(xi_dim, Mt['Qt'][t])
            Xi_t.append(xi_t1)

    # Generate df
    y_col = ['y_{}'.format(i) for i in range(y_dim)]
    xi_col = ['xi_{}'.format(i) for i in range(xi_dim)]
    df_Y = BaseEM._list_to_df(Y_t, y_col)
    df_Xi = BaseEM._list_to_df(Xi_t, xi_col)
    df = pd.concat([df_Xi, df_Y, Xt], axis=1)
    return df, y_col, xi_col


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


def ft(theta: List[float], f: Callable, T: int) -> Dict:
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


class M_wrap(Sequence):
    """
    Wraper of array lists. Improve efficiency by skipping 
    repeated calculation when m_list contains same arrays. 
    """

    def __init__(self, m_list: List[np.ndarray]) -> None:
        """
        Create placeholder for calculated matrix. 

        Parameters:
        ----------
        m_list : list of input arrays. Should be mostly constant
        """
        self.m = None
        self.m_pinvh = None
        self.L = None
        self.D = None
        self.L_I = None
        self.m_pdet = None
        self.m_list = m_list
        

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Returns indexed array of the wrapped list

        Parameters:
        ----------
        index : index of the wrapped list

        Returns:
        ----------
        self.m_list[index] : indexed array of the wrapped list
        """
        return self.m_list[index]


    def __setitem__(self, index: int, val: np.ndarray) -> None:
        """
        Set values of the wrapped list

        Parameters:
        ----------
        index : index of the wrapped list
        val : input array
        """
        self.m_list[index] = val 


    def __len__(self) -> int:
        """
        Required for a Sequence Object

        Returns:
        ----------
        len(self.m_list) : length of the wrapped list
        """
        return len(self.m_list)


    def _equal_M(self, index: int) ->bool:
        """
        Return true if self.m_list[index] == self.m. 
        If false, set self.m = self.m_list[index]

        Parameters: 
        ----------
        index : index of the wrapped list

        Returns:
        ----------
        Boolean that indicates whether we need to perform the operation
        """
        if np.array_equal(self.m, self.m_list[index]):
            return True
        else:
            self.m = self.m_list[index]
            return False
    

    def pinvh(self, index):
        """
        Return pseudo-inverse of self.m_list[index]

        Parameters:
        ----------
        index : index of the wrapped list

        Returns:
        self.m_pinvh : pesudo inverse 
        """
        if (not self._equal_M(index)) or self.m_pinvh is None:
            self.m_pinvh = inv(self.m)
        return self.m_pinvh
    

    def ldl(self, index):
        """
        Calculate L and D from LDL decomposition, and inverse of L

        Parameters:
        ----------
        index : index of the wrapped list

        Returns:
        self.L : L  of LDL
        self.D : D of LDL
        self.L_I : inverse of L
        """
        if (not self._equal_M(index)) or self.L is None:
            self.L, self.D, _ = linalg.ldl(self.m)
            self.L_I, _ = linalg.lapack.clapack.dtrtri(self.L, lower=True)
        return self.L, self.D, self.L_I


    def pdet(self, index):
        """
        Calculate pseudo-determinant. If zero matrix, determinant is 1
        Because we are using log, determinant of 1 is good.

        Parameters:
        ----------
        index : index of the wrapped list
        
        Returns:
        ----------
        self.m_pdet : pseudo-determinant
        """
        if (not self._equal_M(index)) or self.m_pdet is None:
            eig, _ = np.linalg.eigh(self.m)

            # If all eigenvalues are close to 0, np.product(np.array([])) returns 1
            self.m_pdet = np.product(eig[np.abs(eig)>1e-12])
        return self.m_pdet


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

