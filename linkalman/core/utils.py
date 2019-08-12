from collections.abc import Sequence
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.linalg import solve_discrete_lyapunov as lyap
from typing import List, Any, Callable, Dict, Tuple
from pandas.api.types import is_numeric_dtype
from numpy.random import multivariate_normal
from copy import deepcopy
import warnings

__all__ = ['mask_nan', 'inv', 'df_to_list', 'list_to_df', 'create_col',
        'noise', 'simulated_data', 'gen_PSD', 'ft', 'M_wrap', 
        'clean_matrix', 'get_ergodic', 'min_val', 'max_val', 'inf_val']


max_val = 1e10  # detect infinity
inf_val = 1e100  # value used to indicate infinity
min_val = 1e-8  # detect 0


def mask_nan(is_nan: np.ndarray, mat: np.ndarray, 
        dim: str='both', diag: float=0) -> np.ndarray:
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
    dim : whether zero out column or row or both
    diag : replace diagonal NaN with other values

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

    # Replace diagonal values
    if diag != 0:
        for i in range(len(is_nan)):
            if is_nan[i]:
                mat_masked[i][i] = diag

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


def simulated_data(Mt: Dict, Xt: pd.DataFrame=None, T: int=None,
        xi_1_0: np.ndarray=None, P_1_0:np.ndarray=None) -> \
        Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Generate simulated data from a given HMM system. Xt and T
    cannot be both set to None. If P_1_0 and xi_1_0  are 
    not provided, calculate ergodic values from system matrices

    Parameters: 
    ----------
    Mt : system matrices
    Xt : input Xt. Optional and can be set to None
    T : length of the time series
    xi_1_0 : initial mean array
    P_1_0 : initial covariance matrix

    Returns:
    ----------
    df : output dataframe that contains Xi_t, Y_t and X_t
    y_col : column names of y_t
    xi_col : column names of xi_t
    """
    xi_dim = Mt['Ft'][0].shape[0]
    y_dim = Mt['Ht'][0].shape[0]
    x_dim = Mt['Dt'][0].shape[1]
    
    # Set Xt to Constant_M(np.zeros((1, 1))) if set as None
    if Xt is None:
        if T is None:
            raise ValueError('When Xt = None, T must be assigned')
        X_t = Constant_M(np.zeros((x_dim, 1)), T)
    else:
        T = Xt.shape[0]
        X_t = df_to_list(Xt)

    # Create xi_1_0 and P_1_0
    if xi_1_0 is None:
        xi_1_0 = np.zeros([Mt['Ft'][0].shape[1], 1])
    if P_1_0 is None:
        P_1_0 = get_ergodic(Mt['Ft'][0], Mt['Qt'][0], Mt['Bt'][0])
    P_1_0[np.isnan(P_1_0)] = 1  # give an arbitrary value to diffuse priors
    Xi_t = [xi_1_0 + noise(xi_dim, P_1_0)]

    # Iterate through time steps
    Y_t = []
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
    df_Y = list_to_df(Y_t, y_col)
    df_Xi = list_to_df(Xi_t, xi_col)
    df = pd.concat([df_Xi, df_Y, Xt], axis=1)
    return df, y_col, xi_col


def gen_PSD(theta: List[float], dim: int) -> np.ndarray:
    """
    Generate covariance matrix from theta. Requirement:
    len(theta) = (dim**2 + dim) / 2

    Parameters:
    ----------
    theta : parameters used in generating lower triangle matrix
        Diagonal values are always non-negative
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
    
    #enforce non-negative diagonal values
    for i in range(dim):
        L[i][i] = np.exp(L[i][i])
    PSD = L.dot(L.T)
    return PSD


def get_ergodic(F: np.ndarray, Q: np.ndarray, B: np.ndarray=None) -> np.ndarray:
    """
    Calculate initial state covariance matrix, and identify 
    diffuse state. It effectively solves a Lyapuov equation

    Parameters:
    ----------
    F : state transition matrix
    Q : initial error covariance matrix
    B : regression matrix, if not 0, indicating diffuse priors

    Returns:
    ----------
    P_0 : the initial state covariance matrix, np.inf for diffuse state
    """
    Q_ = deepcopy(Q)
    dim = Q.shape[0]
    
    # Check F and Q
    if F.shape[0] != F.shape[1]:
        raise TypeError('F must be a square matrix')
    if Q.shape[0] != Q.shape[1]:
        raise TypeError('Q must be a square matrix')
    if F.shape[0] != Q.shape[0]:
        raise TypeError('Q and F must be of same size')

    is_diffuse = [False for i in range(dim)]
    if B is not None:

        # Check B
        if B.shape[0] != F.shape[0]:
            raise TypeError('B has wrong sizes')
        
        for i in range(dim):

            # If state rely on x_t, it is a diffuse state
            if np.count_nonzero(B[i]) > 0:
                is_diffuse[i] = True
        
        # Modify Q_ to reflect diffuse states
        Q_ = mask_nan(is_diffuse, Q_, diag=inf_val)
        
    # Calculate raw P_0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P_0 = lyap(F, Q_, 'bilinear')

    # Clean up P_0
    is_diffuse = [False for i in range(dim)]
    for i in range(dim):
        if abs(P_0[i][i]) > max_val:
            is_diffuse[i] = True
    P_0 = mask_nan(is_diffuse, P_0, diag=0)
    
    # Enforce PSD
    P_0_PSD = get_nearest_PSD(P_0)

    # Add nan to diffuse diagonal values
    P_0_PSD += np.diag(np.array([np.nan if i else 0 for i in is_diffuse]))
    return P_0_PSD


def get_nearest_PSD(mat: np.ndarray) -> np.ndarray:
    """
    Get the nearest PSD matrix of an input matrix

    Parameters:
    ----------
    mat : input matrix to be processed

    Returns:
    ----------
    PSD_mat : the nearest PSD matrix
    """
    _, S, VT = linalg.svd(mat)
    Sigma = VT.T.dot(np.diag(S)).dot(VT)
    PSD_mat = (Sigma + Sigma.T + mat + mat.T) / 4
    return PSD_mat


def clean_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Enforce 0 and inf on matrix values.
    Convert type to flot if int

    Parameters:
    ----------
    mat : input matrix to be cleaned

    Returns:
    ----------
    cleaned_mat : processed matrix
    """
    cleaned_mat = deepcopy(mat).astype(float, copy=True)
    cleaned_mat[np.abs(cleaned_mat) < min_val] = 0
    cleaned_mat[np.abs(cleaned_mat) > max_val] = inf_val

    return cleaned_mat


def ft(theta: List[float], f: Callable, T: int,
        xi_1_0: np.ndarray=None, P_1_0: np.ndarray=None) -> Dict:
    """
    Duplicate arrays in M = f(theta) and generate list of Mt
    Output of f(theta) must contain all the required keys.

    Parameters:
    ----------
    theta : input of f(theta). Underlying paramters to be optimized
    f : obtained from get_f. Mapping theta to M
    T : length of Mt. "Duplicate" M for T times
    xi_1_0: initial state mean
    P_1_0: initial state cov

    Returns:
    ----------
    Mt : system matrices for a BSTS. Should contain
        all the required keywords. 
    """ 
    M = f(theta)

    # If a value is close to 0 or inf, set them to 0 or inf
    # If an array is 1D, convert it to 2D
    for key in M.keys():
        if M[key].ndim < 2:
            M[key] = M[key].reshape(1, -1)
        M[key] = clean_matrix(M[key])
    
    # Check validity of M
    required_keys = set(['F', 'H', 'Q', 'R'])
    M_keys = set(M.keys())
    if len(required_keys - M_keys) > 0:
        raise ValueError('f does not have required outputs: {}'.
            format(required_keys - M_keys))

    # Check dimensions of M
    for key in M_keys:
        if len(M[key].shape) < 2:
            raise TypeError('System matrices must be 2D')
    
    # Generate ft for required keys
    Ft = Constant_M(M['F'], T)
    Ht = Constant_M(M['H'], T)
    Qt = Constant_M(M['Q'], T)
    Rt = Constant_M(M['R'], T)

    # Set Bt if Bt is not Given
    if 'B' not in M_keys:
        dim_xi = M['F'].shape[0]
        M.update({'B': np.zeros((dim_xi, 1))})
    if 'D' not in M_keys:
        dim_y = M['H'].shape[0]
        M.update({'D': np.zeros((dim_y, 1))})

    # Get Bt and Dt for ft
    Bt = Constant_M(M['B'], T)
    Dt = Constant_M(M['D'], T)

    # Initialization
    if xi_1_0 is None:
        xi_1_0 = np.zeros([M['F'].shape[0],1])
    if P_1_0 is None: 
        P_1_0 = get_ergodic(M['F'], M['Q'], M['B']) 

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
        raise TypeError('Q must be a square matrix')
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
        raise TypeError('Q must be a square matrix')
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

