from collections.abc import Sequence
import numpy as np
import pandas as pd
import networkx as nx
from scipy import linalg
from scipy.linalg import solve_discrete_lyapunov as lyap
from typing import List, Any, Callable, Dict, Tuple
from pandas.api.types import is_numeric_dtype
from numpy.random import multivariate_normal
from copy import deepcopy
import warnings
import inspect
from deprecated import deprecated
warnings.simplefilter('default')

__all__ = ['mask_nan', 'inv', 'df_to_tensor', 'tensor_to_df', 'create_col', 'get_diag',
        'noise', 'simulated_data', 'gen_PSD', 'ft', 'M_wrap', 'LL_correct', 
        'clean_matrix', 'get_ergodic', 'min_val', 'max_val', 'inf_val', 'pdet',
        'permute', 'revert_permute', 'partition_index', 'get_init_mat', 
        'check_consistence', 'gen_Xt', 'preallocate', 'get_explosive_diffuse',
        'get_nearest_PSD', 'Constant_M', 'Constant_M_simple', 'Constant_M_complex',
        'validate_wrapper', 'get_reset', 'build_tensor']


max_val = 1e6  # Smaller value identifies diffuse better
inf_val = 1e50  # value used to indicate infinity
min_val = 1e-7  # detect 0
    

def mask_nan(is_nan: np.ndarray, mat: np.ndarray, 
        dim: str='both', diag: float=0) -> np.ndarray:
    """
    Takes an array of NaN indices and mask the rows and columns 
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


def get_diag(arrays: np.ndarray) -> np.ndarray:
    """
    Convert a tensor of square arrays into a tensor of 1D arrays with 
    diagonal values only

    Parameters:
    ----------
    arrays : tensor of square arrays

    Returns:
    ----------
    diag_arrays : tensor of N x 1 arrays with diagonal values only
    """
    diag_arrays = preallocate(arrays.shape[0], arrays.shape[1], 1)
    for i in range(arrays.shape[0]):
        diag_arrays[i] = np.diag(arrays[i]).reshape(-1, 1)
    
    return diag_arrays
    

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


def get_explosive_diffuse(F: np.ndarray) -> np.ndarray:
    """
    If contains explosive roots, find strongly
    connected components. Check eigenvalues for
    each component submatrix, set large number to
    the ones with large component and run through 
    Lyapunov solver again to identify all the explosive
    roots.
    
    Parameters:
    ----------
    F : input state transition matrix
    
    Returns:
    ----------
    is_diffuse : indicator of diffuse states
    """
    # Find edges
    nonzeros = np.nonzero(F)
    index_ = list(zip(nonzeros[1], nonzeros[0]))
    
    # Calculate strongly connected components (scc)
    DG = nx.DiGraph()
    DG.add_edges_from(index_)
    scc = list(nx.strongly_connected_components(DG))
    F_ = F.copy()
    
    # Loop through each component for explosive roots
    for comp_ in scc:
        l_comp = list(comp_)
        sub_F = F_[np.ix_(l_comp, l_comp)]
        eig_ = linalg.eigvals(sub_F)
        if np.any(np.abs(eig_) > 1 + min_val):
            F_[l_comp, l_comp] = inf_val
    
    # Get diffuse states 
    is_diffuse = np.diag(F_) > max_val
    return is_diffuse


def gen_Xt(Xt: np.ndarray=None, B: np.ndarray=None, 
        T: int=None) -> np.ndarray:
    """
    Generate a list of zero arrays if Xt is None

    Parameters:
    ----------
    Xt : input Xt
    B : provide dimension information for generating dummy Xt
    T : provide length of the output list Xt

    Returns:
    ----------
    Xt_ : output Xt, if input is None, fill with zeros
    """
    if B is None or T is None:
        raise ValueError('B and T must not be None')
    Xt_ = np.zeros((T, B.shape[1], 1))
    return Xt_


def validate_wrapper(wrapper: Any) -> Any:
    """
    Validate whether a wrapper contains required methods. If
    wrapper == None, return M_wrap as the default wrapper

    Parameters:
    ----------
    wrapper : wrapper of list of matrices. It should include 
        pinvh, pdet, and ldl. Users may also modify their wrapper
        to avoid unnecessary calculations. 

    Returns:
    ----------
    output wrapper : validated wrapper object
    """
    if wrapper is None:
        return M_wrap
    else:
        # Check required inputs
        arg_list = set(inspect.getfullargspec(wrapper.__init__).args)
        if len(set(['m_tensor', 'reset']).difference(arg_list)) > 0:
            raise AttributeError("""The wrapper object must contain""" + \
                    """ 'm_tensor' and 'reset'. """)
        # Check required functions
        required_list = ['pinvh', 'pdet', 'ldl']
        methods = inspect.getmembers(wrapper, 
                predicate=inspect.isroutine)
        counter = 0
        for i in methods:
            if i[0] in required_list:
                counter += 1
        if counter < len(required_list):
            raise AttributeError('The wrapper does not contain ' + \
                    'all required methods.')
        return wrapper


def df_to_tensor(df: pd.DataFrame, col_list: List[str]=None) \
        -> np.ndarray:
    """
    Convert pandas dataframe to list of arrays.
    
    Parameters:
    ----------
    df : must be numeric
    col_list : list of columns to be converted

    Returns:
    ----------
    L : tensor of shape (T * len(col_name) * 1)
    """
    # If col_list is None, return None. None is treated by
    # downstream functions as an indicator.
    if col_list is None:
        return None
    else:

        # Raise exception if df is not a dataframe
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a dataframe')

        # Check data types, must be numeric
        for col in col_list:
            if not is_numeric_dtype(df[col]):
                raise TypeError('Input dataframe must be numeric')

        # Convert df to tnesor row-wise
        L = df[col_list].values.reshape((df.shape[0], len(col_list), 1))
        return L


def tensor_to_df(L: np.ndarray, col: List[str]) -> pd.DataFrame:
    """
    Convert list of arrays to a dataframe. Reverse operation of df_to_tensor.

    Parameters:
    ----------
    L : shape (m * n * 1)
    col : list of column names. len(col) = n

    Returns:
    ----------
    df: output dataframe shape (m * n)
    """
    if not isinstance(L, np.ndarray):
        raise TypeError('col must be an ndarray')
    if len(L.shape) < 3:
        raise TypeError('Input array must be a tensor')
    if len(col) != L.shape[1]:
        raise TypeError('Column length does not match input array')
    df = pd.DataFrame(L.reshape(L.shape[0], L.shape[1]), columns=col)
    return df


def build_tensor(arr: np.ndarray, T: int) -> np.ndarray:
    """
    Create tensor from a 2d array

    Parameters:
    ----------
    arr : input matrix
    T : number of times to replicate along the first dimension

    Returns:
    ----------
    tensor : output 3d tensor
    """
    if len(arr.shape) < 2:
        raise TypeError('Input must be 2-d arrays')

    return np.repeat(arr[np.newaxis, :, :], T, axis=0)


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


def preallocate(dim1: int, *dimn: int, default_val: float = 0, 
        arr_type: str='float') -> np.ndarray:
    """
    Preallocate a tensor placeholder

    Parameters:
    ----------
    dim1 : length of tensor
    dimn : dimension of the matrix
    default_val : default value of array
    arr_type : if 'float' then np.float64, if 'int' then int

    Returns:
    ----------
    allocated_tensor : pre-allocated tensor
    """
    if arr_type == 'float':
        array_type = np.float64
        typed_default_val = float(default_val)
    elif arr_type == 'int':
        array_type = int
        typed_default_val = int(default_val)
    elif arr_type == 'bool':
        array_type = bool
        typed_default_val = bool(default_val)
    else:
        raise TypeError("array_type must be 'float', 'int' or 'bool'")
    allocated_tensor = typed_default_val * np.ones((dim1, *dimn), 
            dtype=array_type, order='F')
    return allocated_tensor


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


def check_consistence(Mt: Dict, y_t: np.ndarray, x_t: np.ndarray, 
        init_state: Dict=None, is_checked: bool=False) -> None:
    """
    Check consistence of matrix dimensions. Ensure
    all matrix operations are properly done. The
    assumption is the shape of a matrix remains 
    constant over time. Also assume Mt contains all 
    the required keys. 

    Parameters:
    ----------
    Mt : Dict of system matrices
    y_t : measurement vector
    x_t : regressor vector
    init_state : user-specified initial state values
    is_checked : if True, skip the checking step
    """
    if not is_checked:
        # Check whether Mt contains all elements
        keys = set(['Ft', 'Bt', 'Qt', 'Ht', 'Dt', 'Rt', 
                'xi_1_0', 'P_1_0'])
        M_keys = set(Mt.keys())

        if not keys.issubset(M_keys):
            diff_keys = keys.difference(M_keys)
            raise ValueError('Mt does not contain all required keys. ' + \
                    'The missing keys are {}'.format(list(diff_keys)))

        # Collect dimensional information
        dim = {}
        dim.update({'Ft': Mt['Ft'][0].shape})
        dim.update({'Bt': Mt['Bt'][0].shape})
        dim.update({'Ht': Mt['Ht'][0].shape})
        dim.update({'Dt': Mt['Dt'][0].shape}) 
        dim.update({'Qt': Mt['Qt'][0].shape}) 
        dim.update({'Rt': Mt['Rt'][0].shape})
        dim.update({'xi_t': Mt['xi_1_0'].shape})
        dim.update({'y_t': y_t.shape}) 
        dim.update({'x_t': x_t.shape})
        
        # Check whether dimension is 2-D
        for m_name in dim.keys():
            if len(dim[m_name]) != 2:
                raise ValueError('{} has the wrong dimensions'.format(m_name))

        # Check Ft and xi_t
        if (dim['Ft'][1] != dim['Ft'][0]) or (dim['Ft'][1] != dim['xi_t'][0]):
            raise ValueError('Ft and xi_t do not match in dimensions')

        # Check Ht and xi_t
        if dim['Ht'][1] != dim['xi_t'][0]:
            raise ValueError('Ht and xi_t do not match in dimensions')

        # Check Ht and y_t
        if dim['Ht'][0] != dim['y_t'][0]:
            raise ValueError('Ht and y_t do not match in dimensions')

        # Check Bt and xi_t
        if dim['Bt'][0] != dim['xi_t'][0]:
            raise ValueError('Bt and xi_t do not match in dimensions')

        # Check Bt and x_t
        if dim['Bt'][1] != dim['x_t'][0]:
            raise ValueError('Bt and x_t do not match in dimensions')

        # Check Dt and y_t
        if dim['Dt'][0] != dim['y_t'][0]:
            raise ValueError('Dt and y_t do not match in dimensions')

        # Check Dt and x_t
        if dim['Dt'][1] != dim['x_t'][0]:
            raise ValueError('Dt and x_t do not match in dimensions')

        # Check Qt and xi_t
        if (dim['Qt'][1] != dim['Qt'][0]) or (dim['Qt'][1] != dim['xi_t'][0]):
            raise ValueError('Qt and xi_t do not match in dimensions')

        # Check Rt and y_t
        if (dim['Rt'][1] != dim['Rt'][0]) or (dim['Rt'][1] != dim['y_t'][0]):
            raise ValueError('Rt and y_t do not match in dimensions')

        # Check if y_t is a vector
        if dim['y_t'][1] != 1:
            raise ValueError('y_t must be a vector')

        # Check if xi_t is a vector
        if dim['xi_t'][1] != 1:
            raise ValueError('xi_t must be a vector')

        # Check if x_t is a vector
        if dim['x_t'][1] != 1:
            raise ValueError('x_t must be a vector')

        # If init_state is provided, check consistency
        if init_state is not None:
            check_name = set(['xi_t','P_star_t', 'P_inf_t']).intersection(
                    init_state.keys())
            for name in check_name:

                # Check number of dimensions
                if len(init_state[name].shape) != 2:
                    raise ValueError('User-specified {} '.format(name) + \
                            'does not have 2 dimensions')
                
                # Check if match sizes
                if name == 'xi_t':
                    dim_check = dim['xi_t']
                else:
                    dim_check = dim['Qt']
                if init_state[name].shape != dim_check:
                    raise ValueError('User-specified {} has'.format(name) + \
                            ' wrong dimensions')

            if init_state.get('q', 0) < 0:
                raise ValueError('User-specified q must be non-negative')


def simulated_data(Ft: Callable, theta: np.ndarray, 
        Xt: pd.DataFrame=None, T: int=None, init_state: Dict=None, 
        const_M_type: str='simple', **kwargs) \
        -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Generate simulated data from a given BSTS system. Xt and T
    cannot be both set to None. If P_1_0 and xi_1_0  are 
    not provided, calculate ergodic values from system matrices

    Parameters: 
    ----------
    Ft : ft(theta, T) that returns [Mt]_{1,...,T}
    theta : argument of ft
    Xt : input Xt. Optional and can be set to None
    T : length of the time series
    init_state : user-specified initial state values
    const_M_type : type of constant matrix generator
    kwargs : kwargs for Ft

    Returns:
    ----------
    df : output dataframe that contains Xi_t, Y_t and X_t
    y_col : column names of y_t
    xi_col : column names of xi_t
    """
    # Get dimensional information
    M_ = Ft(theta, 1, **kwargs)
    xi_dim = M_['Ft'][0].shape[0]
    y_dim = M_['Ht'][0].shape[0]
    x_dim = M_['Dt'][0].shape[1]
    
    # Set Xt to Constant_M(np.zeros((1, 1))) if set as None
    if Xt is None:
        if T is None:
            raise ValueError('When Xt = None, T must be assigned')
        X_t = preallocate(T, x_dim, 1)
    else:
        T = Xt.shape[0]
        X_t = df_to_tensor(Xt, list(Xt.columns))
    
    # Check consistence
    x_dim = X_t[0].shape[0]
    y_sample = np.ones([y_dim, 1])
    x_sample = np.ones([x_dim, 1])
    check_consistence(M_, y_sample, x_sample, init_state=init_state)
    
    Mt = Ft(theta, T, **kwargs)
    
    # Generate initial values
    xi_1_0 = Mt['xi_1_0'].copy()
    P_1_0 = Mt['P_1_0'].copy()

    # Override if init_state is provided
    if init_state is not None:
        xi_1_0 = init_state.get('xi_t', xi_1_0)
        P_1_0 = init_state.get('P_star_t', P_1_0)

    P_1_0[np.isnan(P_1_0)] = 1  # give an arbitrary value to diffuse priors
    Xi_t = preallocate(T, xi_1_0.shape[0], 1)
    Xi_t[0] = xi_1_0 + noise(xi_dim, P_1_0)

    # Iterate through time steps
    Y_t = preallocate(T, y_dim, 1)
    for t in range(T):
        # Generate y_t
        Y_t[t] = Mt['Ht'][t].dot(Xi_t[t]) + Mt['Dt'][t].dot(X_t[t]) + \
                noise(y_dim, Mt['Rt'][t])

        # Generate xi_t1
        if t < T - 1:
            Xi_t[t+1] = Mt['Ft'][t].dot(Xi_t[t]) + Mt['Bt'][t].dot(X_t[t]) + \
                    noise(xi_dim, Mt['Qt'][t])

    # Generate df
    y_col = ['y_{}'.format(i) for i in range(y_dim)]
    xi_col = ['xi_{}'.format(i) for i in range(xi_dim)]
    df_Y = tensor_to_df(Y_t, y_col)
    df_Xi = tensor_to_df(Xi_t, xi_col)
    df = pd.concat([df_Xi, df_Y, Xt], axis=1)
    return df, y_col, xi_col


def gen_PSD(theta: np.ndarray, dim: int) -> np.ndarray:
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


def get_ergodic(F: np.ndarray, Q: np.ndarray, B: np.ndarray=None,
        x_0: np.ndarray=None, force_diffuse: List[bool]=None, 
        is_warning: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate initial state covariance matrix, and identify 
    diffuse state. It effectively solves a Lyapuov equation

    Parameters:
    ----------
    F : state transition matrix
    Q : initial error covariance matrix
    B : regression matrix
    x_0 : initial x, used for calculating ergodic mean
    force_diffuse : List of booleans of user-determined diffuse state
    is_warning : whether to show warning message

    Returns:
    ----------
    P_0 : the initial state covariance matrix, np.inf for diffuse state
    xi_0 : the initial state mean, 0 for diffuse state
    """
    Q_ = Q.copy()
    dim = Q.shape[0]
    
    # Is is_diffuse is not supplied, create the list
    if force_diffuse is None:
        is_diffuse = np.zeros(dim, dtype=np.bool)
    else: 
        is_diffuse = deepcopy(force_diffuse)
        if len(is_diffuse) != dim:
            raise ValueError('is_diffuse has wrong size')

    # Check F and Q
    if F.shape[0] != F.shape[1]:
        raise TypeError('F must be a square matrix')
    if Q.shape[0] != Q.shape[1]:
        raise TypeError('Q must be a square matrix')
    if F.shape[0] != Q.shape[0]:
        raise TypeError('Q and F must be of same size')
   
    # If explosive roots, use fully diffuse initialization
    # and issue a warning
    eig = linalg.eigvals(F)
    if np.any(np.abs(eig) > 1):
        if is_warning:
            warnings.warn('Ft contains explosive roots. Assumptions ' + \
                    'of marginal LL correction may be violated, and ' + \
                    'results may be biased or inconsistent. Please provide ' + \
                    'user-defined xi_1_0 and P_1_0.', RuntimeWarning)
        is_diffuse_explosive = get_explosive_diffuse(F)
        is_diffuse = is_diffuse | is_diffuse_explosive

    # Modify Q_ to reflect diffuse states
    Q_ = mask_nan(is_diffuse, Q_, diag=inf_val)
        
    # Calculate raw P_0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P_0 = lyap(F, Q_, 'bilinear')

    # Clean up P_0
    for i in range(dim):
        if np.abs(P_0[i][i]) > max_val:
            is_diffuse[i] = True
    P_0 = mask_nan(is_diffuse, P_0, diag=0)
    
    # Enforce PSD
    P_0_PSD = get_nearest_PSD(P_0)

    # Add nan to diffuse diagonal values
    P_0_PSD += np.diag(np.array([np.nan if i else 0 for i in is_diffuse]))

    # Compute ergodic mean
    if B is None or x_0 is None:
        Bx = np.zeros([dim, 1])
    else:
        if B.shape[0] != dim:
            raise ValueError('B has the wrong dimension')
        Bx = B.dot(x_0)
    Bx[is_diffuse] = 0
    F_star = F.copy()
    F_star[is_diffuse] = 0
    xi_0 = inv(np.eye(dim) - F_star).dot(Bx)

    return P_0_PSD, xi_0


def get_init_mat(P_1_0: np.ndarray) \
        -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get information for diffuse initialization

    Parameters:
    ----------
    P_1_0 : initial state covariance. Diagonal value np.nan if diffuse

    Returns:
    ----------
    number_diffuse : number of diffuse state
    A : selection matrix for diffuse states, equal to P_inf
    Pi : selection matrix for stationary states
    P_star : non-diffuse part of P_1_0
    """
    is_diffuse = np.isnan(P_1_0.diagonal())
    number_diffuse = np.count_nonzero(is_diffuse)
    A = np.diag(is_diffuse.astype(float))
    Pi = np.diag((~is_diffuse).astype(float))

    P_clean = P_1_0.copy()
    P_clean[np.isnan(P_clean)] = 0
    P_star = Pi.dot(P_clean).dot(Pi.T)
    
    return number_diffuse, A, Pi, P_star


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
    cleaned_mat = mat.copy().astype(float, copy=True)
    cleaned_mat[np.abs(cleaned_mat) < min_val] = 0
    cleaned_mat[np.abs(cleaned_mat) > max_val] = inf_val

    return cleaned_mat


def permute(matrix: np.ndarray, index: np.ndarray, 
        axis: str='row') -> np.ndarray:
    """
    Permute a square matrix, may perform row permutation,
    column permutation, or both.
    
    For example:
    a = np.array([[1, 2, 3],
                  [2, 5, 6],
                  [3, 6, 9]])
    b = np.array([2, 0, 1])

    permute(a, b, axis='both') returns:
    np.array([[9, 3, 6],
              [3, 1, 2],
              [6, 2, 5]])

    Parameters:
    ----------
    matrix : input matrix to be permuted
    index : index order for permutation
    axis : axis along which to do permutation

    Returns:
    ----------
    perm_matrix : permuted matrix
    """
    if axis == 'row':
        perm_matrix = matrix[index,:]
    elif axis == 'col':
        perm_matrix = matrix[:, index]
    elif axis == 'both':
        perm_matrix = (matrix[index, :])[:, index]
    else:
        raise ValueError('axis must be "row", "col", or "both".')
    return perm_matrix


def revert_permute(index: np.ndarray) -> np.ndarray:
    """
    Revert matrix indexing to the original order. 

    For example:
    a = np.array([2, 3, 1, 0]) means the original matrix is 
    reordered as [3rd, 4th, 2nd, 1st]. 
    revert_permute(a) returns np.array([3, 2, 0, 1]), which 
    can be used to restore the matrix to its original order.

    Parameters:
    ----------
    index : current indexing

    Returns:
    ----------
    revert_index : index of the original order
    """
    revert_index = index.argsort()
    return revert_index


def partition_index(is_missing: np.ndarray) -> np.ndarray:
    """
    Reshuffle the index with index of observed measurement 
    first. 

    For example:
    is_missing = [True, False, True, False]
    partition_index(is_missing) returns:
    np.array([1, 3, 0, 2])

    Parameters:
    ----------
    is_missing : list of dummies whether y_{t:i} is missing

    Returns:
    ----------
    partitioned_index : partitioned index
    """
    partitioned_index = is_missing.argsort()
    return partitioned_index


def ft(theta: np.ndarray, f: Callable, T: int, x_0: np.ndarray=None, 
        xi_1_0: np.ndarray=None, P_1_0: np.ndarray=None, 
        force_diffuse: List[bool]=None, is_warning: bool=True, 
        const_M_type: str='simple') -> Dict:
    """
    Duplicate arrays in M = f(theta) and generate list of Mt
    Output of f(theta) must contain all the required keys.

    Parameters:
    ----------
    theta : input of f(theta). Underlying parameters to be optimized
    f : obtained from get_f. Mapping theta to M
    T : length of Mt. "Duplicate" M for T times
    x_0 : establish initial state mean
    xi_1_0 : specify initial state mean. override calculated mean
    P_1_0 : initial state cov
    force_diffuse : use-defined diffuse state
    is_warning : whether to display the warning about explosive roots
    const_M_type : type of list of constant matrices, default as 'simple'

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

    # Check PSD of R and Q
    if np.array_equal(M['Q'], M['Q'].T):
        eig_Q = linalg.eigvals(M['Q'])
        if not np.all(eig_Q >= 0):
            raise ValueError('Q is not semi-PSD')
    else: 
        raise ValueError('Q is not symmetric')
    
    if np.array_equal(M['R'], M['R'].T):
        eig_R = linalg.eigvals(M['R'])
        if not np.all(eig_R >= 0):
            raise ValueError('R is not semi-PSD')
    else: 
        raise ValueError('R is not symmetric')
    
    # Generate ft for required keys
    Ft = build_tensor(M['F'], T)
    Ht = build_tensor(M['H'], T)
    Qt = build_tensor(M['Q'], T)
    Rt = build_tensor(M['R'], T)

    # Set Bt if Bt is not Given
    if 'B' not in M_keys:
        dim_xi = M['F'].shape[0]
        if 'D' not in M_keys:
            M.update({'B': np.zeros((dim_xi, 1))})
        else:
            dim_x = M['D'].shape[1]
            M.update({'B': np.zeros((dim_xi, dim_x))})

    if 'D' not in M_keys:
        dim_x = M['B'].shape[1]  # B is already defined
        dim_y = M['H'].shape[0]
        M.update({'D': np.zeros((dim_y, dim_x))})

    # Get Bt and Dt for ft
    Bt = build_tensor(M['B'], T)
    Dt = build_tensor(M['D'], T)

    # Initialization
    if P_1_0 is None or xi_1_0 is None: 
        P_1_0, xi_1_0 = get_ergodic(M['F'], M['Q'], M['B'], 
                x_0=x_0, force_diffuse=force_diffuse, 
                is_warning=is_warning) 
    Mt = {'Ft': Ft, 
            'Bt': Bt, 
            'Ht': Ht, 
            'Dt': Dt, 
            'Qt': Qt, 
            'Rt': Rt,
            'xi_1_0': xi_1_0,
            'P_1_0': P_1_0}
    return Mt


def pdet(array: np.ndarray) -> float:
    """
    Calculate pseudo-determinant. If zero matrix, determinant is 1
    Because we are using log, determinant of 1 is good.

    Parameters:
    ----------
    array : input array
    
    Returns:
    ----------
    array_pdet : pseudo-determinant
    """
    eig, _ = np.linalg.eigh(array)

    # If all eigenvalues are close to 0, np.product(np.array([])) returns 1
    array_pdet = np.product(eig[np.abs(eig) > min_val])
    return array_pdet


def LL_correct(Ht: np.ndarray, Ft: np.ndarray, 
        n_t: np.ndarray, A: np.ndarray, index: np.ndarray=None) \
        -> np.ndarray:
    """
    Calculate Correction term for the marginal likelihood

    Parameters:
    ----------
    Ht : tensor of measurement specification matrices
    Ft : tensor of state transition matrices
    n_t : only the first n_t[t] rows of Ht are used
    A : selection matrix for P_inf_1_0
    index : if not None, sort Ht first

    Returns:
    MLL_correct : correction term for the marginal likelihood
    """
    psi = A.copy()
    MLL_correct = np.zeros(Ft[0].shape)
    if index is None:
        for t in range(len(Ht)):
            if n_t[t] > 0:
                Zt = Ht[t][0:n_t[t]].dot(psi)
                MLL_correct += (Zt.T).dot(Zt)
            psi = Ft[t].dot(psi)
    else:
        for t in range(len(Ht)):
            if n_t[t] > 0:
                Zt = Ht[t][index[t]][0:n_t[t]].dot(psi)
                MLL_correct += (Zt.T).dot(Zt)
            psi = Ft[t].dot(psi)

    return MLL_correct
    

def get_reset(tensor: np.ndarray) -> np.ndarray:
    """
    Get reset array based on whehter tensor[n] == tensor[n-1]

    Parameters:
    ----------
    tensor : input tensor
    
    Returns:
    ----------
    reset_array : true if tensor[n] != tensor[n-1]
    """
    reset_array = np.zeros(tensor.shape[0])
    reset_array[0] = 1
    for i in range(tensor.shape[0]-1):
        if not np.array_equal(
                tensor[i+1, :, :], tensor[i, :, :]):
            reset_array[i+1] = 1
    return reset_array
        

def get_reset_index(Yt: np.ndarray) -> np.ndarray:
    """
    If Yt[i] has missing values, set the corresponding 
    reset_index value to True
    
    Parameters:
    ----------
    Yt : input tensor. If a measurement is missing, use np.nan

    Returns:
    ----------
    reset_index : output array on whether to recalculate LDL etc. 
    """
    T = Yt.shape[0]
    reset_index = ~np.ones(T, dtype=bool)
    reset_index[0] = True
    arr_is_nan = np.isnan(Yt[0])
    for t in range(1, T):
        arr_is_nan_t = np.isnan(Yt[t])
        if not np.array_equal(arr_is_nan_t, arr_is_nan):
            reset_index[t] = True
        arr_is_nan = arr_is_nan_t
    return reset_index


class M_wrap(Sequence):
    """
    Wraper of tensor. Improve efficiency by skipping 
    repeated calculation when m_tensor contains same arrays. 
    """
    
    def __init__(self, m_tensor: np.ndarray, 
            reset: np.ndarray) -> None:
        """
        Create placeholder for calculated matrix. 

        Parameters:
        ----------
        m_tensor : input tensor. 
        reset : if true, then recalculate ldl, etc
        """
        self.m_tensor = m_tensor
        self.reset = deepcopy(reset)

        # Initialize using first value
        self.m_pinvh = None
        self.L = None
        self.D = None
        self.L_I = None
        self.m_pdet = None

    
    def __getitem__(self, index: int) -> np.ndarray:
        """
        Returns indexed array of the wrapped tensor

        Parameters:
        ----------
        index : index of the wrapped tensor

        Returns:
        ----------
        self.m_tensor[index] : indexed array of the wrapped tensor
        """
        return self.m_tensor[index]


    def __setitem__(self, index: int, val: np.ndarray) -> None:
        """
        Set values of the wrapped list

        Parameters:
        ----------
        index : index of the wrapped list
        val : input array
        """
        self.m_tensor[index] = val 


    def __len__(self) -> int:
        """
        Required for a Sequence Object

        Returns:
        ----------
        len(self.m_list) : length of the wrapped list
        """
        return len(self.m_tensor)


    def update_reset(self, reset_index: np.ndarray) -> None:
        """
        Manually insert when to recalculate values

        Parameters:
        ----------
        reset_index : inserted reset
        """
        self.reset = deepcopy(reset_index)


    def refresh(self) -> None:
        """
        Reset self.reset after modification 
        to self.m_tensor. 
        """
        self.reset = get_reset(self.m_tensor)


    def pinvh(self, index: int) -> np.ndarray:
        """
        Return pseudo-inverse of self.m_tensor[index]

        Parameters:
        ----------
        index : index of the wrapped list

        Returns:
        self.m_pinvh : pseudo-inverse
        """
        if self.reset[index] or (self.m_pinvh is None):
            self.m_pinvh = inv(self.m_tensor[index])
        return self.m_pinvh
    

    def ldl(self, index: int) -> Tuple[np.ndarray, np.ndarray, 
            np.ndarray]:
        """
        Calculate L and D from LDL decomposition, and inverse of L

        Parameters:
        ----------
        index : index of the wrapped tensor

        Returns:
        self.L : L of LDL
        self.D : D of LDL
        self.L_I : inverse of L
        """
        if self.reset[index] or (self.L is None):
            self.L, self.D, _ = linalg.ldl(self.m_tensor[index])
            self.L_I, _ = linalg.lapack.dtrtri(self.L, lower=True)
        return self.L, self.D, self.L_I


    def pdet(self, index: int) -> float:
        """
        Calculate pseudo-determinant.

        Parameters:
        ----------
        index : index of the wrapped tensor
        
        Returns:
        ----------
        self.m_pdet : pseudo-determinant
        """
        if self.reset[index] or (self.m_pdet is None):
            self.m_pdet = pdet(self.m_tensor[index])
        return self.m_pdet


@deprecated
def Constant_M(M: np.ndarray, length: int, 
        constant_M_type: str='simple'):
    """
    Determine which class struct to use

    Parameters:
    ----------
    M : input system matrix 
    length : length of the list
    constant_M_type : if 'simple' use simple generator
    """
    if constant_M_type == 'simple':
        return Constant_M_simple(M, length)
    elif constant_M_type == 'complex':
        return Constant_M_complex(M, length)
    else:
        raise ValueError('constant_M_type must take' + \
                ' value "simple" or "complex".')


@deprecated
class Constant_M_simple(object):
    """
    Simple way of creating a list of constant matrix
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

        self.Mt = [M.copy() for _ in range(length)] 
        self.length = length
    
    
    def __setitem__(self, index: int, val: np.ndarray) -> None:
        """
        Set value for the wrapped list of matrices

        Parameters:
        ----------
        index : index number of the list between 0 and self.T
        val : value to replace M at index. 
        """
        self.Mt[index] = val


    def __getitem__(self, index: int) -> np.ndarray:
        """
        Select matrix from the wrapped list of matrices

        Parameters:
        ----------
        index : index number of the list

        Returns:
        ----------
        Mt_index : Constant_M_simple[index]
        """
        return self.Mt[index]


    def __len__(self) -> int:
        """
        Set length of the list, required for an object inherited 
        from Sequence.

        Returns:
        ----------
        self.length : length of the list
        """
        return self.length


@deprecated("This class is deprecated because we are move from " + \
        "list to tensor. It's cool stuff though.")
class Constant_M_complex(Sequence):
    """
    If the sequence of system matrix is mostly constant over time 
    (with the exception of occasional deviation), using 
    Constant_M_complex saves memory space. It mimics the behavior 
    of a regular list but use one single baseline M and stores 
    any deviation.

    Example:
    ----------
    M = np.array([[5, 3],[3, 4]])  # baseline matrix
    T = 100  # intended length of the list
    
    # Mt has similar behavior as [copy.deepcopy(M) for _ in range(T)]
    Mt = Constant_M_complex(M, T)  
    
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

        self.M = M.copy()
        self._M = M.copy()  # benchmark M
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
        self.Mt.update({self.index: self.M.copy()})
        self.M = self._M.copy()


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
            self.Mt.update({index: val.copy()}) 


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
        Mt_index : Constant_M_complex[index]
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

