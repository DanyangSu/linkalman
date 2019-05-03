from collections.abc import Sequence
import numpy as np
from scipy import linalg
from typing import List, Any, Callable

__all__ = ['mask_nan', 'inv', 'M_wrap']

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
            eig = linalg.eigh(self.m, eigvals_only=True)

            # If all eigenvalues are close to 0, np.product(np.array([])) returns 1
            self.m_pdet = np.product(eig[np.abs(eig)>1e-12])
        return self.m_pdet
