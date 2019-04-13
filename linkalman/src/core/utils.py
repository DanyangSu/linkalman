from collections import Sequence
import numpy as np
from scipy import linalg

def inv(h_array):
    """
    Calculate pinvh of PSD array
    """
    return linalg.pinvh(h_array)

class M_wrap(Sequence):
    """
    Wraper of array lists. Improve efficiency by skipping 
    repeated calculation when m_list contains same arrays. 
    """ 
    def __init__(self, m_list):
        self.m = None
        self.m_pinvh = None
        self.L = None
        self.D = None
        self.L_I = None
        self.pdet = None
        self.m_list = m_list
        
    def __getitem__(self, index):
        return self.m_list[index]

    def __len__(self):
        return len(self.m_list)

    def _equal_M(self, t):
        """
        Return true if self.m_list[t] == self.m. 
        If false, set self.m = self.m_list[t]
        """
        if np.array_equal(self.m, self.m_list[index]):
            return True
        else:
            self.m = self.m_list[index]
            return False
    
    def pinvh(self, index):
        """
        Return pseudo-inverse of self.m_list[index]
        """
        if (not self._equal_M(index)) or self.m_pinv is None:
            self.m_pinvh = inv(self.m)
        return self.m_pinvh
    
    def ldl(self, index):
        """
        Calculate L and D from LDL decomposition, and inverse of L
        """
        if (not self._equal_M(index)) or self.m_l is None:
            self.L, self.D, _ = linalg.ldl(self.m)
            self.L_I = linalg.lapack.clapack.dtrtri(self.L, lower=True)
        return self.L, self.D, self.L_I

    def pdet(self, index):
        """
        Calculate pseudo-determinant
        """
        if (not self._equal_M(index)) or self.m_pdet is None:
            eig = linalg.eigh(self.m, eigvals_only=True)

            # If all eigenvalues are close to 0, np.product(np.array([])) returns 1
            self.pdet = np.product(eig[abs(eig)>1e-12])
        return self.pdet
