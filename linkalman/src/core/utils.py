from collections import Sequence
import numpy as np
from scipy import linalg

class M_series(Sequence):
    
    def __init__(self, m_list):
        
        self.m = None
        self.m_pinv = None
        self.m_transpose = None
        self.m_dtrtri = None
        self.m_pinvh = None
        self.m_list = m_list
        
    def __getitem__(self, index):
        return self.m_list[index]

    def __len__(self):
        return len(self.m_list)
    
    def pinv(self, index):
        if (not np.array_equal(self.m, self.m_list[index])) or self.m_pinv is None:
            self.m = self.m_list[index]
            self.m_pinv = linalg.pinv(self.m)
        return self.m_pinv
    
    def transpose(self, index):
        if (not np.array_equal(self.m, self.m_list[index])) or self.m_transpose is None:
            self.m = self.m_list[index]
            self.m_transpose = self.m_list[index].T
        return self.m_transpose 

    def dtrtri(self, index):
        if (not np.array_equal(self.m, self.m_list[index])) or self.m_dtrtri is None:
            self.m = self.m_list[index]
            self.m_dtrtri = linalg.lapack.clapack.dtrtri(self.m)
        return self.m_dtrtri

    def pinvh(self, index):
        if (not np.array_equal(self.m, self.m_list[index])) or self.m_pinvh is None:
            self.m = self.m_list[index]
            self.m_pinvh = linalg.pinvh(self.m)
        return self.m_pinvh

