from collections import Sequence
import numpy as np

class M_series(Sequence):
    
    def __init__(self, m_list):
        
        self.m = None
        self.m_inv = None
        self.m_transpose = None
        self.m_list = m_list
        
    def __getitem__(self, index):
        return self.m_list[index]

    def __len__(self):
        return len(self.m_list)
    
    def inv(self, index):
        if (not np.array_equal(self.m, self.m_list[index])) or self.m_inv is None:
            self.minv = self.m_list[index]
            self.m_inv = np.linalg.pinv(self.m)
        return self.m_inv
    
    def transpose(self, index):
        if (not np.array_equal(self.m, self.m_list[index])) or self.m_transpose is None:
            self.mt = self.m_list[index]
            self.m_transpose = self.m_list[index].T
        return self.m_transpose 
