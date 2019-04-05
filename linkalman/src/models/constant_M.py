from collections import Sequence
from base import Base

class Constant_M(Sequence):

    def __init__(self, M, length):
        self.M = M
        self.length = length

    def __getitem__(self, index):
        return self.M

    def __len__(self):
        return self.length

class F_theta(object):
    
    def __init__(self, f, T):
        self.T = T
        self.f = f

    def __call__(theta):
        M = self.f(theta)
        self.Ft = Constant_M(M.F, self.T)
        self.Bt = Constant_M(M.B, self.T)
        self.Ht = Constant_M(M.H, self.T)
        self.Dt = Constant_M(M.D, self.T)
        self.Qt = Constant_M(M.Q, self.T)
        self.Rt = Constant_M(M.R, self.T)
        self.xi_1_0 = M.xi_1_0
        self.P_1_0 = M.P_1_0
    
    def __getattr__(self, index):
        return getattr(self, index)


class Simple_EM(Base):

    def __init__():
        pass

     
