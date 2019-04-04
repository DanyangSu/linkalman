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
        self.F = Constant_M(M.F, self.T)
        self.B = Constant_M(M.B, self.T)
        self.H = Constant_M(M.H, self.T)
        self.D = Constant_M(M.D, self.T)
        self.Q = Constant_M(M.Q, self.T)
        self.R = Constant_M(M.R, self.T)


class Simple_EM(Base):

    def __init__():
        pass

     
