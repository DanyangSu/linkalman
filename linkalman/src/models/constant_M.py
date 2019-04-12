from collections import Sequence
from base import Base
from ../core.em import EM

class Constant_M(Sequence):

    def __init__(self, M, length):
        self.M = M
        self.length = length

    def __getitem__(self, index):
        return self.M

    def __len__(self):
        return self.length

def F_theta(theta, f, t):
    """
    Duplicate arrays in M = f(theta) and generate list of Mt
    """ 
    M = f(theta)
    Ft = Constant_M(M['F'], T)
    Bt = Constant_M(M['B'], T)
    Ht = Constant_M(M['H'], T)
    Dt = Constant_M(M['D'], T)
    Qt = Constant_M(M['Q'], T)
    Rt = Constant_M(M['R'], T)
    xi_1_0 = M['xi_1_0']
    P_1_0 = M['P_1_0']
    return {'Ft': Ft, 'Bt': Bt, 'Ht': Ht, 'Dt': Dt, 'Dt': Dt, 
            'Qt': Qt, 'Rt': Rt, 'xi_1_0': xi_1_0, 'P_1_0': P_1_0}
    
class Simple_EM(Base):
    """
    EM solver with Mt = M
    """

    def __init__(self, f, t):
        self.f = lambda theta: F_theta(theta, f, t)
        self.theta_opt = None
        self.x_col = None
        self.y_col = None

    def fit(self, df, theta, x_col, y_col):
        """
        Fit the model using EM algorithm
        """
        # Initialize
        em = EM(self.f)

        # Convert dataframe to lists
        Xt = self._df_to_list(df[x_col])
        Yt = self._df_to_list(df[y_col])

        # Run EM solver
        self.theta_opt = em.fit(theta, Xt, Yt)

    def predict(self, df_extend):
        """
        Predict time series
        """
        # Process Xt_extend
        Xt_extend = self._df_to_list(df_extend[self.x_col])
        T_extend = len(Xt_extend)
        
        # Use Kalman Filter to predict Yt_extend


     
