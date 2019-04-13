from collections import Sequence
from base import Base
from ../core.em import EM
from ../core.kalman_fiter import Filter

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

def create_col(col, suffix='_pred'):
    """
    Create column names for filter predictions. Default suffix is '_pred'
    """
    col_pred = []
    for i in col:
        col_pred.append(i + suffix)
    return col_pred
    
class ConstantEM(Base):
    """
    EM solver with Mt = M
    """

    def __init__(self, f, t):
        self.f_M = lambda theta: F_theta(theta, f, t)
        self.f = f
        self.theta_opt = None
        self.x_col = None
        self.y_col = None

    def fit(self, df, theta, x_col, y_col):
        """
        Fit the model using EM algorithm
        """
        # Initialize
        em = EM(self.f_M)
        self.x_col = x_col
        self.y_col = y_col
        # Convert dataframe to lists
        Xt = self._df_to_list(df[x_col])
        Yt = self._df_to_list(df[y_col])

        # Run EM solver
        self.theta_opt = em.fit(theta, Xt, Yt)

    def predict(self, df_extended):
        """
        Predict time series. df_extended should contain both training and test data.
        If y_t in test data is not available, use np.nan
        """
        
        # Generate system matrices for prediction
        Mt = F_theta(self.theta_opt, self.f, T_extend)
        kf = Filter(Mt)
        Xt = self._df_to_list(df[self.x_col])
        Yt = self._df_to_list(df[self.y_col])

        # Run Kalman Filter and get y_t_1t
        kf(Xt, Yt)
        xi_t_1t = kf.xi_t_1t
        y_t_1t = kf.get_y(kf.xi_t_1t)
        return self._list_to_df(df_list, create_col(y_col))

        


     
