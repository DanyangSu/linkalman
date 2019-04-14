import numpy as np
import pandas as pd

class Base(object):
    
    def _df_to_list(self, df):
        """
        Convert pandas dataframe to list of arrays
        """ 
        L = []
        for i in range(df.shape[0]):
            L.append(np.array([df.loc[i,:]]).T)
        return L

    def _list_to_df(self, df_list, col):
        """
        Convert list of n-by-1 arrays to a dataframe
        """
        df_val = np.concatenate([i.T for i in df_list])
        df = pd.DataFrame(data=df_val, columns=col)
        return df

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError




class BaseConstantModel(object):
    """
    Solve HMM with Time series specification in xi. The dimensions of measurement matrices are 
    determined by x and y. 
    x must contain 1 as its first column
    """

    def __init__(self):
        raise NotImplementedError
    
    def fit(self, df, x_col, y_col, **kwargs={}):
        """
        Fit a time-series model. For specification design, refer to theory.pdf
        """
        x_dim = len(x_col)
        y_dim = len(y_col)
        T = df.shape[0]

        # Create f
        kwargs.update({'x_dim': x_dim, 'y_dim': y_dim})
        f = lambda theta: self.get_f(theta, **kwargs)

        # Fit model using ConstantEM
        ConstEM = ConstantEM(f, T)
        ConstEM.fit(df, theta, x_col, y_col)
        self.mod = ConstEM

    def predict(self, df):
        """
        Predict filtered yt
        """
        return self.mod.predict(df)

    def get_f(self, theta, **kwargs):
       raise NotImplementedError


    @staticmethodNotImplementedError
    def gen_PSD(theta, dim):
        """
        Generate covariance matrix from theta. Requirement:
        len(theta) = (dim**2 + dim) / 2
        """
        L = np.zeros([dim, dim])

        # Fill diagonal values
        for i in range(dim):
            L[i][i] = np.exp(theta[i])

        # Fill lower off-diagonal values
        theta_off = theta[dim:]
        idx = np.tril_indices(dim, k=-1)
        L[idx] = theta_off
        return L.dot(L.T)
