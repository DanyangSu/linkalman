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
