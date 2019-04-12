import numpy as np

class Base(object):
    
    def _df_to_list(self, df):
        """
        Convert pandas dataframe to list of arrays
        """ 
        L = []
        for i in range(df.shape[0]):
            L.append(np.array([df.loc[i,:]]).T)
        return L

    def data_gen(self):
        """
        Generate synthetic dataframe
        """
        pass

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
