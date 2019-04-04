import numpy as np

class Base(object):
    
    def _df_to_list(iself, df):
        
        L = []
        for i in range(df.shape[0]):
            L.append(np.matrix(df.loc[i,:]).T)
        return L
