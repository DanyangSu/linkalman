def inv():
    pass

def ldl_transform():
    pass

def check_M():
    pass

class Mt_series(object):

    def __init__(self, f_theta):

        # If a list of matrices has identical matrices, we can save lots of matrix operations
        self.M = None
        self.M_inv = None
        self.M_transpose = None

    
    def inv(self, t):
        if self[t] != self.M:
            self.M = self[t]
            self.M_inv = linalg.pinvh(self[t])
        return self.M_inv

class Mt(object):
    """
    Performance Improvement over native f_theta
    """
    def __init__(self, f_theta)
        pass

    def _check_Mt_names():
        pass

        
