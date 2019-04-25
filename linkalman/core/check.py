import numpy as np
from copy import deepcopy

T = 100

# create param
F = np.array([[2,4],[1,3]])
B = np.array([[3,5],[6,4]])
H = np.array([[8,7],[2,1]])
D = np.array([[5,4],[2,5]])
Q = np.array([[7,3],[3,7]])
R = np.array([[6,2],[2,6]])
xi_1_0 = np.array([[4],[6]])
P_1_0 = np.array([[9,6], [6,9]])

Ft = [deepcopy(F) for _ in range(T)]
Bt = [deepcopy(B) for _ in range(T)]
Ht = [deepcopy(H) for _ in range(T)]
Dt = [deepcopy(D) for _ in range(T)]
Qt = [deepcopy(Q) for _ in range(T)]
Rt = [deepcopy(R) for _ in range(T)]

# create Yt and Xt
X = np.random.multivariate_normal(np.array([2,3]), np.array([[3,2],[2,3]])).reshape(1,-1).reshape(-1,1)
Y = np.random.multivariate_normal(np.array([4,5]), np.array([[5,3],[3,5]])).reshape(1,-1).reshape(-1,1)
Xt = [deepcopy(X) for _ in range(T)]
Yt = [deepcopy(Y) for _ in range(T)]

Yt[0][1][0] = np.nan




M = {'Rt':Rt, 'Bt':Bt, 'Ht': Ht, 'Dt':Dt, 'Qt':Qt, 'Ft':Ft, 'xi_1_0':xi_1_0, 'P_1_0': P_1_0}


from kalman_filter import Filter

kf = Filter(M)
kf(Xt, Yt)

from kalman_smoother import Smoother
ks = Smoother(kf)
ks()

print('Finish')
# drop xi_length and y_length
