from linkalman.src.constant_em import F_theta

import sys
import os

sys.path.append(os.path.abspath('../..'))
def f_t(the):
    return {'F': the, 'B':the+1, 'H':the+2, 'D':the+3, 'Q':the+4,'R':the+5,'xi_1_0':the+6,'P_1_0':the+7}

theta = 2
Mt = F_theta(theta, f_t, 10)


print('df')
    
