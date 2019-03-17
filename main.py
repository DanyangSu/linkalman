#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:14:34 2018

@author: Danyang Su
"""

import pandas as pd
import os
import numpy as np
import datetime
from kalman import Kalman
from copy import deepcopy

# create fake data
path = '/Users/dsu/codebase/kalman'
name = 'fake_data.csv'

pre_T = 2 # This is to get initial value
if not os.path.isfile(os.path.join(path, name)):
    arg = [0, 0.5, 2, 3, 1.1, 0.2]
    sig1 = np.exp(arg[0])
    sig2 = np.exp(arg[1])
    coef = (np.exp(arg[2]) - 1) / (np.exp(arg[2]) + 1)
    a = arg[3]
    h = arg[4]
    q = np.exp(arg[5])
    cov = np.sqrt(sig1 * sig2) * coef

    # Initialize 
    Q = np.matrix([[q, 0], [0, 0]])
    F = np.matrix([[2, -1], [1, 0]])
    R1 = np.matrix([[sig1, cov], [cov, sig2]])
    A1 = np.matrix([[0, 0], [a, 0]]).T #To be consistent with the convention in the textbook
    X1 = np.matrix([[1], [1]])
    H1 = np.matrix([[1, 0], [h, 0]]).T

    sai_n1 = 6
    sai_n2 = 4
    delta_Sai = np.random.multivariate_normal(mean=np.array([0, 0]), cov=Q)
    sai_init = np.matrix([[2 * sai_n1 - sai_n2], [sai_n1]]) + np.matrix(delta_Sai).T

    # generate data
    main_T = 1000
    data_T = pre_T + main_T
    Sai_t = [sai_init]
    ds_t = datetime.datetime.strptime('20170710', '%Y%m%d')
    Ds_t = [deepcopy(ds_t)]
    Y2_t = []
    Y1_t = []
    X2_t = []
    X1_t = []
    for t in range(data_T):

        # generate noise
        delta_y = np.random.multivariate_normal(mean=np.array([0, 0]), cov=R1)
        delta_sai = np.random.multivariate_normal(mean=np.array([0, 0]), cov=Q)

        # generate obs
        y_t = A1.T * X1 + H1.T * Sai_t[t] + np.matrix(delta_y).T
        y2_t = y_t.item((0,0))
        y1_t = y_t.item((1,0))
        ds_t += datetime.timedelta(days=1) 
        sai_t = F * Sai_t[t] + np.matrix(delta_sai).T
        
        Sai_t.append(deepcopy(sai_t))
        Y2_t.append(deepcopy(y2_t))
        Y1_t.append(deepcopy(y1_t))
        Ds_t.append(deepcopy(ds_t))
        X2_t.append(1)
        X1_t.append(1)
        

    _ = Sai_t.pop()
    _ = Ds_t.pop()
    Sai = [i.item((0, 0)) for i in Sai_t]
    df = pd.DataFrame({'Sai_t': Sai, 'Y2_t': Y2_t, 'Y1_t': Y1_t, 'Ds_t': Ds_t, 'X2_t': X2_t, 'X1_t': X1_t}) 
    df.to_csv(os.path.join(path, name))

else:
    # read fake data
    df = pd.read_csv(os.path.join(path, name))

df_pre = df[0: pre_T].copy()
df = df[pre_T: ].copy()
df.reset_index(inplace=True, drop=True)
Y_2_n1 = df_pre.Y2_t[1]
Y_2_n2 = df_pre.Y2_t[0]
df['Ds_t'] = pd.to_datetime(df.Ds_t)
df['ds'] = pd.to_datetime(df.Ds_t.dt.date)
pred_length = 10 #pred 10 days 
Y = df[['Y2_t', 'Y1_t']].copy()
X = df[['X2_t', 'X1_t']].copy()
# run model
kf = Kalman(tol=1e-6, method='Nelder-Mead', options={'disp': True})
kf.fit(X, Y)
print(kf.sol)
pred = kf.predict(pred_t)

df_merge = pd.merge(df, pred, on='ds', how='inner')
df_merge.to_csv(os.path.join(path, 'pred_data.csv'))

