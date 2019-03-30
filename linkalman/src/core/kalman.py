#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
from scipy.stats import multivariate_normal as mn
import numpy as np
from scipy.optimize import minimize
from functools import partial
from copy import deepcopy
import datetime



class Kalman(object):

    def __init__(self, **opt):
        """
        initialize for kalman filter, following notational convention in Halmilton Time Series Book 
        Kalman Filter chapter. 
        """
        self.opt = opt
        self.MLE = None
        
    def kalman_filter(self, Q, F, R1, A1, X1, H1, R2, A2, X2, H2, Y1, Y1, T, T_cutoff):
        """
        iterate forward to get Sai_t_t, Sai_t1_t, P_t_t, and P_t1_t
        """
        Sai_t_t = []
        P_t_t = []

        # Forward filter
        for t in range(T):
            if t < T_cutoff:
                H_t = H1
                R = R1
                A_t = A1
                X_t = X1
                Y_t = np.matrix([[Y1[t]], [Y1[t]]])
            else:
                H_t = H2
                R = R2
                A_t = A2
                X_t = X2
                Y_t = np.matrix([[Y1[t]]])

            sai_t_t, sai_t1_t, p_t_t, p_t1_t = self._filter(H_t, self.P_t1_t[t], R, self.Sai_t1_t[t], Y_t, A_t, X_t, F, Q)
            Sai_t_t.append(deepcopy(sai_t_t))
            P_t_t.append(deepcopy(p_t_t))
            self.P_t1_t.append(deepcopy(p_t1_t))
            self.Sai_t1_t.append(deepcopy(sai_t1_t))

        # Drop last two terms in Sai_t1_t and P_t1_t, since they are sai_T1_T, and P_T1_T. We Don't need them.
        self.Sai_t1_t.pop()
        self.P_t1_t.pop()
        
        return Sai_t_t, P_t_t, T, T_cutoff
    
    @staticmethod
    def _filter(H_t, p_t_1t, R, sai_t_1t, y_t, A_t, x_t, F_t, Q):
        """
        given sai_{t|t-1} and P_{t|t-1}, find sai_
        t1 means t+1, 1t means t-1
        """
        # Update Sai and P
        I = np.matrix(np.eye(2))
        K_t = F_t * p_t_1t * H_t * (H_t.T * p_t_1t * H_t + R).I
        sai_t_t = sai_t_1t + K_t * (y_t - A_t.T * x_t - H_t.T * sai_t_1t)
        
        # Calculate p_t_t by Joseph Form
        p_t_t = (I - K_t * H_t.T) * p_t_1t * (I - K_t * H_t.T).T + K_t * R * K_t.T
        sai_t1_t = F_t * sai_t_t
        p_t1_t = F_t * p_t_t * F_t.T + Q      
    
        return sai_t_t, sai_t1_t, p_t_t, p_t1_t

    def kalman_smoother(self, Q, F, R1, A1, X1, H1, R2, A2, X2, H2, Y1, Y1, T):
        """
        iterate backward to get Sai_t_T and P_t_T
        """
        Sai_t_T = []
        Sai_t_T.append(deepcopy(self.Sai_t_t[-1]))
        P_t_T = []
        P_t_T.append(deepcopy(self.P_t_t[-1]))
        
        # Backward smoother
        for t in reversed(range(T - 1)): # we have already provided the last value for Sai_t_T and P_t_T
            sai_t_T, p_t_T = self._smoother(self.P_t_t[t], F, self.P_t1_t[t], self.Sai_t_t[t], self.Sai_t1_t[t], \
                Sai_t_T[T - t -2], P_t_T[T - t - 2])
            Sai_t_T.append(deepcopy(sai_t_T))
            P_t_T.append(deepcopy(p_t_T))
        return Sai_t_T[::-1], P_t_T[::-1] # perform a reverse to maintain original time order
        
    @staticmethod
    def _smoother(p_t_t, F_t, p_t1_t, sai_t_t, sai_t1_t, sai_t1_T, p_t1_T):
        """
        Backward iteration to get smoothed sai and p
        """ 
        J_t = p_t_t * F_t.T * p_t1_t.I
        sai_t_T = sai_t_t + J_t * (sai_t1_T - sai_t1_t)
        p_t_T = p_t_t + J_t * (p_t1_T - p_t1_t) * J_t.T
        
        return sai_t_T, p_t_T

    def _matrix_init(self, arg):
        """
        Takes argumments to initialize matrices
        Customize the initialization for different set-ups
        """
        sig1 = np.exp(arg[0])
        sig2 = np.exp(arg[1])
        coef = (np.exp(arg[2]) - 1) / (np.exp(arg[2]) + 1)
        a = arg[3]
        h = arg[4]
        q = np.exp(arg[5])
        sai_init = np.matrix([[2 * self.Y_2_n1 - self.Y_2_n2], [self.Y_2_n1]])
        p_init = np.matrix([[q + 5 * sig1, 2 * sig1], [2 * sig1, sig1]])
        cov = np.sqrt(sig1 * sig2) * coef

        # Initialize 
        Q = np.matrix([[q, 0], [0, 0]])
        F = np.matrix([[2, -1], [1, 0]])
        R1 = np.matrix([[sig1, cov], [cov, sig2]])
        A1 = np.matrix([[0, 0], [a, 0]]).T #To be consistent with the convention in the textbook
        X1 = np.matrix([[1], [1]])
        H1 = np.matrix([[1, 0], [h, 0]]).T
        R2 = np.matrix([[sig2]])
        A2 = np.matrix([[a]]).T
        X2 = np.matrix([[1]])
        H2 = np.matrix([[h,0]]).T

        return  Q, F, R1, A1, X1, H1, R2, A2, X2, H2, sai_init, p_init

    def mle_estimator(self, arg, X, Y):
        """
        calculate likelihood given primitives
        formular is:
        s_t - s_1t = s_1t - s_2t + delta_t (in the future, genralize it to add ma2 and seasonality)
        y_t = s_t + w_t
        x_t = beta0 + beta1 * s_t + v_t
        For initialization, I use 
        [sai_1_0, sai_0_(-1), sai_0_(-2)] = 2y_(-1) - y_(-2), y_(-1), y_(-2)]
        All the notations follow Halmilton Time Series Textbook
        """
        
        Q, F, R, H, R, A, H, sai_init, p_init = self._matrix_init(arg)
        self.Sai_t1_t = []
        self.Sai_t1_t.append(sai_init)
        self.P_t1_t = []
        self.P_t1_t.append(p_init)

        # Forward filter
        self.Sai_t_t, self.P_t_t, T, T_cutoff = self.kalman_filter(Q, F, R1, A1, X1, H1, R2, A2, X2, H2, Y2, Y1, T, T_cutoff)

        # Backward smoother
        self.Sai_t_T, self.P_t_T = self.kalman_smoother(Q, F, R1, A1, X1, H1, R2, A2, X2, H2, Y2, Y1, T)
        
        # Calculate MLE
        return -self._mle(A1, X1, H1, R1, A2, X2, H2, R2, Y2, Y1, T, T_cutoff)
    
    def _mle(self, A1, X1, H1, R1, A2, X2, H2, R2, Y2, Y1, T, T_cutoff):
        """
        Use Gaussian Error for MLE. 
        """
        f = 0 
        for t in range(T):
            if t < T_cutoff:
                H_t = H1
                R = R1
                A_t = A1
                X_t = X1
                Y_t = np.matrix([[Y1[t]], [Y1[t]]])
            else:
                H_t = H2
                R = R2
                A_t = A2
                X_t = X2
                Y_t = np.matrix([[Y1[t]]])

            mu = A_t.T * X_t + H_t.T * self.Sai_t_T[t]
            Sigma = H_t.T * self.P_t_T[t] * H_t + R
            Y_t_array = np.squeeze(np.asarray(Y_t))
            mu_array = np.squeeze(np.asarray(mu))
            log_pdf = np.log(max(mn.pdf(Y_t_array, mean=mu_array, cov=np.asarray(Sigma)), 1e-7)) #make sure we dont get too small values
            f += log_pdf
        self.MLE = f
        return f

    def callbackF(self, arg):
        """
        Display
        """
        print('{} --- {}'.format(str(arg), self.MLE))
    
    def fit(self, X, Y, keep=False):
        """
        Fit a Kalman smoothed Filter, use this function to customize Kalman Filter specification
        I assume that Y1 and Y1 have the same start date of availability (if we observe landed numbers,
        we should also observe numbers at the n day). And if we drop Y1 at certain days, we should also 
        drop days at Y1. The current model set up is such that there is no gap in days. 
        Y1 and Y1 should be df with first column named ds and second column named Y
        The function will perform a check before moving on.
        """
        # Check input integrity
        if X.shape[0] != Y.shape[0]:
            raise BaseException('X and Y mismatched')

        # Check completeness of X
        # to-do

        # Start optimization
        obj = partial(self.mle_estimator, X=X, Y=Y)
        self.sol = EM(obj)

    def predict(self, X):
        """
        Predict y_t up to t days after the last day in train data
        Return a DataFrame in the same formate as Y1 or Y1
        Note that since we impose the restriction that Y_t is unbiased measure of sai_t,
        we only need to get sai_t_T for t<=T and make predictions when t>T
        """
        raise BaseException('Not Implemented') 
    
        
    
    
        
        
        
