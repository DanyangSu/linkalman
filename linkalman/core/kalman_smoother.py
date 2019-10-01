import numpy as np
from typing import List, Tuple, Dict
import scipy
from .utils import inv, get_nearest_PSD, permute, pdet, \
        revert_permute, preallocate, get_init_mat, LL_correct
from . import Filter

__all__ = ['Smoother']


class Smoother(object):
    """
    Given a filtered object, Smoother returns smoothed state estimation.
    Given an BSTS:

    xi_{t+1} = F_t * xi_t + B_t * x_t + v_t     (v_t ~ N(0, Qt))
    y_t = H_t * xi_t + D_t * x_t + w_t     (w_t ~ N(0, Rt))

    and initial conditions:

    xi_1_0 = E(xi_1) 
    P_1_0 = Cov(xi_1)

    We want to solve:

    xi_t_{t-1} = E(xi_t|Info(T))
    P_t_{t-1} = Cov(xi_t|Info(T))

    where Info(t) is the information set at time t, and T = max(t). 
    Using forward filtering then backward smoothing, we are able to 
    characterize the distribution of the BSTS based on the full 
    information set up to T. Refer to doc/manual.pdf for details.
    """

    def __init__(self) -> None:
        """
        Initialize a Kalman Smoother. self.delta2 and self.chi2 
        are used for EM algorithms later.
        """
        self.xi_t_T = None
        self.P_t_T = None
        self.xi2_t_T = None
        self.Pcov_t_t1 = None
        self.delta2 = None
        self.chi2 = None
        self.r0_t = None
        self.r1_t = None
        self.N0_t = None
        self.N1_t = None
        self.N2_t = None
        self.is_smoothed = False
        

    def init_attr_smoother(self, kf: Filter) -> None:
        """
        Initialize attributes for Kalman smoother

        Parameters:
        ----------
        kf : a filter instance
        """
        # Include filtered results if kf is fitted and kf.for_smoother is True
        if not kf.is_filtered:
            raise TypeError('The Kalman filter object is not fitted yet')
        if not kf.for_smoother:
            raise TypeError('The Kalman filter object is not for smoothers')
        self.__dict__.update(kf.__dict__)

        # Initiate state variables
        self.xi_t_T = preallocate(self.T)
        self.P_t_T = preallocate(self.T)
        self.xi2_t_T = preallocate(self.T)
        self.Pcov_t_t1 = preallocate(self.T)
        self.delta2 = preallocate(self.T)
        self.chi2 = preallocate(self.T)

        # Initiate r and N
        self.r0_t = preallocate(self.T)
        self.r1_t = preallocate(self.t_q)
        self.N0_t = preallocate(self.T)
        self.N1_t = preallocate(self.t_q)
        self.N2_t = preallocate(self.t_q)
        
        self.r0_t[self.T-1] = np.zeros([self.xi_length, 1])
        self.N0_t[self.T-1] = np.zeros([self.xi_length, self.xi_length])
        
        if self.t_q > 0:
            self.r1_t[self.t_q-1] = np.zeros([self.xi_length, 1])
            self.N1_t[self.t_q-1] = np.zeros([self.xi_length, self.xi_length])
            self.N2_t[self.t_q-1] = np.zeros([self.xi_length, self.xi_length])


    def fit(self, kf: Filter) -> None:
        """
        Run backward smoothing. 

        Parameters: 
        ----------
        kf : a filter instance
        """
        self.init_attr_smoother(kf)

        # Start backward smoothing
        for t in reversed(range(self.T)):
            if t >= self.t_q:
                self._sequential_smooth(t)
            else:
                self._sequential_smooth_diffuse(t)

        self.is_smoothed = True
            

    def _sequential_smooth(self, t: int) -> None:
        """
        Update Kalman Smoother at time t. Refer to doc/manual.pdf
        for details on the notation of each variables.

        Parameters:
        ----------
        t : time index
        """
        H_t = self.Ht_tilde[t]
        d_t = self.d_t[t]
        Upsilon = self.Upsilon_star_t[t]
        L_t = self.L_star_t[t]
        n_t = self.n_t[t]

        # Backwards iteration on r and N
        r_t_1i = self.r0_t[t]
        N_t_1i = self.N0_t[t]
        
        for i in reversed(range(n_t)):
            H_i_T = H_t[i:i+1].T
            r_t_1i = (H_i_T).dot(d_t[i]) / Upsilon[i] + \
                    (L_t[i].T).dot(r_t_1i)
            N_t_1i = (H_i_T).dot(H_i_T.T) / Upsilon[i] + \
                    (L_t[i].T).dot(N_t_1i).dot(L_t[i])

        # Update smoothed xi, P and Pcov
        xi_t_1 = self.xi_t[t][0]
        P_t_1 = self.P_star_t[t][0]
        xi_t_T = xi_t_1 + P_t_1.dot(r_t_1i)
        P_t_T = get_nearest_PSD(P_t_1 - P_t_1.dot(N_t_1i).dot(P_t_1))
        self.xi_t_T[t] = xi_t_T
        self.P_t_T[t] = P_t_T

        # Update r and N for current period
        self.r0_t[t] = r_t_1i
        self.N0_t[t] = N_t_1i
        
        # Update r, N and P_1t_t_T from t to t-1
        if t > 0:
            Pcov = self.P_star_t[t-1][self.n_t[t-1]].dot(self.Ft[t-1].T).dot(
                self.I - N_t_1i.dot(P_t_1))
            self.Pcov_t_t1[t-1] = Pcov
            self.r0_t[t-1] = (self.Ft[t-1].T).dot(r_t_1i)
            self.N0_t[t-1] = (self.Ft[t-1].T).dot(N_t_1i).dot(self.Ft[t-1])


    def _sequential_smooth_diffuse(self, t: int) -> None:
        """
        Update diffuse Kalman Smoother at time t. Refer to doc/manual.pdf
        for details on the notation of each variables.

        Parameters:
        ----------
        t : time index
        """
        d_t = self.d_t[t]
        H_t = self.Ht_tilde[t]
        Upsilon_inf = self.Upsilon_inf_t[t]
        Upsilon_star = self.Upsilon_star_t[t]
        L0_t = self.L0_t[t]
        L1_t = self.L1_t[t]
        L_star_t = self.L_star_t[t]
        gt_0 = self.Upsilon_inf_gt_0_t[t]
        n_t = self.n_t[t]

        # Backwards iteration on r and N
        r0_t_1i = self.r0_t[t]
        r1_t_1i = self.r1_t[t]
        N0_t_1i = self.N0_t[t]
        N1_t_1i = self.N1_t[t]
        N2_t_1i = self.N2_t[t]
        
        for i in reversed(range(n_t)):
            H_i_T = H_t[i:i+1].T
            
            # If Upsilon_inf > 0
            if gt_0[i]:
                
                # Must update r1 first, bc it uses r0_t_1i
                r1_t_1i = (H_i_T).dot(d_t[i]) / Upsilon_inf[i] + \
                        (L1_t[i].T).dot(r0_t_1i) + (L0_t[i].T).dot(r1_t_1i)
                r0_t_1i = (L0_t[i].T).dot(r0_t_1i)

                # Order of updating: N2->N1->N0
                L1N1L0 = (L1_t[i].T).dot(N1_t_1i).dot(L0_t[i])
                L1N0L0 = (L1_t[i].T).dot(N0_t_1i).dot(L0_t[i])
                N2_t_1i = -(H_i_T).dot(H_i_T.T) * Upsilon_star[i] / \
                        np.power(Upsilon_inf[i], 2) + L1N1L0 + L1N1L0.T + \
                        (L0_t[i].T).dot(N2_t_1i).dot(L0_t[i]) + \
                        (L1_t[i].T).dot(N0_t_1i).dot(L1_t[i])
                N1_t_1i = (H_i_T).dot(H_i_T.T) / Upsilon_inf[i] + \
                        L1N0L0 + L1N0L0.T + (L0_t[i].T).dot(N1_t_1i).dot(L0_t[i])
                N0_t_1i = (L0_t[i].T).dot(N0_t_1i).dot(L0_t[i])

            # If Upsilon_inf == 0
            else:
                r0_t_1i = (H_i_T).dot(d_t[i]) / Upsilon_star[i] + \
                        (L_star_t[i].T).dot(r0_t_1i)
                N0_t_1i = (H_i_T).dot(H_i_T.T) / Upsilon_star[i] + \
                        (L_star_t[i].T).dot(N0_t_1i).dot(L_star_t[i])
                N1_t_1i = (L_star_t[i].T).dot(N1_t_1i).dot(L_star_t[i])

        # Update smoothed xi and P
        xi_t_1 = self.xi_t[t][0]
        P_inf_t1 = self.P_inf_t[t][0]
        P_star_t1 = self.P_star_t[t][0]
        xi_t_T = xi_t_1 + P_star_t1.dot(r0_t_1i) + \
                P_inf_t1.dot(r1_t_1i)
        P_inf_N1 = P_inf_t1.dot(N1_t_1i) 
        P_inf_N1_P_star = P_inf_N1.dot(P_star_t1)
        P_star_N0 = P_star_t1.dot(N0_t_1i)
        P_star_N0_P_star = P_star_N0.dot(P_star_t1)
        P_inf_N2 = P_inf_t1.dot(N2_t_1i)
        P_inf_N2_P_inf = P_inf_N2.dot(P_inf_t1)
        P_star_N1 = P_star_t1.dot(N1_t_1i)
        P_t_T = get_nearest_PSD(P_star_t1 - P_inf_N1_P_star - \
                P_inf_N1_P_star.T - P_star_N0_P_star - P_inf_N2_P_inf)
        self.xi_t_T[t] = xi_t_T
        self.P_t_T[t] = P_t_T

        # Update r and N for current period
        self.r0_t[t] = r0_t_1i
        self.r1_t[t] = r1_t_1i
        self.N0_t[t] = N0_t_1i
        self.N1_t[t] = N1_t_1i
        self.N2_t[t] = N2_t_1i
        
        if t > 0:
            # Update P_1t_t_T
            Pcov = self.P_star_t[t-1][self.n_t[t-1]].dot(self.Ft[t-1].T).dot(
                    self.I - P_inf_N1.T - P_star_N0.T) - \
                    self.P_inf_t[t-1][self.n_t[t-1]].dot(self.Ft[t-1].T).dot(
                    P_inf_N2.T + P_star_N1.T)
            self.Pcov_t_t1[t-1] = Pcov
                        
            # Update r and N from t to t-1
            self.r0_t[t-1] = (self.Ft[t-1].T).dot(r0_t_1i)
            self.r1_t[t-1] = (self.Ft[t-1].T).dot(r1_t_1i)
            self.N0_t[t-1] = (self.Ft[t-1].T).dot(N0_t_1i).dot(self.Ft[t-1])
            self.N1_t[t-1] = (self.Ft[t-1].T).dot(N1_t_1i).dot(self.Ft[t-1])
            self.N2_t[t-1] = (self.Ft[t-1].T).dot(N2_t_1i).dot(self.Ft[t-1])


    def _E_delta2(self, Mt: List[np.ndarray], t: int) -> np.ndarray:
        """
        Calculated expected value of delta2. See doc/manual.pdf for details.

        Parameters:
        ----------
        Mt : system matrix from ft(theta)  # Note: not theta_i
        t : time index

        Returns:
        ----------
        delta2 : expectation term for xi in G
        """
        # For initial state, use xi_1_0 and P_1_0 instead
        if t == 0:
            delta = self.xi_t_T[t] - Mt['xi_1_0']
            delta2 = delta.dot(delta.T) + self.P_t_T[t]

        # For other state, use formular derived in doc/manual.pdf Appendix E
        else:
            delta = self.xi_t_T[t] - Mt['Ft'][t-1].dot(self.xi_t_T[t-1]) - \
                    Mt['Bt'][t-1].dot(self.Xt[t-1])
            FPcov = Mt['Ft'][t-1].dot(self.Pcov_t_t1[t-1])  # Pcov_t_t1 starts at t=1
            delta2 = delta.dot(delta.T) + self.P_t_T[t] - FPcov - FPcov.T + \
                    Mt['Ft'][t-1].dot(self.P_t_T[t-1]).dot(Mt['Ft'][t-1].T)

        return delta2


    def _E_chi2(self, Mt: List[np.ndarray], t: int) -> np.ndarray:
        """
        Calculate expected value of chi2. See doc/manual.pdf for details.

        Parameters:
        ----------
        Mt : system matrix from ft(theta)  # Note: not theta_i
        t : time index

        Returns:
        ----------
        chi2 : expectation term for y in G
        """
        n_t = self.n_t[t]
        R_t = self.Rt[t]
        y_t = self.Yt[t]

        # Ht and Dt from Mt that is parameterized by theta
        H_t_M = permute(Mt['Ht'][t], self.partition_index[t], 
                axis='row')[0:n_t]
        D_t_M = permute(Mt['Dt'][t], self.partition_index[t], 
                axis='row')[0:n_t]
        chi = y_t[0:n_t] - H_t_M.dot(self.xi_t_T[t]) - \
                D_t_M.dot(self.Xt[t])
        chi2 = chi.dot(chi.T) + H_t_M.dot(
                self.P_t_T[t]).dot(H_t_M.T)
        return chi2


    def G(self, theta) -> float:
        """
        Calculate G

        Parameters:
        ----------
        theta : parameters to feed in self.ft

        Returns:
        G : objective value for EM algorithms
        """
        Mt = self.ft(theta, self.T, **self.ft_kwargs)
        G1 = 0
        G2 = 0
       
        # Raise Error if the smoother object is not run first
        if not self.is_smoothed:
            raise ValueError('Smoother is not fitted.')

        for t in range(self.T):
            if t == 0:
                _, A, Pi, P_star_1 = get_init_mat(Mt['P_1_0'])
                G1 += np.log(pdet(P_star_1)) + scipy.trace(inv(
                    P_star_1).dot(self._E_delta2(Mt, t))) 
            else:
                G1 += np.log(pdet(Mt['Qt'][t-1])) + scipy.trace(inv(
                    Mt['Qt'][t-1]).dot(self._E_delta2(Mt, t)))
            
            if self.n_t[t] > 0:    
                # Sort Rt index
                R_t = permute(Mt['Rt'][t], self.partition_index[t], 
                        axis='both')[0:self.n_t[t], 0:self.n_t[t]]
                G2 += np.log(pdet(R_t)) + scipy.trace(inv(
                        R_t).dot(self._E_chi2(Mt, t)))
        
        G = G1 + G2

        # Only use marginal correction if non-explosive diffuse
        if (not self.explosive) and self.t_q > 0:
            G -= np.log(pdet(LL_correct(Mt['Ht'], Mt['Ft'],
                    self.n_t, A, index=self.partition_index)))

        return -G.item()


    def get_smoothed_val(self, is_xi: bool=True, xi_col: List[int]=None) \
            -> List[np.ndarray]:
        """
        Generated smoothed xi. If state is diffuse,
        no covariance for Yt. Use xi_col to include 
        only the important xi. 

        Parameters:
        ----------
        is_xi : whether returns xi values
        xi_col : column index of xi to be included. 

        Returns:
        ----------
        xi_t_T : smoothed state means
        P_t_T : smoothed state covariances
        """
        # Raise error if not fitted yet
        if not self.is_smoothed:
            raise TypeError('The Kalman smoother object is not fitted yet')

        # If xi_col is not specified, use all columns
        if xi_col is None:
            xi_col = list(range(self.xi_length))

        Yt_smoothed = preallocate(self.T)
        Yt_smoothed_cov = preallocate(self.T)
        xi_t_T = preallocate(self.T)
        P_t_T = preallocate(self.T)

        for t in range(self.T):
            # Get smoothed mean and cov of y_t
            Yt_smoothed[t], Yt_smoothed_cov[t] = \
                    self.get_smoothed_y(t)

            # Get xi if needed
            if is_xi:
                xi_t_T[t] = self.xi_t_T[t][xi_col]
                P_t_T[t] = self.P_t_T[t][xi_col][:, xi_col]
            
        return Yt_smoothed, Yt_smoothed_cov, xi_t_T, P_t_T

    
    def get_smoothed_y(self, t: int) -> Tuple[np.ndarray]:
        """
        Get smoothed mean and variance for y_t

        Parameters:
        ----------
        t : time index

        Returns:
        ----------
        y_t_T : smoothed y_t
        y_cov_T : covariance matrix for smoothed y_t
        """
        # If no missing, use y_t and 0
        if self.n_t[t] == self.y_length:
            y_t_T = self.Yt[t]
            y_cov_T = np.zeros([self.y_length, self.y_length])
 
        # if all missing, simple calculation from xi_t_T and P_t_T
        elif self.n_t[t] == 0: 
            y_t_T = self.Ht[t].dot(self.xi_t_T[t]) + \
                    self.Dt[t].dot(self.Xt[t])
            y_cov_T = self.Ht[t].dot(self.P_t_T[t]).dot(
                    self.Ht[t].T) + self.Rt[t]

        # If partially missing, need to rearrange the measurements
        else: 
            n_t = self.n_t[t]
            y1_t_T = self.Yt[t][0:n_t].copy()
            y_cov_T_permute = np.zeros([self.y_length, self.y_length])

            R_22 = self.Rt[t][n_t:][:, n_t:]
            R_21 = self.Rt[t][n_t:][:, :n_t]
            R_11 = self.Rt[t][:n_t][:, :n_t]
            B = R_21.dot(inv(R_11))
            H1 = self.Ht[t][:n_t]
            H2 = self.Ht[t][n_t:]
            D1 = self.Dt[t][:n_t]
            D2 = self.Dt[t][n_t:]

            # Get mean
            epsilon = y1_t_T - H1.dot(self.xi_t_T[t]) - D1.dot(self.Xt[t])
            y2_t_T = H2.dot(self.xi_t_T[t]) + D2.dot(self.Xt[t]) + B.dot(epsilon)
            y_t_T_permute = np.concatenate((y1_t_T, y2_t_T), axis=0) 

            # Get covariance
            R22_1 = R_22 - R_21.dot(inv(R_11)).dot(R_21.T)
            diff_H2 = H2 - B.dot(H1)
            y2_cov_T = diff_H2.dot(self.P_t_T[t]).dot(diff_H2.T) + \
                    R22_1
            y_cov_T_permute[n_t:][:, n_t:] = y2_cov_T

            # Restore to the original index
            original_index = revert_permute(self.partition_index[t])
            y_t_T = permute(y_t_T_permute, original_index)
            y_cov_T = permute(y_cov_T_permute, original_index, 
                    axis='both')
        
        return y_t_T, y_cov_T


    def get_filtered_state(self, t: int) -> Dict:
        """
        Call Filter.get_filtered_state

        Parameters:
        ----------
        t : time index

        Returns:
        ----------
        state_val : state info at time t
        """
        state_val = Filter.get_filtered_state(self, t)
        return state_val

