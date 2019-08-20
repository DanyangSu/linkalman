import numpy as np
from typing import List, Any, Callable, Tuple
from copy import deepcopy
from scipy import linalg
from .utils import inv, get_nearest_PSD, min_val
from . import Filter

__all__ = ['Smoother']


class Smoother(object):
    """
    Given a filtered object, Smoother returns smoothed state estimation.
    Given an HMM:

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
    characterize the distribution of the HMM based on the full 
    information set up to T. Refer to doc/theory.pdf for details.
    """

    def __init__(self) -> None:
        """
        Initialize a Kalman Smoother. self.delta2 and self.chi2 
        are used for EM algorithms later.
        """
        self.xi_t_T = []
        self.P_t_T = []
        self.xi2_t_T = []
        self.Pcov_1t_t = []
        self.delta2 = []
        self.chi2 = []
        self.r0_t = []
        self.r1_t = []
        self.N0_t = []
        self.N1_t = []
        self.N2_t = []
        

    def __call__(self, kf: Filter) -> None:
        """
        Run backward smoothing. 

        Parameters: 
        ----------
        kf : a Filter instance
        """
        # Include filtered results
        self.__dict__.update(kf.__dict__)

        # Initiate r and N
        self.r0_t.append(np.zeros([self.xi_length, 1]))
        self.r1_t.append(np.zeros([self.xi_length, 1]))
        self.N0_t.append(np.zeros([self.xi_length, self.xi_length]))
        self.N1_t.append(np.zeros([self.xi_length, self.xi_length]))
        self.N2_t.append(np.zeros([self.xi_length, self.xi_length]))

        # Start backward smoothing
        for t in reversed(range(self.T)):
            if t >= self.t_q:
                self._sequential_smooth(t)
            else:
                self._sequential_smooth_diffuse(t)

        # Reverse the order of t to restore chronological order
        self.xi_t_T = list(reversed(self.xi_t_T))
        self.P_t_T = list(reversed(self.P_t_T))
        self.Pcov_1t_t = list(reversed(self.Pcov_1t_t))
        self.r0_t = list(reversed(self.r0_t))
        self.r1_t = list(reversed(self.r1_t))
        self.N0_t = list(reversed(self.N0_t))
        self.N1_t = list(reversed(self.N1_t))
        self.N2_t = list(reversed(self.N2_t))
            

    def _sequential_smooth(self, t: int) -> None:
        """
        Update Kalman Smoother at time t. Refer to doc/theory.pdf 
        for details on the notation of each variables.

        Parameters:
        ----------
        t : time index
        """
        H_t = self.Ht_tilda[t]
        d_t = self.d_t[t]
        Upsilon = self.Upsilon_star_t[t]
        L_t = self.L_star_t[t]
        is_missing = self.Yt_missing[t]

        # Backwards iteration on r and N
        counter = self.y_length - is_missing.sum() - 1 
        r_t_1i = deepcopy(self.r0_t[-1])
        N_t_1i = deepcopy(self.N0_t[-1])
        
        for i in reversed(range(self.y_length)):
            if is_missing[i]: 
                continue
            else:
                H_i = H_t[i].reshape(1, -1)
                r_t_1i = (H_i.T).dot(d_t[counter]) / Upsilon[counter] + \
                        (L_t[counter].T).dot(r_t_1i)
                N_t_1i = (H_i.T).dot(H_i) / Upsilon[counter] + \
                        (L_t[counter].T).dot(N_t_1i).dot(L_t[counter])
                counter -= 1

        # Update smoothed xi, P and Pcov
        xi_t_1 = self.xi_t[t][0]
        P_t_1 = self.P_star_t[t][0]
        xi_t_T = xi_t_1 + P_t_1.dot(r_t_1i)
        P_t_T = get_nearest_PSD(P_t_1 - P_t_1.dot(N_t_1i).dot(P_t_1))
        self.xi_t_T.append(xi_t_T)
        self.P_t_T.append(P_t_T)

        # Update r and N for current period
        self.r0_t[-1] = deepcopy(r_t_1i)
        self.N0_t[-1] = deepcopy(N_t_1i)
        
        # Update r, N and P_1t_t_T from t to t-1
        if t > 0:
            Pcov = self.P_star_t[t-1][-1].dot(self.Ft[t-1].T).dot(
                self.I - N_t_1i.dot(P_t_1))
            self.Pcov_1t_t.append(Pcov)
            self.r0_t.append((self.Ft[t-1].T).dot(r_t_1i))
            self.N0_t.append((self.Ft[t-1].T).dot(N_t_1i).dot(self.Ft[t-1]))


    def _sequential_smooth_diffuse(self, t: int) -> None:
        """
        Update diffuse Kalman Smoother at time t. Refer to doc/theory.pdf 
        for details on the notation of each variables.

        Parameters:
        ----------
        t : time index
        """
        H_t = self.Ht_tilda[t]
        d_t = self.d_t[t]
        Upsilon_inf = self.Upsilon_inf_t[t]
        Upsilon_star = self.Upsilon_star_t[t]
        L0_t = self.L0_t[t]
        L1_t = self.L1_t[t]
        L_star_t = self.L_star_t[t]
        is_missing = self.Yt_missing[t]

        # Backwards iteration on r and N
        counter = self.y_length - is_missing.sum() - 1 
        r0_t_1i = deepcopy(self.r0_t[-1])
        r1_t_1i = deepcopy(self.r1_t[-1])
        N0_t_1i = deepcopy(self.N0_t[-1])
        N1_t_1i = deepcopy(self.N1_t[-1])
        N2_t_1i = deepcopy(self.N2_t[-1])
        
        for i in reversed(range(self.y_length)):
            if is_missing[i]: 
                continue
            else:
                H_i = H_t[i].reshape(1, -1)
                abs_Hi = np.abs(H_i)

                # If Upsilon_inf > 0
                if Upsilon_inf[counter] > min_val * np.power(
                        abs_Hi[abs_Hi > min_val].min(), 2):
                    
                    # Must update r1 first, bc it uses r0_t_1i
                    r1_t_1i = (H_i.T).dot(d_t[counter]) / Upsilon_inf[counter] + \
                            (L1_t[counter].T).dot(r0_t_1i) + (L0_t[counter].T).dot(r1_t_1i)
                    r0_t_1i = (L0_t[counter].T).dot(r0_t_1i)

                    # Order of updating: N2->N1->N0
                    L1N1L0 = (L1_t[counter].T).dot(N1_t_1i).dot(L0_t[counter])
                    L1N0L0 = (L1_t[counter].T).dot(N0_t_1i).dot(L0_t[counter])
                    N2_t_1i = -(H_i.T).dot(H_i) * Upsilon_star[counter] / \
                            np.power(Upsilon_inf[counter], 2) + L1N1L0 + L1N1L0.T + \
                            (L0_t[counter].T).dot(N2_t_1i).dot(L0_t[counter]) + \
                            (L1_t[counter].T).dot(N0_t_1i).dot(L1_t[counter])
                    N1_t_1i = (H_i.T).dot(H_i) / Upsilon_inf[counter] + \
                            L1N0L0 + L1N0L0.T + (L0_t[counter].T).dot(N1_t_1i).dot(L0_t[counter])
                    N0_t_1i = (L0_t[counter].T).dot(N0_t_1i).dot(L0_t[counter])

                # If Upsilon_inf == 0
                else:
                    r0_t_1i = (H_i.T).dot(d_t[counter]) / Upsilon_star[counter] + \
                            (L_star_t[counter].T).dot(r0_t_1i)
                    N0_t_1i = (H_i.T).dot(H_i) / Upsilon_star[counter] + \
                            (L_star_t[counter].T).dot(N0_t_1i).dot(L_star_t[counter])
                    N1_t_1i = (L_star_t[counter].T).dot(N1_t_1i).dot(L_star_t[counter])
                counter -= 1

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
        self.xi_t_T.append(xi_t_T)
        self.P_t_T.append(P_t_T)

        # Update r and N for current period
        self.r0_t[-1] = deepcopy(r0_t_1i)
        self.r1_t[-1] = deepcopy(r1_t_1i)
        self.N0_t[-1] = deepcopy(N0_t_1i)
        self.N1_t[-1] = deepcopy(N1_t_1i)
        self.N2_t[-1] = deepcopy(N2_t_1i)
        
        if t > 0:
            # Update P_1t_t_T
            Pcov = self.P_star_t[t-1][-1].dot(self.Ft[t-1].T).dot(
                    self.I - P_inf_N1.T - P_star_N0.T) - self.P_inf_t[t-1][-1].dot(
                    self.Ft[t-1].T).dot(P_inf_N2.T + P_star_N1.T)
            self.Pcov_1t_t.append(Pcov)
                        
            # Update r and N from t to t-1
            self.r0_t.append((self.Ft[t-1].T).dot(r0_t_1i))
            self.r1_t.append((self.Ft[t-1].T).dot(r1_t_1i))
            self.N0_t.append((self.Ft[t-1].T).dot(N0_t_1i).dot(self.Ft[t-1]))
            self.N1_t.append((self.Ft[t-1].T).dot(N1_t_1i).dot(self.Ft[t-1]))
            self.N2_t.append((self.Ft[t-1].T).dot(N2_t_1i).dot(self.Ft[t-1]))


    def _E_delta2(self, t: int) -> np.ndarray:
        """
        Calculated expected value of delta2. See Appendix E 
        in doc/theory.pdf for details. 

        Parameters:
        ----------
        t : time index

        Returns:
        ----------
        delta2 : expectation term for xi in MLE
        """
        # For initial state, use xi_1_0 and P_1_0 instead
        if t == 0:
            term2 = self.xi_t_1t[t].dot(self.xi_t_T[t].T)
            delta2 = self.xi2_t_T[t] - term2 - term2.T + \
                    self.xi_t_1t[t].dot(self.xi_t_1t[t].T)

        # For other state, use formular derived in doc/theory.pdf Appendix E
        else:
            Bx = self.Bt[t-1].dot(self.Xt[t-1])
            term3 = Bx.dot(self.xi_t_T[t].T)
            term4 = self.xi_t_xi_1t_T[t].dot(self.Ft[t-1].T)
            term5 = self.Ft[t-1].dot(self.xi2_t_T[t-1]).dot(self.Ft[t-1].T)
            term6 = Bx.dot(self.xi_t_T[t-1].T).dot(self.Ft[t-1].T)
            delta2 = self.xi2_t_T[t] - term4.T - term3 - term4 + term5 + \
                    term6 - term3.T + term6.T + Bx.dot(Bx.T)
        return delta2


    def _E_chi2(self, t: int) -> np.ndarray:
        """
        Calculate expected value of chi2. See Appendix F 
        in doc/theory.pdf for details.

        Parameters:
        ----------
        t : time index

        Returns:
        ----------
        chi2 : expectation term for y in MLE
        """
        Dx = self.Dt[t].dot(self.Xt[t])
        term1 = (self.Yt[t] - Dx).dot((self.Yt[t] - Dx).T)
        term2 = self.Ht[t].dot(self.xi_t_T[t]).dot((self.Yt[t] - Dx).T)
        term4 = self.Ht[t].dot(self.xi2_t_T[t]).dot(self.Ht[t].T)
        chi2 = term1 - term2 - term2.T + term4
        return chi2


    def get_smoothed_y(self) -> List[np.ndarray]:
        """
        Generated smoothed Yt. It will also generate
        smoothed values for missing measurements.

        Returns:
        ----------
        Yt_smoothed : smoothed Yt
        Yt_smoothed_cov : standard error of smoothed Yt
        """
        Yt_smoothed = []
        Yt_smoothed_cov = []
        for t in range(self.T):
            # Get smoothed y_t
            yt_s = self.Ht_raw[t].dot(self.xi_t_T[t]) + \
                    self.Dt_raw[t].dot(self.Xt[t])
            Yt_smoothed.append(yt_s)

            # Get standard error of smoothed y_t
            yt_error = self.Ht_raw[t].dot(self.P_t_T[t]).dot(
                    self.Ht_raw[t].T) + self.Rt_raw[t]
            Yt_smoothed_cov.append(yt_error)
        return Yt_smoothed, Yt_smoothed_cov

