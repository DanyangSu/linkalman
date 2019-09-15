import numpy as np
from typing import List, Any, Callable, Tuple
import scipy
from copy import deepcopy 
from .utils import mask_nan, inv, LL_correct, M_wrap, Constant_M, \
        min_val, pdet, check_consistence, get_init_mat, permute, \
        partition_index, gen_Xt, preallocate

__all__ = ['Filter']


class Filter(object):
    """
    Given a BSTS model:

    xi_{t+1} = F_t * xi_t + B_t * x_t + v_t     (v_t ~ N(0, Qt))
    y_t = H_t * xi_t + D_t * x_t + w_t     (w_t ~ N(0, Rt))

    and initial conditions:

    xi_1_0 = E(xi_1) 
    P_1_0 = Cov(xi_1) (P_1_0 may contain diffuse parts)

    We want to solve:

    xi_t_{t-1} = E(xi_t|Info(t-1))
    P_t_{t-1} = Cov(xi_t|Info(t-1))

    where Info(t) is the information set at time t. With Gaussian
    asusmptions on v_t and w_t, we are able to characterize the
    distribution of the BSTS model. Refer to doc/theory.pdf for details.
    """

    def __init__(self, ft: Callable, for_smoother: bool=False) -> None:
        """
        Initialize a Kalman Filter. Filter take system matrices 
        Mt returns characteristics of the BSTS model. Note that self.Qt
        correspond to Q_{t+1}.

        Parameters:
        ----------
        ft : function that generate Mt
        for_smoother : whether to store extra information for Kalman smoothers
        """
        self.ft = ft
        self.for_smoother = for_smoother
        self.is_filtered = False  # determine if the filter object has been fitted

        # Create placeholders for other class attributes
        self.theta = None
        self.Ht_raw = None
        self.Dt_raw = None
        self.Rt_raw = None
        self.Ft = None
        self.Bt = None
        self.Ht = None
        self.Dt = None
        self.Qt = None
        self.Rt = None
        self.xi_length = None
        self.y_length = None
        self.Yt = None
        self.Xt = None
        self.T = None
        self.I = None
        self.P_star = None

        # Create output matrices
        self.xi_1_0 = None
        self.P_1_0 = None
        self.xi_t = None
        self.Ht_tilde = None
        self.Yt_missing = None
        self.Upsilon_inf_t = None
        self.Upsilon_star_t = None
        self.d_t = None
        self.L0_t = None
        self.L1_t = None
        self.L_star_t = None
        self.P_inf_t = None
        self.P_star_t = None
        self.Upsilon_inf_gt_0_t = None  # store whether Upsilon_{\inf} > 0
        self.t_q = 0

        # Create output matrices for smoothers
        if self.for_smoother:
            self.l_t = None  # l from ldl
            self.l_t_inv = None
            self.n_t = None
            self.partition_index = None


    def init_attr(self, theta: List, Yt: List[np.ndarray], 
            Xt: List[np.ndarray]=None) -> None:
        """
        Initialize inputs to the Kalman filter. 
    
        Parameters:
        ----------
        theta : input parameters
        Yt : input Yt from self.fit()
        Xt : input Xt from self.fit()
        """
        # Initialize data inputs
        self.theta = theta
        self.Yt = deepcopy(Yt)
        self.T = len(self.Yt)

        # Generate Mt and Xt, and populate system matrices of the BSTS model
        Mt = self.ft(self.theta, self.T)
        self.Xt = gen_Xt(Xt=Xt, B=Mt['Bt'][0], T=self.T)

        # Check consistence
        check_consistence(Mt, self.Yt[0], self.Xt[0])

        self.Ft = M_wrap(Mt['Ft'])
        self.Bt = M_wrap(Mt['Bt'])
        self.Ht = M_wrap(Mt['Ht'])
        self.Dt = M_wrap(Mt['Dt'])
        self.Qt = M_wrap(Mt['Qt'])
        self.Rt = M_wrap(Mt['Rt'])
        self.xi_length = self.Ft[0].shape[0]
        self.y_length = self.Ht[0].shape[0]
        self.I = np.eye(self.xi_length)

        # Create output matrices
        self.xi_1_0 = Mt['xi_1_0']
        self.P_1_0 = Mt['P_1_0']

        # Collect initialization information
        self.q, self.A, self.Pi, self.P_star = get_init_mat(self.P_1_0)

        # Initialize xi_1_0 and  P_1_0
        self.xi_t = preallocate(self.T, self.y_length + 1)
        self.xi_t[0][0] = self.xi_1_0
        self.d_t = preallocate(self.T, self.y_length)
        self.Upsilon_star_t = preallocate(self.T, self.y_length)
        self.P_star_t = preallocate(self.T, self.y_length + 1)
        self.P_star_t[0][0] = self.P_star
        
        if self.q > 0:
            self.P_inf_t = preallocate(self.T, self.y_length + 1)
            self.P_inf_t[0][0] = self.A
            self.Upsilon_inf_t = preallocate(self.T, self.y_length)
            self.Upsilon_inf_gt_0_t = preallocate(self.T, self.y_length)

        if self.for_smoother:
            self.l_t = preallocate(self.T)
            self.l_t_inv = preallocate(self.T)
            self.n_t = preallocate(self.T)
            self.partition_index = preallocate(self.T)
            self.L_star_t = preallocate(self.T, self.y_length)

            if self.q > 0:
                self.L0_t = preallocate(self.T, self.y_length)
                self.L1_t = preallocate(self.T, self.y_length)


    def fit(self, theta: np.ndarray, Yt: List[np.ndarray], 
            Xt: List[np.ndarray]=None) -> None:
        """
        Run forward filtering, given input measurements and regressors

        Parameters:
        ----------
        theta : list of parameters for self.ft
        Yt : measurements, may contain missing values
        Xt : regressors, must be deterministic and has no missing values.
            If set as None, will generate zero vectors
        """
        # Initialize input data
        self.init_attr(theta, Yt, Xt)

        # Filter
        for t in range(self.T):
            if self.q > 0:
                self._sequential_update_diffuse(t)
            
            else:
                self._sequential_update(t)

        # Mark the object as fitted
        self.is_filtered = True


    def _sequential_update_diffuse(self, t: int) -> None:
        """
        Sequentially update diffuse kalman filters at time t. If Q_t 
        is not diagonal, we first transform it to a diagonal
        matrix using LDL transformation. Also update q_t. 
        All outputs are lists of numpy arrays. 
        
        Parameters:
        ----------
        t : time index
        """
        # LDL 
        n_t, Y_t, H_t, D_t, R_t, l_t, l_inv, partitioned_index = self._LDL(t)
        self.t_q += 1
        self.n_t[t] = n_t

        # Start sequential updating xi_{t:(i)}, 
        # P_inf_t_{t:(i)}, and P_star_t_{t:(i)}
        for i in range(1, n_t+1):
            ob_index = i - 1  # y index starts from 0 not 1
            P_inf = self.P_inf_t[t][ob_index]  # the most recent P
            P_star = self.P_star_t[t][ob_index]
            xi_i = self.xi_t[t][ob_index]
            H_i = H_t[ob_index:i]
            abs_Hi = np.abs(H_i)
            D_i = D_t[ob_index:i]
            sigma2 = R_t[ob_index][ob_index]
            Upsilon_inf = H_i.dot(P_inf).dot(H_i.T)
            Upsilon_star = H_i.dot(P_star).dot(H_i.T) + sigma2
            d_t_i = Y_t[ob_index] - H_i.dot(xi_i) - D_i.dot(self.Xt[t])    

            # If Upsilon_inf > 0
            gt_0 = Upsilon_inf > min_val * np.power(
                    abs_Hi[abs_Hi > min_val].min(), 2)
            if gt_0:
                K_0 = P_inf.dot(H_i.T) / Upsilon_inf
                K_1 = (P_star.dot(H_i.T) - K_0.dot(Upsilon_star)) / Upsilon_inf
                xi_t_i1 = xi_i + K_0.dot(d_t_i)
                L0_t_i = self.I - K_0.dot(H_i)
                L1_t_i = - K_1.dot(H_i)
                KRK = K_0.dot(sigma2).dot(K_0.T)
                P_inf_i1 = self._joseph_form(L0_t_i, P_inf)
                P_star_i1 = self._joseph_form(L0_t_i, P_star, KRK)

                if self.for_smoother:
                    self.L0_t[t][ob_index] = L0_t_i
                    self.L1_t[t][ob_index] = L1_t_i

                # update number of diffuse state
                self.q = self.q - 1

            # If Upsilon_inf == 0
            else:
                K_star = P_star.dot(H_i.T) / Upsilon_star
                xi_t_i1 = xi_i + K_star.dot(d_t_i)
                L_star_t_i = self.I - K_star.dot(H_i)
                KRK = K_star.dot(sigma2).dot(K_star.T)
                P_inf_i1 = deepcopy(P_inf)
                P_star_i1 = self._joseph_form(L_star_t_i, P_star, KRK)
                
                if self.for_smoother:
                    self.L_star_t[t][ob_index] = L_star_t_i 

            self.xi_t[t][i] = xi_t_i1
            self.P_inf_t[t][i] = P_inf_i1
            self.P_star_t[t][i] = P_star_i1
            self.Upsilon_inf_t[t][ob_index] = Upsilon_inf
            self.Upsilon_star_t[t][ob_index] = Upsilon_star
            self.d_t[t][ob_index] = d_t_i
            self.Upsilon_inf_gt_0_t[t][ob_index] = gt_0

        # Calculate xi_t1_t, P_inf_t1_t, and P_star_t1_t, 
        # and add placeholders for others
        xi_t1_1 = self.Ft[t].dot(self.xi_t[t][n_t]) + \
                self.Bt[t].dot(self.Xt[t])
        P_inf_t1_1 = self.Ft[t].dot(
                self.P_inf_t[t][n_t]).dot(self.Ft[t].T)
        P_star_t1_1 = self.Ft[t].dot(
                self.P_star_t[t][n_t]).dot(self.Ft[t].T) + self.Qt[t]      

        # Raise exception if we don't have enough data
        if t == self.T - 1:
            raise ValueError('Not enough data to handle diffuse priors')
        else:  # if no error raised, we are able to update the filter for t+1
            self.xi_t[t+1][0] = xi_t1_1
            self.P_inf_t[t+1][0] = P_inf_t1_1
            self.P_star_t[t+1][0] = P_star_t1_1 

        if self.for_smoother:
            self.l_t[t] = l_t  # l from ldl
            self.l_t_inv[t] = l_inv
            self.n_t[t] = n_t
            self.partition_index[t] = partitioned_index


    def _sequential_update(self, t: int) -> None:
        """
        Sequentially update Kalman Filter at time t. If Q_t
        is not diagonal, we first transform it to a diagonal
        matrix using LDL transformation. 

        Parameters:
        ----------
        t : time index
        """
        # LDL 
        n_t, Y_t, H_t, D_t, R_t, l_t, l_inv, partitioned_index = self._LDL(t)
        self.n_t[t] = n_t

        # Skip missing measurements
        for i in range(1, n_t+1):
            ob_index = i - 1
            xi_i = self.xi_t[t][ob_index]
            P_i = self.P_star_t[t][ob_index]
            H_i = H_t[ob_index:i]
            D_i = D_t[ob_index:i]
            sigma2 = R_t[ob_index][ob_index]
            Upsilon = H_i.dot(P_i).dot(H_i.T) + sigma2
            d_t_i = Y_t[ob_index] - H_i.dot(xi_i) - D_i.dot(self.Xt[t])
            K = P_i.dot(H_i.T) / Upsilon
            xi_t_i1 = xi_i + K.dot(d_t_i)
            L_t_i = self.I - K.dot(H_i)
            KRK = K.dot(sigma2).dot(K.T)
            P_i1 = self._joseph_form(L_t_i, P_i, KRK)

            self.xi_t[t][i] = xi_t_i1
            self.P_star_t[t][i] = P_i1
            self.Upsilon_star_t[t][ob_index] = Upsilon
            self.d_t[t][ob_index] = d_t_i

            if self.for_smoother:
                self.L_star_t[t][ob_index] = L_t_i

        # Calculate xi_t1_t and P_t, and add placeholders for others
        xi_t1_1 = self.Ft[t].dot(self.xi_t[t][n_t]) + \
                self.Bt[t].dot(self.Xt[t])
        P_t1_1 = self.Ft[t].dot(self.P_star_t[t][n_t]).dot(
                self.Ft[t].T) + self.Qt[t]

        if t < self.T - 1:
            self.xi_t[t+1][0] = xi_t1_1
            self.P_star_t[t+1][0] = P_t1_1

        if self.for_smoother:
            self.l_t[t] = l_t  # l from ldl
            self.l_t_inv[t] = l_inv
            self.partition_index[t] = partitioned_index


    def _joseph_form(self, L: np.ndarray, P: np.ndarray, 
            KRK: np.ndarray=None) -> np.ndarray:
        """
        Update P using Joseph Form, which guarantees 
        PSD of the covariance matrix. 

        Parameters: 
        ----------
        L : (I - K * d) 
        H : Hi
        P : state Covariance, can be either regular or diffuse
        KRK : for P_star or regular P, KRK is part of updating

        Returns:
        ----------
        P_1 : filtered P
        """
        if KRK is None:
            P_1 = L.dot(P).dot(L.T)
        else:
            P_1 = L.dot(P).dot(L.T) + KRK
        return P_1


    def _LDL(self, t: int) -> Tuple[int, np.ndarray, np.ndarray, \
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        """
        Transform the BSTS model using LDL methods. 

        Parameters:
        ----------
        t : time index of the BSTS system

        Returns:
        ----------
        n_t : number of observed measurement at time t
        Y_t : transformed Yt[t], NaN values are replaced with 0
        H_t : transformed Yt[t], rows that correspond to NaN 
            in Yt[t] are replaced with 0
        D_t : transformed Dt[t], rows that correspond to NaN 
            in Yt[t] are replaced with 0
        R_t : transformed Rt[t], rows and columns that correspond
            to NaN in Yt[t] are replaced with 0, then diagonalized
        L_t : l from ldl
        L_inv : l^{-1}, used for EM algorithms
        partitioned_index : sorted index, used for EM algorithms
        """
        # Preprocess Rt and Yt if Yt has missing measurements
        is_missing = np.hstack(np.isnan(self.Yt[t]))
        n_t = (~is_missing).sum()

        # Sort observed measurements to the top
        partitioned_index = partition_index(is_missing)  

        if np.any(is_missing):
            self.Yt[t] = mask_nan(is_missing, self.Yt[t], dim='row')
            self.Yt[t] = permute(self.Yt[t], partitioned_index)
            self.Rt[t] = permute(self.Rt[t], partitioned_index, axis='both')
            self.Ht[t] = permute(self.Ht[t], partitioned_index)
            self.Dt[t] = permute(self.Dt[t], partitioned_index)

        # Diagonalize Y_t, H_t, and D_t
        L_t, R_t, L_inv = self.Rt.ldl(t)
        Y_t = L_inv.dot(self.Yt[t])
        H_t = L_inv.dot(self.Ht[t])
        D_t = L_inv.dot(self.Dt[t])
        
        return n_t, Y_t, H_t, D_t, R_t, L_t, L_inv, partitioned_index


    def get_filtered_y(self) -> List[np.ndarray]:
        """
        Generated filtered Yt. It will also generate
        filtered values for missing measurements. If
        state is diffusal, no covariance for Yt

        Returns:
        ----------
        Yt_filtered : filtered Yt
        Yt_filtered_cov : standard error of filtered Yt
        """
        # Raise error if not fitted yet
        if not self.is_filtered:
            raise TypeError('The Kalman filter object is not fitted yet')
        
        Mt = self.ft(self.theta, self.T)

        Yt_filtered = preallocate(self.T)
        Yt_filtered_cov = preallocate(self.T)

        for t in range(self.T):
            # Get filtered y_t
            yt_f = Mt['Ht'][t].dot(self.xi_t[t][0]) + \
                    Mt['Dt'][t].dot(self.Xt[t])
            Yt_filtered[t] = yt_f

            # Get standard error of filtered y_t
            if t >= self.t_q:
                yt_error_var = Mt['Ht'][t].dot(self.P_star_t[t][0]).dot(
                        Mt['Ht'][t].T) + Mt['Rt'][t]
                Yt_filtered_cov[t] = yt_error_var
            
        return Yt_filtered, Yt_filtered_cov


    def get_LL(self) -> float:
        """
        Calculate the marginal likelihood of Yt for a BSTS model.

        Returns:
        ----------
        LL : marginal likelihood 
        """
        LL = 0
        for t in range(self.T):
            Upsilon_star = self.Upsilon_star_t[t]
            d_t = self.d_t[t]

            # If diffuse, use Psi_t
            if t < self.t_q:
                gt_0 = self.Upsilon_inf_gt_0_t[t]
                Upsilon_inf = self.Upsilon_inf_t[t]
                for i in range(self.n_t[t]):
                    if gt_0[i]:
                        LL += np.log(Upsilon_inf[i])
                    else:
                        LL += np.log(Upsilon_star[i]) + (
                                d_t[i].T).dot(d_t[i]) / Upsilon_star[i]

            # If not diffuse, skip Psi_t
            else:
                for i in range(self.n_t[t]):
                    LL += np.log(Upsilon_star[i]) + (
                            d_t[i].T).dot(d_t[i]) / Upsilon_star[i]
        
        # Add marginal correction term
        LL -= np.log(pdet(LL_correct(self.Ht, self.Ft, self.A)))
        return -np.asscalar(LL)
