import numpy as np
from typing import List, Callable, Tuple, Dict, Any
import scipy.linalg as linalg
from copy import deepcopy 
from .utils import mask_nan, LL_correct, validate_wrapper, \
        min_val, pdet, check_consistence, get_init_mat, \
        permute, partition_index, gen_Xt, preallocate

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
    assumptions on v_t and w_t, we are able to characterize the
    distribution of the BSTS model. Refer to doc/manual.pdf for details.
    """

    def __init__(self, ft: Callable, Yt: np.ndarray, Xt: np.ndarray=None, 
            for_smoother: bool=False, wrapper: Any=None, 
            reset_index: np.ndarray=None, **kwargs) -> None:
        """
        Initialize a Kalman Filter. Filter take system matrices 
        Mt returns characteristics of the BSTS model. Note that self.Qt
        correspond to Q_{t+1}.

        Parameters:
        ----------
        ft : function that generate Mt
        Yt : input Yt from self.fit()
        Xt : input Xt from self.fit()
        for_smoother : whether to store extra information for Kalman smoothers
        kwargs : kwargs for ft
        wrapper : wrapper of list of system matrices, determine whether 
            or not to carry out certain operations to boost performance.
            If not None, then it has to be a class object.
        reset_index : update wrapper reset attribute
        """
        self.ft = ft
        self.for_smoother = for_smoother
        self.is_filtered = False  # determine if the filter object has been fitted
        self.ft_kwargs = kwargs
        self.wrapper = validate_wrapper(wrapper)
        self.is_checked = False
        self.is_init = False  # if true then only reset to np.nan
        self.n_t = None
        self.is_missing = None
        self.partitioned_index = None

        # Create placeholders for other class attributes
        self.theta = None
        self.Ft = None
        self.Bt = None
        self.Ht = None
        self.Dt = None
        self.Qt = None
        self.Rt = None
        self.xi_length = None
        self.y_length = None
        self.Yt = deepcopy(Yt)
        self.Xt = deepcopy(Xt)
        if reset_index is None:
            self.reset_index = np.ones(Yt.shape[0], dtype=bool)
        else:
            self.reset_index = reset_index
        self.T = len(self.Yt)
        self.I = None
        self.P_star = None
        self.xi_T1 = None
        self.P_T1 = None
        self.explosive = False  # default to be non-explosive

        # Create output matrices
        self.xi_1_0 = None
        self.P_1_0 = None
        self.xi_t = None
        self.Upsilon_inf_t = None
        self.Upsilon_star_t = None
        self.d_t = None
        self.P_inf_t = None
        self.P_star_t = None
        self.Upsilon_inf_gt_0_t = None  # store whether Upsilon_{\inf} > 0
        self.Pi = None
        self.A = None
        self.q = None
        self.t_q = None

        # Create output matrices for smoothers
        if self.for_smoother:
            self.partitioned_index = None
            self.L0_t = None
            self.L1_t = None
            self.L_star_t = None
            self.Ht_tilde = None

    def init_attr(self, theta: np.ndarray, 
            init_state: Dict=None) -> None:
        """
        Initialize inputs to the Kalman filter. 
    
        Parameters:
        ----------
        theta : input parameters
        init_state : user-specified initial state 
        """
        # Initialize data inputs
        self.theta = theta
        
        # Generate Mt and Xt, and populate system matrices of the BSTS model
        M_ = self.ft(self.theta, T=1, **self.ft_kwargs)
        if self.Xt is None:
            self.Xt = gen_Xt(Xt=None, B=M_['Bt'][0], T=self.T)

        # Check consistence
        Mt = self.ft(self.theta, self.T, **self.ft_kwargs)
        check_consistence(Mt, self.Yt[0], self.Xt[0], 
                init_state=init_state, is_checked=self.is_checked)
        self.is_checked = True  # only check for the first time

        self.Ft = self.wrapper(Mt['Ft'], self.reset_index)
        self.Bt = self.wrapper(Mt['Bt'], self.reset_index)
        self.Ht = self.wrapper(Mt['Ht'], self.reset_index)
        self.Dt = self.wrapper(Mt['Dt'], self.reset_index)
        self.Qt = self.wrapper(Mt['Qt'], self.reset_index)
        self.Rt = self.wrapper(Mt['Rt'], self.reset_index)
        self.xi_length = self.Ft[0].shape[0]
        self.y_length = self.Ht[0].shape[0]
        self.I = np.eye(self.xi_length)
        self.t_q = 0

        # Create output matrices
        self.xi_1_0 = Mt['xi_1_0']
        self.P_1_0 = Mt['P_1_0']

        # Check whether BSTS has explosive states
        eig = linalg.eigvals(self.Ft[0])
        self.explosive = np.any(np.abs(eig) > 1 + min_val)

        # Collect initialization information
        self.q, self.A, self.Pi, self.P_star = \
                get_init_mat(self.P_1_0)

        # If init_state is provided, override
        if init_state is not None:
            self.q = init_state.get('q', self.q)
            self.A = init_state.get('P_inf_t', self.A)
            self.P_star = init_state.get('P_star_t')
            self.xi_1_0 = init_state.get('xi_t', self.xi_1_0)
            self.explosive = init_state.get('explosive', 
                    self.explosive)

        if not self.is_init:
            # Preprocess Yt if Yt has missing measurements, only do once
            self.is_missing = preallocate(*self.Yt.shape[0:2], arr_type='bool')
            self.n_t = preallocate(self.T, arr_type='int')
            self.partitioned_index = preallocate(*self.Yt.shape[0:2], arr_type='int')

            for t in range(self.T):
                self.is_missing[t] = np.isnan(self.Yt[t])[:,0]
                self.n_t[t] = (~self.is_missing[t]).sum()
            
                # Sort observed measurements to the top
                self.partitioned_index[t] = partition_index(self.is_missing[t])  
                if self.n_t[t] < self.y_length:
                    self.Yt[t] = mask_nan(self.is_missing[t], self.Yt[t], dim='row')
                    self.Yt[t] = permute(self.Yt[t], self.partitioned_index[t])

            # Initialize xi_1_0 and  P_1_0
            self.xi_t = preallocate(self.T, self.y_length + 1, 
                    self.xi_length, 1, default_val=np.nan)
            self.d_t = preallocate(self.T, self.y_length, 1, default_val=np.nan)
            self.Upsilon_star_t = preallocate(self.T, self.y_length, 1, 
                    default_val=np.nan)
            self.P_star_t = preallocate(self.T, self.y_length + 1, 
                    self.xi_length, self.xi_length, default_val=np.nan)

            # Use self.T not self.q + 1 so it 
            #     can handle different q in optimization.
            #     Not self.xi_length + 1 to account for 
            #     complete missing period
            self.P_inf_t = preallocate(self.T, self.y_length + 1,
                    self.xi_length, self.xi_length, default_val=np.nan)
            self.Upsilon_inf_t = preallocate(self.T, self.y_length, 
                    1, default_val=np.nan)
            self.Upsilon_inf_gt_0_t = preallocate(self.T, self.y_length, 
                    1, default_val=np.nan)

            if self.for_smoother:
                self.Ht_tilde = preallocate(self.T, self.y_length, self.xi_length,
                        default_val=np.nan)
                self.L_star_t = preallocate(self.T, self.y_length, 
                        self.xi_length, self.xi_length, default_val=np.nan)

                self.L0_t = preallocate(self.T, self.y_length, 
                        self.xi_length, self.xi_length, default_val=np.nan)
                self.L1_t = preallocate(self.T, self.y_length,
                        self.xi_length, self.xi_length, default_val=np.nan)
            self.is_init = True  # only initialize for the first time
        else:
            self.xi_t[:] = np.nan
            self.d_t[:] = np.nan
            self.Upsilon_star_t[:] = np.nan
            self.P_star_t[:] = np.nan 

            if self.q > 0:
                self.P_inf_t[:] = np.nan
                self.Upsilon_inf_t[:] = np.nan
                self.Upsilon_inf_gt_0_t[:] = np.nan

            if self.for_smoother:
                self.Ht_tilde[:] = np.nan
                self.L_star_t[:] = np.nan

                if self.q > 0:
                    self.L0_t[:] = np.nan
                    self.L1_t[:] = np.nan

        self.xi_t[0][0] = self.xi_1_0
        self.P_star_t[0][0] = self.P_star
        if self.q > 0:
            self.P_inf_t[0][0] = self.A


    def fit(self, theta: np.ndarray, init_state: Dict=None) -> None:
        """
        Run forward filtering, given input measurements and regressors

        Parameters:
        ----------
        theta : list of parameters for self.ft
        init_state : user-specified initial state values
        """
        # Initialize input data
        self.init_attr(theta, init_state=init_state)

        # Filter
        for t in range(self.T):
            if self.q > 0:
                self._sequential_update_diffuse(t)
            
            else:
                self._sequential_update(t)

        # Mark the object as fitted
        self.is_filtered = True


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
        Y_t, H_t, D_t, R_t, l_t, l_inv = self._LDL(t)

        # Skip missing measurements
        for i in range(1, self.n_t[t]+1):
            ob_index = i - 1  # state variables have one more value
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
        xi_t1_1 = self.Ft[t].dot(self.xi_t[t][self.n_t[t]]) + \
                self.Bt[t].dot(self.Xt[t])
        P_t1_1 = self.Ft[t].dot(self.P_star_t[t][self.n_t[t]]).dot(
                self.Ft[t].T) + self.Qt[t]

        if t < self.T - 1:
            self.xi_t[t+1][0] = xi_t1_1
            self.P_star_t[t+1][0] = P_t1_1
        else:
            self.xi_T1 = xi_t1_1
            self.P_T1 = P_t1_1

        if self.for_smoother:
            self.Ht_tilde[t] = H_t


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
        Y_t, H_t, D_t, R_t, l_t, l_inv = self._LDL(t)
        self.t_q += 1

        # Start sequential updating xi_{t:(i)}, 
        # P_inf_t_{t:(i)}, and P_star_t_{t:(i)}
        for i in range(1, self.n_t[t]+1):
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
                P_inf_i1 = P_inf
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
        xi_t1_1 = self.Ft[t].dot(self.xi_t[t][self.n_t[t]]) + \
                self.Bt[t].dot(self.Xt[t])
        P_inf_t1_1 = self.Ft[t].dot(
                self.P_inf_t[t][self.n_t[t]]).dot(self.Ft[t].T)
        P_star_t1_1 = self.Ft[t].dot(
                self.P_star_t[t][self.n_t[t]]).dot(self.Ft[t].T) + self.Qt[t]      

        # Update q at the end of time t
        self.q = min(self.q, np.linalg.matrix_rank(P_inf_t1_1))

        # Raise exception if we don't have enough data
        if t == self.T - 1:
            raise ValueError('Not enough data to handle diffuse priors')
        else:  # if no error raised, we are able to update the filter for t+1
            self.xi_t[t+1][0] = xi_t1_1
            self.P_inf_t[t+1][0] = P_inf_t1_1
            self.P_star_t[t+1][0] = P_star_t1_1 

        if self.for_smoother:
            self.Ht_tilde[t] = H_t


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
        Y_t : transformed Yt[t], NaN values are replaced with 0
        H_t : transformed Yt[t], rows that correspond to NaN 
            in Yt[t] are replaced with 0
        D_t : transformed Dt[t], rows that correspond to NaN 
            in Yt[t] are replaced with 0
        R_t : transformed Rt[t], rows and columns that correspond
            to NaN in Yt[t] are replaced with 0, then diagonalized
        L_t : l from ldl
        L_inv : l^{-1}, used for EM algorithms
        """
        if self.n_t[t] < self.y_length:
            self.Rt[t] = permute(self.Rt[t], self.partitioned_index[t], axis='both')
            self.Ht[t] = permute(self.Ht[t], self.partitioned_index[t])
            self.Dt[t] = permute(self.Dt[t], self.partitioned_index[t])

        # Diagonalize Y_t, H_t, and D_t
        L_t, R_t, L_inv = self.Rt.ldl(t)
        Y_t = L_inv.dot(self.Yt[t])
        H_t = L_inv.dot(self.Ht[t])
        D_t = L_inv.dot(self.Dt[t])
        
        return Y_t, H_t, D_t, R_t, L_t, L_inv


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

        # Add marginal correction term if not explosive
        if (not self.explosive) and self.t_q > 0:
            LL -= np.log(pdet(LL_correct(self.Ht, self.Ft, 
                self.n_t, self.A)))
        return -LL.item()


    def get_filtered_val(self, is_xi: bool=True, xi_col: List[int]=None) \
            -> List[np.ndarray]:
        """
        Generated filtered Yt. It will also generate
        filtered values for missing measurements. If
        state is diffusal, no covariance for Yt. Use 
        xi_col to include only the important xi. For 
        example, we don't need all xi in models that
        have weekly effects.

        Parameters:
        ----------
        is_xi : whether returns xi values
        xi_col : column index of xi to be included. 

        Returns:
        ----------
        Yt_filtered : filtered Yt
        Yt_filtered_cov : covariance of filtered Yt
        xi_t : selected xi_t
        P_t : selected P_t
        """
        # Raise error if not fitted yet
        if not self.is_filtered:
            raise TypeError('The Kalman filter object is not fitted yet')

        # If xi_col is not specified, use all columns
        if xi_col is None:
            xi_col = list(range(self.xi_length))
        xi_len = len(xi_col)
        
        Mt = self.ft(self.theta, self.T, **self.ft_kwargs)

        Yt_filtered = preallocate(self.T, self.y_length, 1)
        Yt_filtered_cov = preallocate(self.T, self.y_length, self.y_length)
        xi_t = preallocate(self.T, xi_len, 1)
        P_t = preallocate(self.T, xi_len, xi_len)

        for t in range(self.T):
            # Get filtered y_t
            Yt_filtered[t] = Mt['Ht'][t].dot(self.xi_t[t][0]) + \
                    Mt['Dt'][t].dot(self.Xt[t])

            # Get variance of filtered y_t
            if t >= self.t_q:
                Yt_filtered_cov[t] = Mt['Ht'][t].dot(self.P_star_t[t][0]).dot(
                        Mt['Ht'][t].T) + Mt['Rt'][t]
            else:
                Yt_filtered_cov[t] = np.nan * np.ones([self.y_length, self.y_length])

            # Get xi if needed
            if is_xi:
                xi_t[t] = self.xi_t[t][0][xi_col]
                if t >= self.t_q:
                    P_t[t] = self.P_star_t[t][0][xi_col][:, xi_col]
                else:
                    P_t[t] = np.nan * np.ones((xi_len, xi_len))
            else:
                xi_t = np.nan * np.ones(self.T)
                P_t = np.nan * np.ones(self.T)
            
        return Yt_filtered, Yt_filtered_cov, xi_t, P_t

    
    def get_filtered_state(self, t: int) -> Dict:
        """
        Get xi_t[t][0] and P_star_t[t][0]

        Parameters:
        ----------
        t : time index

        Returns:
        ----------
        state_val : state info at time t
        """
        if not self.is_filtered:
            raise ValueError('Kalman filter is not fitted yet')

        if (t >= self.t_q) and (t < self.T):
            xi_t = deepcopy(self.xi_t[t][0])
            P_star_t = deepcopy(self.P_star_t[t][0])
            P_inf_t = np.zeros(P_star_t.shape)
            q = 0
        
        # Successful training always ends with regular Kalman filters
        elif t == self.T:
            xi_t = deepcopy(self.xi_T1)
            P_star_t = deepcopy(self.P_T1)
            P_inf_t = np.zeros(P_star_t.shape)
            q = 0
        elif t > self.T:
            raise ValueError('Maximum t allowed is {}'.format(self.T))
        else:
            raise ValueError('Diffuse state at ' + \
                    'time {}'.format(t))
        
        state_val = {'xi_t': xi_t, 'P_star_t': P_star_t, 
                'P_inf_t': P_inf_t, 'q': q}
        return state_val
