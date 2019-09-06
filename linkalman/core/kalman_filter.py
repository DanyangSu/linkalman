import numpy as np
from typing import List, Any, Callable, Tuple
import scipy
from copy import deepcopy 
from .utils import mask_nan, inv, LL_correct, M_wrap, Constant_M, min_val, pdet

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

    def __init__(self, ft: Callable) -> None:
        """
        Initialize a Kalman Filter. Filter take system matrices 
        Mt returns characteristics of the BSTS model. Note that self.Qt
        correspond to Q_{t+1}.

        Parameters:
        ----------
        ft : function that generate Mt
        """
        self.ft = ft

        # Create placeholders for other class attributes
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

        # Create output matrices
        self.xi_1_0 = None
        self.P_1_0 = None
        self.xi_t = []
        self.Ht_tilde = []
        self.Yt_missing = []
        self.Upsilon_inf_t = []
        self.Upsilon_star_t = []
        self.d_t = []
        self.L0_t = []
        self.L1_t = []
        self.L_star_t = []
        self.P_inf_t = []
        self.P_star_t = []
        self.Upsilon_inf_gt_0_t = []  # store whether Upsilon_{\inf} > 0
        self.t_q = 0


    def get_selection_mat(self, P_1_0: np.ndarray) \
            -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Get information for diffuse initialization

        Parameters:
        ----------
        P_1_0 : initial state covariance. Diagonal value np.nan if diffuse

        Returns:
        ----------
        number_diffuse : number of diffuse state
        A : selection matrix for diffuse states
        Pi : selection matrix for stationary states
        """
        is_diffuse = np.isnan(self.P_1_0.diagonal())
        number_diffuse = np.count_nonzero(is_diffuse)
        A = np.diag(is_diffuse.astype(float))
        Pi = np.diag((~is_diffuse).astype(float))
        return number_diffuse, A, Pi


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
        self.Yt = deepcopy(Yt)
        self.T = len(self.Yt)

        # Generate Mt and populate system matrices of the BSTS model
        Mt = self.ft(theta, self.T)
        self.Ht_raw = deepcopy(M_wrap(Mt['Ht']))
        self.Dt_raw = deepcopy(M_wrap(Mt['Dt']))
        self.Rt_raw = deepcopy(M_wrap(Mt['Rt']))

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
        self.xi_1_0 = deepcopy(Mt['xi_1_0'])
        self.P_1_0 = deepcopy(Mt['P_1_0'])

        # Generate zeros arrays for Xt if Xt is None
        self.gen_Xt(Xt)
        
        # Check Mt and Xt, Yt dimension consistence
        self.check_consistence()
        
        # Collect initialization information
        self.q, self.A, self.Pi = self.get_selection_mat(Mt['P_1_0'])

        # Initialize xi_1_0 and  P_1_0
        self.xi_t.append([self.xi_1_0])
        self.d_t.append([])
        self.L_star_t.append([])
        self.Upsilon_star_t.append([])
        
        if self.q > 0:
            P_clean = deepcopy(self.P_1_0)
            P_clean[np.isnan(P_clean)] = 0
            self.P_inf_t.append([self.A.dot(self.A.T)])
            self.P_star_t.append([self.Pi.dot(P_clean).dot(self.Pi.T)])
            self.L0_t.append([])
            self.L1_t.append([])
            self.Upsilon_inf_t.append([])
            self.Upsilon_inf_gt_0_t.append([])
        else:
            self.P_star_t.append([self.P_1_0])

        # Filter
        for t in range(self.T):
            if self.q > 0:
                self._sequential_update_diffuse(t)
            
                # Raise exception if we don't have enough data
                if t == self.T - 1:
                    raise ValueError('Not enough data to handle diffuse priors')
            else:
                self._sequential_update(t)

        # Drop P_{T+1|T} and xi_{T+1|T}
        self.P_star_t.pop(-1)
        self.xi_t.pop(-1)


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
        Y_t, H_t, D_t, R_t, is_missing = self._LDL(t)
        self.Ht_tilde.append(H_t)
        self.Yt_missing.append(is_missing)
        self.t_q += 1

        # Start sequential updating xi_{t:(i)}, 
        # P_inf_t_{t:(i)}, and P_star_t_{t:(i)}
        for i in range(self.y_length):
            if is_missing[i]:
                continue
            else:
                P_inf = self.P_inf_t[t][-1]  # the most recent P
                P_star = self.P_star_t[t][-1]
                xi_i = self.xi_t[t][-1]
                H_i = H_t[i].reshape(1, -1)
                abs_Hi = np.abs(H_i)
                D_i = D_t[i].reshape(1, -1)
                sigma2 = R_t[i][i]
                Upsilon_inf = H_i.dot(P_inf).dot(H_i.T)
                Upsilon_star = H_i.dot(P_star).dot(H_i.T) + sigma2
                d_t_i = Y_t[i] - H_i.dot(xi_i) - D_i.dot(self.Xt[t])
                
                # If Upsilon_inf > 0
                gt_0 = Upsilon_inf > min_val * np.power(
                        abs_Hi[abs_Hi > min_val].min(), 2)
                if gt_0:
                    K_0 = P_inf.dot(H_i.T) / Upsilon_inf
                    K_1 = (P_star.dot(H_i.T) - K_0.dot(Upsilon_star)) / Upsilon_inf
                    xi_t_i1 = xi_i + K_0.dot(d_t_i)
                    L0_t_i = self.I - K_0.dot(H_i)
                    L1_t_i = - K_1.dot(H_i)
                    L_star_t_i = None
                    KRK = K_0.dot(sigma2).dot(K_0.T)
                    P_inf_i1 = self._joseph_form(L0_t_i, P_inf)
                    P_star_i1 = self._joseph_form(L0_t_i, P_star, KRK)

                    # update number of diffuse state
                    self.q = min(self.q - 1, np.linalg.matrix_rank(P_inf_i1))

                # If Upsilon_inf == 0
                else:
                    K_star = P_star.dot(H_i.T) / Upsilon_star
                    xi_t_i1 = xi_i + K_star.dot(d_t_i)
                    L_star_t_i = self.I - K_star.dot(H_i)
                    KRK = K_star.dot(sigma2).dot(K_star.T)
                    P_inf_i1 = deepcopy(P_inf)
                    P_star_i1 = self._joseph_form(L_star_t_i, P_star, KRK)
                    L0_t_i = None
                    L1_t_i = None

                self.xi_t[t].append(xi_t_i1)
                self.P_inf_t[t].append(P_inf_i1)
                self.P_star_t[t].append(P_star_i1)
                self.L0_t[t].append(L0_t_i)
                self.L1_t[t].append(L1_t_i)
                self.L_star_t[t].append(L_star_t_i)
                self.Upsilon_inf_t[t].append(Upsilon_inf)
                self.Upsilon_star_t[t].append(Upsilon_star)
                self.d_t[t].append(d_t_i)
                self.Upsilon_inf_gt_0_t[t].append(gt_0)

        # Calculate xi_t1_t, P_inf_t1_t, and P_star_t1_t, 
        # and add placeholders for others
        xi_t1_1 = self.Ft[t].dot(self.xi_t[t][-1]) + \
                self.Bt[t].dot(self.Xt[t])
        P_inf_t1_1 = self.Ft[t].dot(
                self.P_inf_t[t][-1]).dot(self.Ft[t].T)
        P_star_t1_1 = self.Ft[t].dot(
                self.P_star_t[t][-1]).dot(self.Ft[t].T) + self.Qt[t]

        self.xi_t.append([xi_t1_1])
        self.P_inf_t.append([P_inf_t1_1])
        self.P_star_t.append([P_star_t1_1])
        self.Upsilon_inf_t.append([])
        self.Upsilon_star_t.append([])
        self.L0_t.append([])
        self.L1_t.append([])
        self.L_star_t.append([])
        self.d_t.append([])
        self.Upsilon_inf_gt_0_t.append([])


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
        Y_t, H_t, D_t, R_t, is_missing = self._LDL(t)
        self.Ht_tilde.append(H_t)
        self.Yt_missing.append(is_missing)

        # Skip missing measurements
        for i in range(self.y_length):
            if is_missing[i]:
                continue
            else:
                xi_i = self.xi_t[t][-1]
                P_i = self.P_star_t[t][-1]
                H_i = H_t[i].reshape(1, -1)
                D_i = D_t[i].reshape(1, -1)
                sigma2 = R_t[i][i]
                Upsilon = H_i.dot(P_i).dot(H_i.T) + sigma2
                d_t_i = Y_t[i] - H_i.dot(xi_i) - D_i.dot(self.Xt[t])
                K = P_i.dot(H_i.T) / Upsilon
                xi_t_i1 = xi_i + K.dot(d_t_i)
                L_t_i = self.I - K.dot(H_i)
                KRK = K.dot(sigma2).dot(K.T)
                P_i1 = self._joseph_form(L_t_i, P_i, KRK)

            self.xi_t[t].append(xi_t_i1)
            self.P_star_t[t].append(P_i1)
            self.L_star_t[t].append(L_t_i)
            self.Upsilon_star_t[t].append(Upsilon)
            self.d_t[t].append(d_t_i)

        # Calculate xi_t1_t and P_t, and add placeholders for others
        xi_t1_1 = self.Ft[t].dot(self.xi_t[t][-1]) + \
                self.Bt[t].dot(self.Xt[t])
        P_t1_1 = self.Ft[t].dot(self.P_star_t[t][-1]).dot(
                self.Ft[t].T) + self.Qt[t]
        self.xi_t.append([xi_t1_1])
        self.P_star_t.append([P_t1_1])
        self.L_star_t.append([])
        self.Upsilon_star_t.append([])
        self.d_t.append([])


    def _LDL(self, t: int) -> Tuple[np.ndarray]: 
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
        is_missing : indicator whether elements in Yt[t] are missing
        """
        # Preprocess Rt and Yt if Yt has missing measurements
        is_missing = np.hstack(np.isnan(self.Yt[t]))
        if np.any(is_missing):
            self.Yt[t] = mask_nan(is_missing, self.Yt[t])
            self.Rt[t] = mask_nan(is_missing, self.Rt[t])
            self.Ht[t] = mask_nan(is_missing, self.Ht[t], dim='row')
            self.Dt[t] = mask_nan(is_missing, self.Dt[t], dim='row')

        # Diagonalize Y_t, H_t, and D_t
        L_t, R_t, L_inv = self.Rt.ldl(t)
        Y_t = L_inv.dot(self.Yt[t])
        H_t = L_inv.dot(self.Ht[t])
        D_t = L_inv.dot(self.Dt[t])
        return Y_t, H_t, D_t, R_t, is_missing


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
        Yt_filtered = []
        Yt_filtered_cov = []
        for t in range(self.T):
            # Get filtered y_t
            yt_f = self.Ht_raw[t].dot(self.xi_t[t][0]) + \
                    self.Dt_raw[t].dot(self.Xt[t])
            Yt_filtered.append(yt_f)

            # Get standard error of filtered y_t
            if t < self.t_q:
                yt_error = None
            else:
                yt_error = self.Ht_raw[t].dot(self.P_star_t[t][0]).dot(
                        self.Ht_raw[t].T) + self.Rt_raw[t]
            
            Yt_filtered_cov.append(yt_error)
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
                for i in range(len(Upsilon_inf)):
                    if gt_0[i]:
                        LL += np.log(Upsilon_inf[i])
                    else:
                        LL += np.log(Upsilon_star[i]) + (
                                d_t[i].T).dot(d_t[i]) / Upsilon_star[i]

            # If not diffuse, skip Psi_t
            else:
                for i in range(len(Upsilon_star)):
                    LL += np.log(Upsilon_star[i]) + (
                            (d_t[i].T).dot(d_t[i]) / Upsilon_star[i])
        
        # Add marginal correction term
        LL -= np.log(pdet(LL_correct(self.Ht_tilde, self.Ft, self.A)))
        return -np.asscalar(LL)
