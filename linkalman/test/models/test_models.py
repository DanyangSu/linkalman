import numpy as np
import pandas as pd
import linkalman
import scipy
from linkalman.models import BaseConstantModel as BCM
import pytest


# Test whether ft_kwargs flow through
def test_ft_kwargs_flow(scipy_solver, f_ar1, df_Y):
    """
    Test whether first value of filtered var
    """
    model = BCM()
    model.set_f(f_ar1, xi_1_0 = np.array([[10]]), 
            P_1_0 = np.array([[4]]))
    model.set_solver(scipy_solver)
    theta = np.array([1.1, 0.5, 0.1, 0.3])
    y_col = list(df_Y.columns)
    theta_init = theta.copy()
    model.fit(df_Y, theta_init, y_col=y_col)
    df_out = model.predict(df_Y)
    
    expected_xi_1_0 = 10
    expected_P_1_0 = 4
    
    assert expected_xi_1_0 == df_out.loc[0, 'xi0_filtered'] 
    assert expected_P_1_0 == df_out.loc[0, 'P0_filtered']


# Test model.simulated_data
def test_simulated_data_no_theta(scipy_solver, f_ar1):
    """
    Test error when no theta
    """
    model = BCM()
    model.set_f(f_ar1, xi_1_0 = np.array([[10]]), 
            P_1_0 = np.array([[4]]))
    model.set_solver(scipy_solver)
    with pytest.raises(ValueError) as error:
        model.simulated_data()
    expected_result = 'Model needs theta'
    result = str(error.value)
    assert result == expected_result


def test_simulated_data_init_state(scipy_solver, f_ar1):
    """
    Test if init_state value works for simulated_data
    """
    init_state = {'xi_t': 100 * np.ones([1, 1]), 
            'P_star_t': np.zeros([1, 1])}

    theta = 0.1 * np.ones(4)
    model = BCM()
    model.set_f(f_ar1, xi_1_0 = np.array([[10]]), 
            P_1_0 = np.array([[4]]))
    model.set_solver(scipy_solver)
    df, _, _ = model.simulated_data(theta, T=4, init_state=init_state)
    result = np.array([df.loc[0, ['xi_0']]]).T
    expected_result = 100 * np.ones([1, 1])
    np.testing.assert_array_equal(result, expected_result)


def test_simulated_data_no_ft(scipy_solver):
    """
    Test error when no ft
    """
    model = BCM()
    model.set_solver(scipy_solver)
    with pytest.raises(ValueError) as error:
        model.simulated_data(np.array([2]))
    expected_result = 'Model needs ft'
    result = str(error.value)
    assert result == expected_result


# Test set_f
def test_set_f_reset_things(scipy_solver, f_ar1, df_Y):
    """
    Test if provide a new ft, reset  
    """
    model = BCM()
    model.set_f(f_ar1, xi_1_0 = np.array([[10]]), 
            P_1_0 = np.array([[4]]))
    assert model.theta_opt is None
    
    # Fit model, self.theta_opt should have values
    model.set_solver(scipy_solver)
    theta = np.array([1.1, 0.5, 0.1, 0.3])
    y_col = list(df_Y.columns)
    theta_init = theta.copy()
    model.fit(df_Y, theta_init, y_col=y_col)
    assert model.theta_opt is not None

    # Reset for set_f
    model.set_f(f_ar1, xi_1_0 = np.array([[10]]), 
            P_1_0 = np.array([[4]]))
    assert model.theta_opt is None


def test_set_f_no_reset(scipy_solver, f_ar1, df_Y):
    """
    Test if provide a new ft, reset  
    """
    model = BCM()
    model.set_f(f_ar1, xi_1_0 = np.array([[10]]), 
            P_1_0 = np.array([[4]]))
    assert model.theta_opt is None
    
    # Fit model, self.theta_opt should have values
    model.set_solver(scipy_solver)
    theta = np.array([1.1, 0.5, 0.1, 0.3])
    y_col = list(df_Y.columns)
    theta_init = theta.copy()
    model.fit(df_Y, theta_init, y_col=y_col)
    assert model.theta_opt is not None

    # No reset for set_f
    model.set_f(f_ar1, xi_1_0 = np.array([[10]]), 
            P_1_0 = np.array([[4]]), reset=False)
    assert model.theta_opt is not None


# Test init_state values override ft arguments
def test_init_override(scipy_solver, f_ar1, df_Y):
    init_pred = {'P_star_t': 9 * np.ones([1, 1]), 
            'xi_t': 60 * np.ones([1, 1]), 'q': 0}
    init_fit = {'P_star_t': 3 * np.ones([1, 1]), 'xi_t': 80 * np.ones([1, 1])}
    model = BCM()
    model.set_f(f_ar1, xi_1_0 = np.array([[10]]), 
            P_1_0 = np.array([[4]]))
    model.set_solver(scipy_solver)
    theta = np.array([1.1, 0.5, 0.1, 0.3])
    y_col = list(df_Y.columns)
    theta_init = theta.copy()
    model.fit(df_Y, theta_init, y_col=y_col, init_state=init_fit)
    df_out = model.predict(df_Y, init_state=init_pred)
    ks = model.ks_fitted
    
    # Test fit values
    np.testing.assert_array_equal(ks.xi_t[0][0], np.array([[80]]))
    np.testing.assert_array_equal(ks.P_star_t[0][0], np.array([[3]]))

    # Test predict values
    expected_xi_1_0 = 60
    expected_P_1_0 = 9
    assert expected_xi_1_0 == df_out.loc[0, 'xi0_filtered'] 
    assert expected_P_1_0 == df_out.loc[0, 'P0_filtered']


# Test if breakppoint mechanism works
def test_break_point(scipy_solver, f_ar1, df_Y):
    
    model = BCM()
    model.set_f(f_ar1, xi_1_0 = np.array([[10]]), 
            P_1_0 = np.array([[4]]))
    model.set_solver(scipy_solver)
    theta = np.array([1.1, 0.5, 0.1, 0.3])
    y_col = list(df_Y.columns)
    theta_init = theta.copy()

    # split data in to trian and test
    len_df = df_Y.shape[0]
    cutoff = len_df // 2
    df_train = df_Y.loc[df_Y.index < cutoff].copy()
    df_test = df_Y.loc[df_Y.index >= cutoff].copy()

    # Train model
    model.fit(df_train, theta_init, y_col=y_col)
    df_out1 = model.predict(df_Y)
    df_out2 = model.predict_t(df_test, t_index=-1)

    result1 = df_out1.loc[len_df - 1, 'xi0_filtered']
    result2 = df_out2.loc[len_df - 1, 'xi0_filtered']
    
    assert result1 == result2
    
