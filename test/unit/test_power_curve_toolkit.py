import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

from openoa.utils import power_curve
from test.conftest import reset_prng
from openoa.utils.power_curve.parametric_forms import logistic5param, logistic5param_capped


noise = 0.1


@pytest.fixture
def generic_power_curve():
    reset_prng()
    params = [1300, -7, 11, 2, 0.5]
    x = pd.Series(np.random.random(100) * 30)
    y = pd.Series(logistic5param(x, *params) + np.random.random(100) * noise)
    return x, y


@pytest.fixture
def nrel_15mw_power_curve():
    # power curve source: https://github.com/NREL/turbine-models/blob/master/Offshore/2020ATB_NREL_Reference_15MW_240.csv
    nrel_15mw_wind = pd.Series(np.arange(4, 26))
    nrel_15mw_power = pd.Series(
        np.array(
            [
                720,
                1239,
                2271,
                3817,
                5876,
                8450,
                11536,
                15000,
                15000,
                15000,
                15000,
                15000,
                15000,
                15000,
                15000,
                15000,
                15000,
                15000,
                15000,
                15000,
                15000,
                1500,
            ]
        )
    )
    return nrel_15mw_wind, nrel_15mw_power


def test_IEC(generic_power_curve):
    # Create test data using logistic5param form
    x, y = generic_power_curve
    curve = power_curve.IEC(x, y)
    y_pred = curve(x)
    # Does the IEC power curve match the test data?
    npt.assert_allclose(
        y_pred, y, rtol=1, atol=noise * 2, err_msg="Power curve did not properly fit."
    )


def test_IEC_with_bounds(nrel_15mw_power_curve):
    # Create the power curve with bounds at 4m/s adn 25m/s and bin width from power curve of 1m/s
    nrel_15mw_wind, nrel_15mw_power = nrel_15mw_power_curve
    cut_in = 4
    cut_out = 25
    curve = power_curve.IEC(
        nrel_15mw_wind,
        nrel_15mw_power,
        windspeed_start=cut_in,
        windspeed_end=cut_out,
        bin_width=1,
    )

    # Create the test data
    test_windspeeds = np.arange(0, 31)
    test_power = curve(test_windspeeds)

    # Test all windspeeds outside of cut-in and cut-out windspeeds produce no power
    should_be_zeros = test_power[(test_windspeeds < cut_in) | (test_windspeeds > cut_out)]
    npt.assert_array_equal(should_be_zeros, np.zeros(should_be_zeros.shape))

    # Test all the valid windspeeds are equal
    valid_power = test_power[(test_windspeeds >= cut_in) & (test_windspeeds <= cut_out)]
    npt.assert_array_equal(nrel_15mw_power, valid_power)


def test_logistic_5_param(generic_power_curve):
    x, y = generic_power_curve
    # Create test data using logistic5param form
    curve = power_curve.logistic_5_parametric(x, y)
    y_pred = curve(x)
    # Does the logistic-5 power curve match the test data?
    npt.assert_allclose(
        y_pred, y, rtol=1, atol=noise * 2, err_msg="Power curve did not properly fit."
    )


def test_gam(generic_power_curve):
    x, y = generic_power_curve
    # Create test data using logistic5param form
    curve = power_curve.gam(windspeed_col=x, power_col=y, n_splines=20)
    y_pred = curve(x)
    # Does the spline-fit power curve match the test data?
    npt.assert_allclose(y_pred, y, rtol=0.05, atol=20, err_msg="Power curve did not properly fit.")


def test_3paramgam(generic_power_curve):
    x, y = generic_power_curve
    # Create test data using logistic5param form
    winddir = pd.Series(np.random.random(100), name="winddir")
    airdens = pd.Series(np.random.random(100), name="airdens")
    curve = power_curve.gam_3param(
        windspeed_col=x,
        wind_direction_col=winddir,
        air_density_col=airdens,
        power_col=y,
        n_splines=20,
    )
    y_pred = curve(x, winddir, airdens)
    # Does the spline-fit power curve match the test data?
    npt.assert_allclose(y_pred, y, rtol=0.05, atol=20, err_msg="Power curve did not properly fit.")


def test_logistic5parameter():
    y_pred = logistic5param(np.array([1.0, 2.0, 3.0]), *[1300.0, -7.0, 11.0, 2.0, 0.5])
    y = np.array([2.29403585, 5.32662505, 15.74992462])
    npt.assert_allclose(y_pred, y, err_msg="Power curve did not properly fit.")

    y_pred = logistic5param(np.array([1, 2, 3]), *[1300.0, -7.0, 11.0, 2.0, 0.5])
    y = np.array([2.29403585, 5.32662505, 15.74992462])
    npt.assert_allclose(y_pred, y, err_msg="Power curve did not handle integer inputs properly.")

    y_pred = logistic5param(np.array([0.01, 0.0]), 1300, 7, 11, 2, 0.5)
    y = np.array([1300.0, 1300.0])
    npt.assert_allclose(y_pred, y, err_msg="Power curve did not handle zero properly (b>0).")

    y_pred = logistic5param(np.array([0.01, 0.0]), 1300, -7, 11, 2, 0.5)
    y = np.array([2.0, 2.0])
    npt.assert_allclose(y_pred, y, err_msg="Power curve did not handle zero properly (b<0).")


def test_logistic5parameter_capped():
    # Numpy array + Lower Bound
    y_pred = logistic5param_capped(
        np.array([1.0, 2.0, 3.0]), *[1300.0, -7.0, 11.0, 2.0, 0.5], lower=5.0, upper=20.0
    )
    y = np.array([5.0, 5.32662505, 15.74992462])
    npt.assert_allclose(y_pred, y, err_msg="Power curve did not properly fit.")

    # Numpy array + Upper and Lower Bound
    y_pred = logistic5param_capped(
        np.array([1.0, 2.0, 3.0]), *[1300.0, -7.0, 11.0, 2.0, 0.5], lower=5.0, upper=10.0
    )
    y = np.array([5.0, 5.32662505, 10.0])
    npt.assert_allclose(y_pred, y, err_msg="Power curve did not properly fit.")

    # Pandas Series + Upper and Lower Bound
    y_pred = logistic5param_capped(
        pd.Series([1.0, 2.0, 3.0]), *[1300.0, -7.0, 11.0, 2.0, 0.5], lower=5.0, upper=20.0
    )
    y = pd.Series([5.0, 5.32662505, 15.74992462])
    npt.assert_allclose(y_pred, y, err_msg="Power curve did not properly fit.")

    # Pandas Series + Upper and Lower Bound
    y_pred = logistic5param_capped(
        pd.Series([1.0, 2.0, 3.0]), *[1300.0, -7.0, 11.0, 2.0, 0.5], lower=5.0, upper=10.0
    )
    y = pd.Series([5.0, 5.32662505, 10.0])
    npt.assert_allclose(y_pred, y, err_msg="Power curve did not properly fit.")
