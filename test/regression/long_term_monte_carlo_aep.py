import copy

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

from openoa.analysis import MonteCarloAEP


from test.conftest import reset_prng  # isort: skip


@pytest.fixture(scope="module")
def aep_project(project):
    """Load and prepare the ENGIE plant data for each regression test."""
    project = project(None)
    return project


@pytest.fixture(scope="module")
def reanalysis_aep_project(aep_project):
    """Load and prepare the ENGIE plant data for each regression test."""
    project = copy.deepcopy(aep_project)
    project.reanalysis["merra2"] = project.reanalysis["merra2"].loc[:"2019-04-15 12:30"]
    project.reanalysis["era5"] = project.reanalysis["era5"].loc["1999-01-15 12:00":]
    return project


def test_monthly_inputs(subtests, aep_project):
    """Test inputs to the regression model, at monthly time resolution."""
    reset_prng()

    analysis = MonteCarloAEP(
        aep_project,
        reanalysis_products=["merra2", "era5"],
        time_resolution="MS",
        reg_temperature=True,
        reg_wind_direction=True,
    )
    df = analysis.aggregate
    df_rean = analysis._reanalysis_aggregate

    # Check the pre-processing functions
    # Energy Nan flags are all zero
    with subtests.test("Check pre-processing results"):
        npt.assert_array_equal(df["energy_nan_perc"].values, np.repeat(0.0, df.shape[0]))

        # Expected number of days per month are equal to number of actual days
        npt.assert_array_equal(df["num_days_expected"], df["num_days_actual"])

        # Check a few energy values
        expected_gwh = pd.Series([0.692400, 1.471730, 0.580035])
        actual_gwh = df.loc[
            pd.to_datetime(["2014-06-01", "2014-12-01", "2015-10-01"]), "energy_gwh"
        ]
        npt.assert_array_almost_equal(actual_gwh, expected_gwh)

    with subtests.test("Check availability and curtailment"):
        # Availablity, curtailment nan fields both 0, NaN flag is all False
        npt.assert_array_equal(df["avail_nan_perc"].values, np.repeat(0.0, df.shape[0]))
        npt.assert_array_equal(df["curt_nan_perc"].values, np.repeat(0.0, df.shape[0]))
        npt.assert_array_equal(df["nan_flag"].values, np.repeat(False, df.shape[0]))  # noqa: FBT003

        # Check a few reported availabilty and curtailment values
        expected_avail_gwh = pd.Series([0.029417, 0.021005, 0.000444])
        expected_curt_gwh = pd.Series([0.013250, 0.000000, 0.000000])
        expected_avail_pct = pd.Series([0.040019, 0.014071, 0.000765])
        expected_curt_pct = pd.Series([0.018026, 0.000000, 0.000000])

        date_ind = pd.to_datetime(["2014-06-01", "2014-12-01", "2015-10-01"])

        npt.assert_array_almost_equal(df.loc[date_ind, "availability_gwh"], expected_avail_gwh)
        npt.assert_array_almost_equal(df.loc[date_ind, "curtailment_gwh"], expected_curt_gwh)
        npt.assert_array_almost_equal(df.loc[date_ind, "availability_pct"], expected_avail_pct)
        npt.assert_array_almost_equal(df.loc[date_ind, "curtailment_pct"], expected_curt_pct)

    with subtests.test("Check reanalysis data"):
        # check that date range is truncated correctly
        assert (df_rean.index[0], df_rean.index[-1]) == (
            pd.to_datetime("1999-01-01 00:00:00"),
            pd.to_datetime("2019-04-01 00:00:00"),
        )

        expected = {
            "merra2": [5.42523278, 6.86883337, 5.02690892],
            "era5": [5.20508049, 6.71586744, 5.23824611],
            "merra2_WMETR_HorWdDir": [11.74700241, 250.90081133, 123.70142025],
            "era5_WMETR_HorWdDir": [23.4291153, 253.14150601, 121.25886916],
            "merra2_WMETR_EnvTmp": [289.87128364, 275.26493716, 281.72562887],
            "era5_WMETR_EnvTmp": [290.82110632, 276.62490053, 282.71629935],
        }
        date_ind = pd.to_datetime(["2014-06-01", "2014-12-01", "2015-10-01"])
        computed = {key: df.loc[date_ind, key].to_numpy() for key in expected}
        for key in expected:
            npt.assert_array_almost_equal(computed[key], expected[key])


def test_reanalysis_aggregate_monthly(reanalysis_aep_project):
    """Test reanalysis start and end dates depending on time resolution and end date argument."""
    reset_prng()
    # ____________________________________________________________________
    # Test default aggregate reanalysis values and date range, at monthly time resolution
    analysis = MonteCarloAEP(
        reanalysis_aep_project, reanalysis_products=["merra2", "era5"], time_resolution="MS"
    )
    df_rean = analysis._reanalysis_aggregate

    # check that date range is truncated correctly
    assert (df_rean.index[0], df_rean.index[-1]) == (
        pd.to_datetime("1999-02-01"),
        pd.to_datetime("2019-03-01"),
    )

    # Check wind speed values at start and end dates
    expected = {"merra2": [7.584891, 8.679547], "era5": [7.241081, 8.188632]}
    computed = {
        key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy() for key in expected
    }

    for key in expected:
        npt.assert_array_almost_equal(computed[key], expected[key])

    # ____________________________________________________________________
    # Check for invalid user-defined end dates
    # Date range doesn't include full 20 years
    with pytest.raises(ValueError):
        MonteCarloAEP(
            reanalysis_aep_project,
            reanalysis_products=["merra2", "era5"],
            time_resolution="MS",
            end_date_lt="2018-12-15 12:00",
        )

    # End date out of bounds, monthly
    with pytest.raises(ValueError):
        MonteCarloAEP(
            reanalysis_aep_project,
            reanalysis_products=["merra2", "era5"],
            time_resolution="MS",
            end_date_lt="2019-04-15 13:00",
        )

    # ____________________________________________________________________
    # Test aggregate reanalysis values and date range with user-defined end date, at monthly time resolution
    analysis = MonteCarloAEP(
        reanalysis_aep_project,
        reanalysis_products=["merra2", "era5"],
        time_resolution="MS",
        end_date_lt="2019-02-10 12:00",
    )
    df_rean = analysis._reanalysis_aggregate

    # check that date range is truncated correctly
    assert (df_rean.index[0], df_rean.index[-1]) == (
        pd.to_datetime("1999-02-01"),
        pd.to_datetime("2019-02-01"),
    )

    # Check wind speed values at start and end dates
    expected = {"merra2": [7.584891, 6.529796], "era5": [7.241081, 6.644804]}
    computed = {
        key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy() for key in expected
    }

    for key in expected:
        npt.assert_array_almost_equal(computed[key], expected[key])


def test_reanalysis_aggregate_daily(reanalysis_aep_project):
    reset_prng()
    # ____________________________________________________________________
    # Test default aggregate reanalysis values and date range, at daily time resolution
    analysis = MonteCarloAEP(
        reanalysis_aep_project, reanalysis_products=["merra2", "era5"], time_resolution="D"
    )
    df_rean = analysis._reanalysis_aggregate

    # check that date range is truncated correctly
    assert (df_rean.index[0], df_rean.index[-1]) == (
        pd.to_datetime("1999-01-16"),
        pd.to_datetime("2019-03-31"),
    )

    # Check wind speed values at start and end dates
    expected = {"merra2": [12.868168, 5.152958], "era5": [12.461761, 5.238968]}
    computed = {
        key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy() for key in expected
    }

    for key in expected:
        npt.assert_array_almost_equal(computed[key], expected[key])

    # ____________________________________________________________________
    # Check for invalid user-defined end dates
    # Date range doesn't include full 20 years
    with pytest.raises(ValueError):
        MonteCarloAEP(
            reanalysis_aep_project,
            reanalysis_products=["merra2", "era5"],
            time_resolution="D",
            end_date_lt="2019-01-14 23:00",
        )

    # End date out of bounds, daily
    with pytest.raises(ValueError):
        MonteCarloAEP(
            reanalysis_aep_project,
            reanalysis_products=["merra2", "era5"],
            time_resolution="D",
            end_date_lt="2019-04-15 13:00",
        )

    # ____________________________________________________________________
    # Test aggregate reanalysis values and date range with user-defined end date, at daily time resolution
    analysis = MonteCarloAEP(
        reanalysis_aep_project,
        reanalysis_products=["merra2", "era5"],
        time_resolution="D",
        end_date_lt="2019-02-10 12:00",
    )
    df_rean = analysis._reanalysis_aggregate

    # check that date range is truncated correctly
    assert (df_rean.index[0], df_rean.index[-1]) == (
        pd.to_datetime("1999-01-16"),
        pd.to_datetime("2019-02-10"),
    )

    # Check wind speed values at start and end dates
    expected = {"merra2": [12.868168, 14.571084], "era5": [12.461761, 14.045798]}
    computed = {
        key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy() for key in expected
    }

    for key in expected:
        npt.assert_array_almost_equal(computed[key], expected[key])


def test_reanalysis_aggregate_hourly(reanalysis_aep_project):
    reset_prng()
    # ____________________________________________________________________
    # Test default aggregate reanalysis values and date range, at hourly time resolution
    analysis = MonteCarloAEP(
        reanalysis_aep_project, reanalysis_products=["merra2", "era5"], time_resolution="h"
    )
    df_rean = analysis._reanalysis_aggregate

    # check that date range is truncated correctly
    assert (df_rean.index[0], df_rean.index[-1]) == (
        pd.to_datetime("1999-01-15 12:00"),
        pd.to_datetime("2019-03-31 23:00"),
    )

    # Check wind speed values at start and end dates
    expected = {"merra2": [10.509840, 9.096710], "era5": [9.202639, 9.486806]}
    computed = {
        key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy() for key in expected
    }

    for key in expected:
        npt.assert_array_almost_equal(computed[key], expected[key])

    # ____________________________________________________________________
    # Check for invalid user-defined end dates
    # Date range doesn't include full 20 years
    with pytest.raises(ValueError):
        MonteCarloAEP(
            reanalysis_aep_project,
            reanalysis_products=["merra2", "era5"],
            time_resolution="h",
            end_date_lt="2019-01-15 10:00",
        )

    # End date out of bounds, hourly
    with pytest.raises(ValueError):
        MonteCarloAEP(
            reanalysis_aep_project,
            reanalysis_products=["merra2", "era5"],
            time_resolution="h",
            end_date_lt="2019-04-15 13:00",
        )

    # ____________________________________________________________________
    # Test aggregate reanalysis values and date range with user-defined end date, at hourly time resolution
    analysis = MonteCarloAEP(
        reanalysis_aep_project,
        reanalysis_products=["merra2", "era5"],
        time_resolution="h",
        end_date_lt="2019-02-10 12:00",
    )
    df_rean = analysis._reanalysis_aggregate

    # check that date range is truncated correctly
    assert (df_rean.index[0], df_rean.index[-1]) == (
        pd.to_datetime("1999-01-15 12:00"),
        pd.to_datetime("2019-02-10 12:00"),
    )

    # Check wind speed values at start and end dates
    expected = {"merra2": [10.509840, 16.985526], "era5": [9.202639, 15.608469]}
    computed = {
        key: df_rean.loc[[df_rean.index[0], df_rean.index[-1]], key].to_numpy() for key in expected
    }

    for key in expected:
        npt.assert_array_almost_equal(computed[key], expected[key])


def test_monthly_lin(aep_project):
    reset_prng()
    # ____________________________________________________________________
    # Test linear regression model, at monthly time resolution
    analysis = MonteCarloAEP(
        aep_project,
        reanalysis_products=["merra2", "era5"],
        time_resolution="MS",
        reg_model="lin",
        reg_temperature=False,
        reg_wind_direction=False,
    )
    # Run Monte Carlo AEP analysis, confirm the results are consistent
    analysis.run(num_sim=10)
    sim_results = analysis.results

    # Make sure AEP results are consistent to six decimal places
    expected_results = [11.284629, 10.801102, 1.130812, 4.287147, 0.061666, 5.365357]

    calculated_results = [
        sim_results.aep_GWh.mean(),
        sim_results.aep_GWh.std() / sim_results.aep_GWh.mean() * 100,
        sim_results.avail_pct.mean() * 100,
        sim_results.avail_pct.std() / sim_results.avail_pct.mean() * 100,
        sim_results.curt_pct.mean() * 100,
        sim_results.curt_pct.std() / sim_results.curt_pct.mean() * 100,
    ]

    npt.assert_array_almost_equal(calculated_results, expected_results)


# Test inputs to the regression model, at daily time resolution
def test_daily_inputs(subtests, aep_project):
    reset_prng()
    # ____________________________________________________________________
    # Test inputs to the regression model, at monthly time resolution
    analysis = MonteCarloAEP(
        aep_project,
        reanalysis_products=["merra2", "era5"],
        time_resolution="D",
        reg_temperature=True,
        reg_wind_direction=True,
    )
    df = analysis.aggregate

    # Check the pre-processing functions
    with subtests.test("Revenue meter pre-processing"):
        # Energy Nan flags are all zero
        npt.assert_array_equal(df["energy_nan_perc"].values, np.repeat(0.0, df.shape[0]))

        # Check a few energy values
        expected_gwh = pd.Series([0.0848525, 0.0253657, 0.0668642])
        actual_gwh = df.loc[
            pd.to_datetime(["2014-01-02", "2014-10-12", "2015-12-28"]), "energy_gwh"
        ]
        npt.assert_array_almost_equal(actual_gwh, expected_gwh)

    with subtests.test("Availability and curtailment pre-processing"):
        # Availablity, curtailment nan fields both 0, NaN flag is all False
        npt.assert_array_equal(df["avail_nan_perc"].values, np.repeat(0.0, df.shape[0]))
        npt.assert_array_equal(df["curt_nan_perc"].values, np.repeat(0.0, df.shape[0]))
        npt.assert_array_equal(df["nan_flag"].values, np.repeat(False, df.shape[0]))  # noqa: FBT003

        # Check a few reported availabilty and curtailment values
        expected_avail_gwh = pd.Series([0.0000483644, 0.000000, 0.000000])
        expected_curt_gwh = pd.Series([0.000000, 0.00019581, 0.000000])
        expected_avail_pct = pd.Series([0.000569658, 0.000000, 0.000000])
        expected_curt_pct = pd.Series([0.000000, 0.00766034, 0.000000])

        date_ind = pd.to_datetime(["2014-01-02", "2014-10-12", "2015-12-28"])

        npt.assert_array_almost_equal(df.loc[date_ind, "availability_gwh"], expected_avail_gwh)
        npt.assert_array_almost_equal(df.loc[date_ind, "curtailment_gwh"], expected_curt_gwh)
        npt.assert_array_almost_equal(df.loc[date_ind, "availability_pct"], expected_avail_pct)
        npt.assert_array_almost_equal(df.loc[date_ind, "curtailment_pct"], expected_curt_pct)

    with subtests.test("Reanalysis pre-processing"):
        expected = {
            "merra2": np.array([11.02459231, 7.04306896, 8.41880152]),
            "era5": np.array([10.47942319, 7.71069617, 9.60864791]),
            "merra2_WMETR_HorWdDir": np.array([213.81683361, 129.08053181, 170.39815032]),
            "era5_WMETR_HorWdDir": np.array([212.23854097, 127.75317448, 170.33488958]),
            "merra2_WMETR_EnvTmp": np.array([279.67922333, 285.69317833, 278.15574917]),
            "era5_WMETR_EnvTmp": np.array([281.14880642, 285.81961816, 280.42017656]),
        }

        date_ind = pd.to_datetime(["2014-01-02", "2014-10-12", "2015-12-28"])
        computed = {key: df.loc[date_ind, key].to_numpy() for key in expected}

        for key in expected:
            with subtests.test(f"checking {key}"):
                npt.assert_array_almost_equal(computed[key], expected[key])


def test_daily_gam(aep_project):
    reset_prng()
    # ____________________________________________________________________
    # Test GAM regression model (can be used at daily time resolution only)
    analysis = MonteCarloAEP(
        aep_project,
        reanalysis_products=["merra2", "era5"],
        time_resolution="D",
        reg_model="gam",
        reg_temperature=True,
        reg_wind_direction=True,
    )
    # Run Monte Carlo AEP analysis, confirm the results are consistent
    analysis.run(num_sim=5)
    sim_results = analysis.results

    # Make sure AEP results are consistent to six decimal places
    expected_results = [12.627031, 7.9379, 1.298632, 6.403933, 0.040224, 12.434362]

    calculated_results = [
        sim_results.aep_GWh.mean(),
        sim_results.aep_GWh.std() / sim_results.aep_GWh.mean() * 100,
        sim_results.avail_pct.mean() * 100,
        sim_results.avail_pct.std() / sim_results.avail_pct.mean() * 100,
        sim_results.curt_pct.mean() * 100,
        sim_results.curt_pct.std() / sim_results.curt_pct.mean() * 100,
    ]

    npt.assert_array_almost_equal(calculated_results, expected_results)


@pytest.mark.xfail(reason="Fails intermittently depending, mostly on MacOS.")
def test_daily_gbm(aep_project):
    reset_prng()
    # ____________________________________________________________________
    # Test GBM regression model (can be used at daily time resolution only)
    analysis = MonteCarloAEP(
        aep_project,
        reanalysis_products=["era5"],
        time_resolution="D",
        reg_model="gbm",
        reg_temperature=True,
        reg_wind_direction=False,
    )
    # Run Monte Carlo AEP analysis, confirm the results are consistent
    analysis.run(num_sim=5)
    sim_results = analysis.results

    # Make sure AEP results are consistent to six decimal places
    expected_results = [12.955753, 14.918292, 1.310864, 5.250295, 0.049952, 9.226035]

    calculated_results = [
        sim_results.aep_GWh.mean(),
        sim_results.aep_GWh.std() / sim_results.aep_GWh.mean() * 100,
        sim_results.avail_pct.mean() * 100,
        sim_results.avail_pct.std() / sim_results.avail_pct.mean() * 100,
        sim_results.curt_pct.mean() * 100,
        sim_results.curt_pct.std() / sim_results.curt_pct.mean() * 100,
    ]

    npt.assert_array_almost_equal(calculated_results, expected_results)


def test_daily_etr(aep_project):
    reset_prng()
    # ____________________________________________________________________
    # Test ETR regression model (can be used at daily time resolution only)
    analysis = MonteCarloAEP(
        aep_project,
        reanalysis_products=["merra2"],
        time_resolution="D",
        reg_model="etr",
        reg_temperature=False,
        reg_wind_direction=False,
    )
    # Run Monte Carlo AEP analysis, confirm the results are consistent
    analysis.run(num_sim=5)
    sim_results = analysis.results

    # Make sure AEP results are consistent to six decimal places
    expected_results = [12.938535, 8.561778, 1.334704, 5.33619, 0.057874, 11.339982]

    calculated_results = [
        sim_results.aep_GWh.mean(),
        sim_results.aep_GWh.std() / sim_results.aep_GWh.mean() * 100,
        sim_results.avail_pct.mean() * 100,
        sim_results.avail_pct.std() / sim_results.avail_pct.mean() * 100,
        sim_results.curt_pct.mean() * 100,
        sim_results.curt_pct.std() / sim_results.curt_pct.mean() * 100,
    ]

    npt.assert_array_almost_equal(calculated_results, expected_results)


def test_daily_gam_outliers(aep_project):
    reset_prng()
    # ____________________________________________________________________
    # Test GAM regression model (can be used at daily time resolution only)
    analysis = MonteCarloAEP(
        aep_project,
        reanalysis_products=["merra2", "era5"],
        time_resolution="D",
        outlier_detection=True,
        reg_model="gam",
        reg_temperature=True,
        reg_wind_direction=True,
    )
    # Run Monte Carlo AEP analysis, confirm the results are consistent
    analysis.run(num_sim=5)
    sim_results = analysis.results

    # Make sure AEP results are consistent to six decimal places
    expected_results = [12.407638, 15.787861, 1.295542, 6.693327, 0.039864, 15.81297]
    calculated_results = [
        sim_results.aep_GWh.mean(),
        sim_results.aep_GWh.std() / sim_results.aep_GWh.mean() * 100,
        sim_results.avail_pct.mean() * 100,
        sim_results.avail_pct.std() / sim_results.avail_pct.mean() * 100,
        sim_results.curt_pct.mean() * 100,
        sim_results.curt_pct.std() / sim_results.curt_pct.mean() * 100,
    ]

    npt.assert_array_almost_equal(calculated_results, expected_results)
