import random
import unittest

import numpy as np
import pandas as pd
import pytest
from numpy import testing as nptest

from openoa.analysis import wake_losses

from test.conftest import project_ENGIE, example_data_path_str  # isort: skip


def reset_prng():
    np.random.seed(42)
    random.seed(42)


class TestWakeLosses(unittest.TestCase):
    def setUp(self):
        """
        Python Unittest setUp method.
        Load data from disk into PlantData objects and prepare the data for testing the WakeLosses method.
        """
        reset_prng()

        # Set up data to use for testing (ENGIE example plant)
        self.project = project_ENGIE.prepare(example_data_path_str, use_cleansed=False)
        self.project.analysis_type.append("WakeLosses-scada")
        self.project.validate()

        # Apply estimated northing calibration to SCADA wind directions
        self.project.scada["WMET_HorWdDir"] = (self.project.scada["WMET_HorWdDir"] + 15.85) % 360.0

    def test_wake_losses_without_UQ(self):
        reset_prng()
        # ____________________________________________________________________
        # Test POR and long-term corrected wake losses at plant and turbine level, without UQ.
        # Limit wind direction assets to three reliable turbines and limit date range to exclude
        # change in wind direction reference. Otherwise, use default parameters.
        self.analysis = wake_losses.WakeLosses(
            plant=self.project,
            wind_direction_asset_ids=["R80711", "R80721", "R80736"],
            end_date="2015-11-25 00:00",
            UQ=False,
        )

        # Run Wake Loss analysis, using default parameters. Aside from no_wakes_ws_thresh_LT_corr,
        # use default parameters. Confirm the results are consistent.
        self.analysis.run(
            no_wakes_ws_thresh_LT_corr=15.0,
            num_years_LT=20,
            freestream_sector_width=90.0,
            wind_bin_mad_thresh=7.0,
        )
        self.check_simulation_results_wake_losses_without_UQ()

    def test_wake_losses_with_UQ(self):
        reset_prng()
        # ____________________________________________________________________
        # Test POR and long-term corrected wake losses at plant and turbine level, with UQ.
        # Limit wind direction assets to three reliable turbines and limit date range to exclude
        # change in wind direction reference. Otherwise, use default parameters.
        self.analysis = wake_losses.WakeLosses(
            plant=self.project,
            wind_direction_asset_ids=["R80711", "R80721", "R80736"],
            end_date="2015-11-25 00:00",
            UQ=True,
        )

        # Run Wake Loss analysis with 50 Monte Carlo iterations.
        # Aside from no_wakes_ws_thresh_LT_corr and num_sim, use default parameters.
        # Confirm the results are consistent.
        self.analysis.run(
            num_sim=50, no_wakes_ws_thresh_LT_corr=15.0, reanalysis_products=["merra2", "era5"]
        )
        self.check_simulation_results_wake_losses_with_UQ()

    def test_wake_losses_with_UQ_new_parameters(self):
        reset_prng()
        # ____________________________________________________________________
        # Test POR and long-term corrected wake losses at plant and turbine level, with UQ.
        # Limit wind direction assets to three reliable turbines. Assign non-default start and
        # end dates and end date for reanalysis data for long-term correction.
        self.analysis = wake_losses.WakeLosses(
            plant=self.project,
            wind_direction_asset_ids=["R80711", "R80721", "R80736"],
            start_date="2014-03-01 00:00",
            end_date="2015-10-31 23:50",
            end_date_lt="2018-06-30 23:00",
            UQ=True,
        )

        # Run Wake Loss analysis with 50 Monte Carlo iterations.
        # Use non-default values for wind direction bin width for identifying freestream turbines,
        # freestream sector width, freestream power and wind speed averaging methods, and number of
        # years for long-term correction. Further, do not correct for derated turbines and do not
        # assume no wake losses above a certain wind speed for long-term correction.
        # Confirm the results are consistent.
        self.analysis.run(
            num_sim=50,
            wd_bin_width=10.0,
            freestream_sector_width=(60.0, 100.0),
            freestream_power_method="median",
            freestream_wind_speed_method="median",
            correct_for_derating=False,
            num_years_LT=(5, 15),
            assume_no_wakes_high_ws_LT_corr=False,
            reanalysis_products=["merra2", "era5"],
        )
        self.check_simulation_results_wake_losses_with_UQ_new_params()

    def test_wake_losses_with_heterogeneity_corrections(self):
        reset_prng()
        # ____________________________________________________________________
        # Test POR and long-term corrected wake losses at plant and turbine level, without UQ, with
        # corrections for freestream wind speed heterogeneity across the wind plant using the wind
        # speedup factor file from the examples folder. Limit wind direction assets to three
        # reliable turbines and limit date range to exclude change in wind direction reference.
        # Otherwise, use default parameters.
        self.analysis = wake_losses.WakeLosses(
            plant=self.project,
            wind_direction_asset_ids=["R80711", "R80721", "R80736"],
            end_date="2015-11-25 00:00",
            UQ=False,
            correct_for_ws_heterogeneity=True,
            ws_speedup_factor_map="examples/example_la_haute_borne_ws_speedup_factors.csv",
        )

        # Run Wake Loss analysis, using default parameters. Aside from no_wakes_ws_thresh_LT_corr,
        # use default parameters. Confirm the results are consistent.
        self.analysis.run(
            no_wakes_ws_thresh_LT_corr=15.0,
            num_years_LT=20,
            freestream_sector_width=90.0,
            wind_bin_mad_thresh=7.0,
        )
        self.check_simulation_results_wake_losses_with_heterogeneity_corrections()

    def check_simulation_results_wake_losses_without_UQ(self):
        # Make sure wake loss results are consistent to six decimal places
        # Confirm plant-level and turbine-level wake losses for POR and long-term corrected
        # wake loss estimates.
        expected_results_por = [0.340045, -11.727658, 10.898059, 4.065239, -1.910556]
        expected_results_lt = [0.373332, -9.713340, 10.282598, 2.933038, -2.034775]

        calculated_results_por = [100 * self.analysis.wake_losses_por]
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por))

        nptest.assert_array_almost_equal(expected_results_por, calculated_results_por)

        calculated_results_lt = [100 * self.analysis.wake_losses_lt]
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt))

        nptest.assert_array_almost_equal(expected_results_lt, calculated_results_lt)

    def check_simulation_results_wake_losses_with_UQ(self):
        # Make sure wake loss results are consistent to six decimal places
        # Confirm plant-level and turbine-level means and std. devs. from Monte Carlo simulation results
        # for POR and long-term corrected wake loss estimates.
        expected_results_por = [
            0.466709,
            1.519220,
            -11.560934,
            11.021836,
            4.167384,
            -1.795656,
            1.698255,
            1.364234,
            1.483180,
            1.545941,
        ]
        expected_results_lt = [
            0.644775,
            1.372648,
            -9.436220,
            10.614411,
            3.111282,
            -1.732393,
            1.546301,
            1.323577,
            1.365420,
            1.426536,
        ]

        calculated_results_por = [
            100 * self.analysis.wake_losses_por_mean,
            100 * self.analysis.wake_losses_por_std,
        ]
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por_mean))
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por_std))

        nptest.assert_array_almost_equal(expected_results_por, calculated_results_por)

        calculated_results_lt = [
            100 * self.analysis.wake_losses_lt_mean,
            100 * self.analysis.wake_losses_lt_std,
        ]
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt_mean))
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt_std))

        nptest.assert_array_almost_equal(expected_results_lt, calculated_results_lt)

    def check_simulation_results_wake_losses_with_UQ_new_params(self):
        # Make sure wake loss results are consistent to six decimal places
        # Confirm plant-level and turbine-level means and std. devs. from Monte Carlo simulation results
        # for POR and long-term corrected wake loss estimates.
        expected_results_por = [
            0.917651,
            2.541353,
            -10.941171,
            11.134159,
            5.245831,
            -1.768214,
            2.867614,
            2.271275,
            2.404548,
            2.631516,
        ]
        expected_results_lt = [
            1.140835,
            2.426398,
            -8.811414,
            10.995446,
            3.487754,
            -1.108443,
            2.525045,
            2.318111,
            2.507327,
            2.43125,
        ]

        calculated_results_por = [
            100 * self.analysis.wake_losses_por_mean,
            100 * self.analysis.wake_losses_por_std,
        ]
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por_mean))
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por_std))

        nptest.assert_array_almost_equal(expected_results_por, calculated_results_por)

        calculated_results_lt = [
            100 * self.analysis.wake_losses_lt_mean,
            100 * self.analysis.wake_losses_lt_std,
        ]
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt_mean))
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt_std))

        nptest.assert_array_almost_equal(expected_results_lt, calculated_results_lt)

    def check_simulation_results_wake_losses_with_heterogeneity_corrections(self):
        # Make sure wake loss results are consistent to six decimal places
        # Confirm plant-level and turbine-level wake losses for POR and long-term corrected
        # wake loss estimates.
        expected_results_por = [1.670518, -0.077131, 2.223721, 0.278168, 4.298961]
        expected_results_lt = [1.610428, 0.482218, 2.671905, -0.158225, 3.490358]

        calculated_results_por = [100 * self.analysis.wake_losses_por]
        calculated_results_por += list(100 * np.array(self.analysis.turbine_wake_losses_por))

        nptest.assert_array_almost_equal(expected_results_por, calculated_results_por)

        calculated_results_lt = [100 * self.analysis.wake_losses_lt]
        calculated_results_lt += list(100 * np.array(self.analysis.turbine_wake_losses_lt))

        nptest.assert_array_almost_equal(expected_results_lt, calculated_results_lt)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
