import pytest
import numpy.testing as npt

from test.conftest import reset_prng


@pytest.fixture
def gap_analysis_project(project):
    return project(None)


def test_gap_analysis(gap_analysis_project):
    reset_prng()
    # Set up operational results data
    oa_data = {
        "aep": 448.0,
        "availability_losses": 0.0493,
        "electrical_losses": 0.012,
        "turbine_ideal_energy": 477.8,
    }
    # AEP (GWh/yr), availability loss (fraction), electrical loss (fraction), turbine ideal energy (GWh/yr)

    eya_data = {
        "aep": 467.0,
        "gross_energy": 597.14,
        "availability_losses": 0.062,
        "electrical_losses": 0.024,
        "turbine_losses": 0.037,
        "blade_degradation_losses": 0.011,
        "wake_losses": 0.087,
    }

    # AEP (GWh/yr), gross energy (GWh/yr), availability loss (fraction), electrical loss (fraction),
    # turbine performance loss (fraction), blade degradation loss (fraction), wake loss (fraction)

    analysis = gap_analysis_project.EYAGapAnalysis(eya_estimates=eya_data, oa_results=oa_data)
    analysis.run()

    expected_compiled_data = [467.0, -41.441648, 6.59437, 6.230899, 9.61638]
    actual_compiled_data = analysis.compiled_data
    npt.assert_array_almost_equal(actual_compiled_data, expected_compiled_data, decimal=3)
