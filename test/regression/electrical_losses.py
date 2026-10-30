import pytest
import numpy.testing as npt

from test.conftest import reset_prng
from openoa.analysis.electrical_losses import ElectricalLosses


@pytest.fixture
def el_project(project):
    return project("ElectricalLosses")


def test_electrical_losses_results(el_project):
    reset_prng()
    analysis = ElectricalLosses(el_project, UQ=False, uncertainty_correction_threshold=0.95)
    analysis.run()

    expected_losses = 0.02
    expected_losses_std = 0
    actual_compiled_data = analysis.electrical_losses.mean()
    actual_compiled_data_std = analysis.electrical_losses.std()

    npt.assert_array_almost_equal(actual_compiled_data, expected_losses, decimal=3)
    npt.assert_array_almost_equal(actual_compiled_data_std, expected_losses_std, decimal=3)


def test_electrical_losses_results_with_UQ(el_project):
    reset_prng()
    analysis_uq = ElectricalLosses(
        el_project, UQ=True, num_sim=3000, uncertainty_correction_threshold=(0.9, 0.995)
    )
    analysis_uq.run()

    expected_losses_uq = 0.02
    expected_losses_uq_std = 0.0069
    actual_compiled_data_uq = analysis_uq.electrical_losses.mean()
    actual_compiled_data_uq_std = analysis_uq.electrical_losses.std()

    npt.assert_array_almost_equal(actual_compiled_data_uq, expected_losses_uq, decimal=3)
    npt.assert_array_almost_equal(actual_compiled_data_uq_std, expected_losses_uq_std, decimal=3)
