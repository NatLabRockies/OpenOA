import pytest
import numpy.testing as npt

from openoa.analysis import TurbineLongTermGrossEnergy


from test.conftest import reset_prng  # isort: skip


@pytest.fixture(scope="module")
def tie_project(project):
    project = project("TurbineLongTermGrossEnergy")
    return project


def test_long_term_gross_energy_no_UQ(tie_project):
    reset_prng()
    analysis = TurbineLongTermGrossEnergy(
        tie_project,
        UQ=False,
        max_power_filter=0.85,
        wind_bin_threshold=1.0,
        correction_threshold=0.9,
    )
    analysis.run(reanalysis_products=["era5", "merra2"])

    res = analysis.plant_gross.mean()
    npt.assert_almost_equal(res / 1e6, 12.91634141, decimal=4)


def test_longterm_gross_energy_with_UQ(tie_project):
    reset_prng()
    analysis_uq = TurbineLongTermGrossEnergy(tie_project, UQ=True, num_sim=10)
    analysis_uq.run(reanalysis_products=["era5", "merra2"])

    reset_prng()

    # TODO: Determine why there is such instability in the results, or speed up code to run
    # more quickly and get more stability through more simulations. Alternatively, figure why
    # the numbers changed in the first place

    # Test UQ case, mean value
    res_uq = analysis_uq.plant_gross.mean()
    npt.assert_almost_equal(res_uq / 1e6, 13.6134409)

    # Test UQ case, stdev
    res_std_uq = analysis_uq.plant_gross.std()
    npt.assert_almost_equal(res_std_uq / 1e6, 0.28508504)
