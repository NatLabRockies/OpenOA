import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from test.conftest import reset_prng
from openoa.utils.machine_learning_setup import MachineLearningSetup


@pytest.fixture
def create_X_y():
    reset_prng()

    # Create 2 random features, 1st to mimic wind speed, the other air density
    x1 = pd.Series(np.random.random(1000) * 30)  # Wind speed
    x2 = pd.Series(np.random.random(1000) * 360)  # Wind direction
    x3 = pd.Series(1 + np.random.random(1000) * 0.2)  # Air density

    # Group features together
    X = pd.DataFrame(data={"ws": x1, "wd": x2, "dens": x3})

    # Create simple power relationship with feature variables
    y = x3 * np.power(x1, 3) * np.log(x2) / 6  # Units of kW
    return X, y


@pytest.mark.parametrize(
    "algorithm,r2,rmse",
    [("etr", 0.999852, 130.0), ("gbm", 0.999999, 60.0), ("gam", 0.98397, 1400.0)],
)
def test_algorithm_defaults(subtests, create_X_y, algorithm, r2, rmse):
    reset_prng()

    X, y = create_X_y

    ml = MachineLearningSetup(algorithm)

    # Perform a limited randomized grid search for efficiency
    ml.hyper_optimize(X, y, n_iter_search=7, report=False, cv=KFold(n_splits=2))

    # Predict power based on model results
    y_pred = ml.random_search.predict(X)

    # Compute performance metrics which we'll test
    r2_actual = np.corrcoef(y, y_pred)[0, 1]  # Correlation between predicted and actual power
    rmse_actual = np.sqrt(mean_squared_error(y, y_pred))  # RMSE between predicted and actual power

    # Mean power in GW is within 3 decimal places
    with subtests.test(f"Sum of predicted and actual power for {algorithm} not close enough"):
        npt.assert_approx_equal(y.sum() / 1e6, y_pred.sum() / 1e6, significant=3)

    # Test correlation of model fit
    with subtests.test(f"Correlation between {algorithm} features and response is wrong"):
        npt.assert_approx_equal(r2_actual, r2, significant=4)

    # Test RMSE of model fit
    with subtests.test(f"RMSE of {algorithm} fit is too high"):
        assert rmse_actual < rmse
