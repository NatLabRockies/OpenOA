import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

from openoa.utils import filters


def test_range_flag_series():
    x = pd.Series(np.array([-1, 0, 1]), name="data")
    y = pd.Series([True, False, True], name="data")
    y_test = filters.range_flag(x, -0.5, 0.5)
    assert isinstance(y_test, pd.Series)
    assert y.equals(y_test)


def test_range_flag_dataframe():
    x = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["a", "b"])
    y_test = filters.range_flag(x, [2, 1], [8, 7])
    y = pd.DataFrame(
        [[True, False], [False, False], [False, False], [False, False], [False, True]],
        columns=["a", "b"],
    )
    assert isinstance(y_test, pd.DataFrame)
    assert y.equals(y_test)


def test_range_flag_dataframe_filtered():
    x = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["a", "b"])
    y_test = filters.range_flag(x, [2], [6], ["a"])
    y = pd.DataFrame([[True], [False], [False], [False], [True]], columns=["a"])
    assert isinstance(y_test, pd.DataFrame)
    assert y.equals(y_test)


def test_range_flag_errors():
    x = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["a", "b"])
    with pytest.raises(ValueError):
        filters.range_flag(x, [2, 1], [6, 5], ["a"])

    with pytest.raises(ValueError):
        filters.range_flag(x, [2], [6, 5], ["a"])

    with pytest.raises(ValueError):
        filters.range_flag(x, [2], [6], ["a", "b"])

    with pytest.raises(ValueError):
        filters.range_flag(x, 2, [6], ["a", "b"])

    with pytest.raises(ValueError):
        filters.range_flag(x, [2], 6, ["a", "b"])


def test_unresponsive_flag():
    x = pd.Series(np.array([-1, -1, -1, 2, 2, 2, 3, 4, 5, 1, 1, 1, 1, 3, 3]), name="data")
    y = pd.Series(
        [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
        ],
        name="data",
    )
    y_test = filters.unresponsive_flag(x, threshold=3)
    assert y.equals(y_test)


def test_unresonsive_flag_errors():
    x = pd.Series(np.array([-1, -1, -1, 2, 2, 2, 3, 4, 5, 1, 1, 1, 1, 3, 3]), name="data")
    with pytest.raises(TypeError):
        filters.unresponsive_flag(x, threshold=3.5)


def test_unresponsive_flag_df():
    x = pd.DataFrame(
        [
            [-1, -1],
            [-1, -2],
            [-1, -3],
            [2, 2],
            [2, 2],
            [2, 2],
            [3, 2],
            [4, 3],
            [5, 4],
            [1, 6],
            [1, 8],
            [1, 1],
            [1, 1],
            [3, 1],
            [3, 1],
        ],
        columns=["a", "b"],
    )
    y = pd.DataFrame(
        [
            [True, False],
            [True, False],
            [True, False],
            [True, True],
            [True, True],
            [True, True],
            [False, True],
            [False, False],
            [False, False],
            [True, False],
            [True, False],
            [True, True],
            [True, True],
            [True, True],
            [True, True],
        ],
        columns=["a", "b"],
    )
    y_test = filters.unresponsive_flag(x, threshold=2)
    assert y.equals(y_test)


def test_window_range_flag():
    x = pd.Series(np.array([-1, -1, -1, 1, 1, 1, -1]), name="data")
    window = pd.Series(np.array([1, 2, 3, 4, 5, 6, 7]), name="window")
    y = pd.Series([False, False, True, False, False, False, True], name="data")
    y_test = filters.window_range_flag(window, 3, 8, x, -0.5, 1.5)
    assert y.equals(y_test)


def test_std_range_flag():
    x = pd.Series(np.array([-1, -1, -1, 1, -1, -1, -1]), name="data")
    y_test = filters.std_range_flag(x, 2)
    y = pd.Series([False, False, False, True, False, False, False])
    npt.assert_array_equal(y_test, y)


def test_std_range_flag_df():
    x = pd.DataFrame(
        [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, 1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        columns=["a", "b", "c"],
    )
    y_test = filters.std_range_flag(x, 2, col=["b", "c"])
    y = pd.DataFrame(
        [
            [False, False],
            [False, False],
            [True, False],
            [False, False],
            [False, False],
            [False, False],
            [False, False],
        ],
        columns=["b", "c"],
    )
    assert y.equals(y_test)


def test_std_range_flag_errors():
    x = pd.DataFrame(
        [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, 1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        columns=["a", "b", "c"],
    )
    with pytest.raises(ValueError):
        filters.std_range_flag(x, [2], col=["b", "c"])


# TODO: Test more code paths in bin_filter
def test_bin_filter():
    x_val = pd.Series(np.array([-1, -1, -1, -1, -1, 10, -1]))
    x_bin = pd.Series(np.array([1, 1.5, 2, 2.5, 3, 3.5, 4]))
    flag = filters.bin_filter(x_bin, x_val, 3)
    expected = pd.Series([False, False, False, False, False, True, False])
    npt.assert_array_equal(flag, expected)


@pytest.mark.flaky(reruns=2, delay=5)
def test_cluster_mahalanobis_2d():
    col1 = pd.Series(np.array([1.0, 1.01, 1.001, 2.0, 2.01, 2.001, 2.0001]))
    col2 = pd.Series(np.array([3.0, 3.02, 3.001, 4.0, 4.01, 4.001, 5.0001]))
    flag = filters.cluster_mahalanobis_2d(col1, col2, 2, 1.5)
    expected = pd.Series(np.array([False, False, False, False, False, False, True]))
    npt.assert_array_equal(flag, expected)
