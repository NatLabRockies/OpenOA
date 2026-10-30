import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

from openoa.utils import imputing


@pytest.fixture
def test_df_2_assets_1_NaN():
    """Test dataframe with two assets, one NaN entry in first asset that requires imputation."""
    test_df = pd.DataFrame(
        data={
            "time": ["01", "02", "03", "04", "05", "01", "02", "03", "04", "05"],
            "data": [0, np.nan, 4, 5, 8, 13, 18, 20, 20, 30],
            "asset_id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        },
        index=np.arange(10),
    ).set_index(["time", "asset_id"])
    return test_df


def test_asset_correlation_matrix(subtests, test_df_2_assets_1_NaN):
    # Test 1, make sure a simple correlation of two assets works
    y = imputing.asset_correlation_matrix(test_df_2_assets_1_NaN, "data")
    with subtests.test("Simple correlation of 2 assets"):
        npt.assert_array_almost_equal(
            np.array([[np.nan, 0.970166], [0.970166, np.nan]]), y, decimal=4
        )

    # Test 2, if no overlapping data are present, make sure correlation matrix is all NaN
    # Data frame of two assets with no overlapping occurrences of valid data
    test_df = pd.DataFrame(
        data={
            "time": ["01", "02", "03", "04", "05", "01", "02", "03", "04", "05"],
            "data": [0, np.nan, np.nan, 5, 8, np.nan, 20, 20, np.nan, np.nan],
            "asset_id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        },
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    ).set_index(["time", "asset_id"])
    y2 = imputing.asset_correlation_matrix(test_df, "data")
    with subtests.test("No overlapping data, corr mat is NaNs"):
        npt.assert_array_equal(y2, np.array([[np.nan, np.nan], [np.nan, np.nan]]))


def test_impute_data(subtests, test_df_2_assets_1_NaN):
    # Data shared between at least three subtests
    test_df_single_asset_3_nan = pd.DataFrame(
        index=["a", "b", "c", "d", "e"],
        data={"data1": [1, np.nan, np.nan, np.nan, 5], "align": ["a", "b", "c", "d", "e"]},
    )
    test_df_single_asset_all_nan = pd.DataFrame(
        index=["a", "b", "c", "d", "e"],
        data={
            "data1": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "align": ["a", "b", "c", "d", "e"],
        },
    )

    # Test 1a, make sure single NaN is imputed using old style of inputs
    y = np.float64(2.989779)
    y_test = imputing.impute_data(
        target_data=test_df_2_assets_1_NaN.xs("a", level=1),
        reference_data=test_df_2_assets_1_NaN.xs("b", level=1),
        target_col="data",
        reference_col="data",
        align_col="time",
        method="linear",
    )
    npt.assert_almost_equal(y_test.values[1], y, decimal=4)

    # Test 1b, make sure single NaN is imputed properly and that the index is correct
    data = test_df_2_assets_1_NaN.unstack().droplevel(0, axis=1)
    y_test = imputing.impute_data(
        data=data, target_col="a", reference_col="b", method="polynomial", degree=1
    )
    npt.assert_almost_equal(y_test.values[1], y, decimal=4)

    # Test 2, make sure only the first NaN entry is imputed
    # 2 assets, first asset as 2 NaN entries, 2nd asset has one NaN entry overlapping with the first
    test_df = pd.DataFrame(
        data={
            "time": ["01", "02", "03", "04", "05", "01", "02", "03", "04", "05"],
            "data": [0, np.nan, np.nan, 5, 8, 13, np.nan, 20, 20, 30],
            "asset_id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        },
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )
    y_test = imputing.impute_data(
        "data",
        "data",
        test_df.loc[test_df.asset_id == "a"],
        test_df.loc[test_df.asset_id == "b"],
        "time",
    )
    with subtests.test("2 assets w/NaNs, only the first NaN is imputed"):
        npt.assert_almost_equal(np.float64(y_test.loc["c"]), np.float64(3.874429), decimal=4)
        npt.assert_equal(y_test.loc["b"], np.nan)

    # Test 3, make sure no data is imputed when no NaN are present
    # NOTE: This test shouldn't have ever worked, but for some reason does, but open to
    # other amendments that don't just get rid of a test case

    test_df_single_asset_no_nan1 = pd.DataFrame(
        index=["a", "b", "c", "d", "e"],
        data={"data1": [1, 2, 3, 4, 5], "align": ["a", "b", "c", "d", "e"]},
    )

    # Single asset, one less entry than prior df
    test_df_single_asset_no_nan2 = pd.DataFrame(
        index=np.arange(4), data={"data2": [1, 2, 4, 5], "align": ["a", "c", "d", "e"]}
    )
    y_test = imputing.impute_data(
        target_col="data1",
        reference_col="data2",
        target_data=test_df_single_asset_no_nan1,
        reference_data=test_df_single_asset_no_nan2,
        align_col="align",
    )
    with subtests.test("No imputation on NaN-less data"):
        npt.assert_array_almost_equal(
            test_df_single_asset_no_nan1["data1"].to_numpy(), y_test.to_numpy()
        )

    # Test 4, make sure two NaNs are imputed when there 3 total but reference data only matches 2

    y_test = imputing.impute_data(
        reference_col="data2",
        target_col="data1",
        reference_data=test_df_single_asset_no_nan2,
        target_data=test_df_single_asset_3_nan,
        align_col="align",
    )

    with subtests.test("NaN imputation only when matching reference data"):
        npt.assert_array_almost_equal(np.array([2.0, 4.0]), y_test.loc[["c", "d"]], decimal=4)

    # Test 5a: Invalid reference data column
    with pytest.raises(ValueError):
        imputing.impute_data(
            target_data=test_df_single_asset_3_nan,
            target_col="data1",
            reference_data=test_df_single_asset_all_nan,
            reference_col="data2",
            align_col="align",
        )

    # Test 5b: Invalid target data column
    with pytest.raises(ValueError):
        imputing.impute_data(
            target_data=test_df_single_asset_3_nan,
            target_col="data2",
            reference_data=test_df_single_asset_all_nan,
            reference_col="data1",
            align_col="align",
        )

    # Test 5c: Bad data inputs
    with pytest.raises(TypeError):
        imputing.impute_data(
            target_data=test_df_single_asset_3_nan.to_numpy(),
            target_col="data1",
            reference_data=test_df_single_asset_all_nan.to_numpy(),
            reference_col="data2",
            align_col="align",
        )

    # Test 5c: Invalid alignment column
    with pytest.raises(ValueError):
        imputing.impute_data(
            target_data=test_df_single_asset_3_nan,
            target_col="data1",
            reference_data=test_df_single_asset_all_nan,
            reference_col="data2",
            align_col="index",
        )


def test_impute_all_assets_by_correlation(subtests):
    # Test 1: pass data frame with three highly correlated assets, ensure all NaN data are imputed in
    # final output
    # Data frame of 3 assets with overlapping NaN occurrences and highly correlated data
    test_df = pd.DataFrame(
        data={
            "time": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
            ],
            "data": [
                0,
                np.nan,
                np.nan,
                5,
                8,
                11,
                14,
                np.nan,
                23,
                33,
                np.nan,
                48,
                60,
                68,
                20.5,
                41,
                np.nan,
                85,
                np.nan,
                120,
                145,
            ],
            "asset_id": [
                "a",
                "a",
                "a",
                "a",
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "b",
                "b",
                "b",
                "b",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
            ],
        },
    ).set_index(["time", "asset_id"])
    with subtests.test("3 highly correlated assets w/NaNs"):
        y_test = imputing.impute_all_assets_by_correlation(
            test_df, "data", "data", r2_threshold=0.7
        ).to_frame()
        y = pd.Series([0.440789, 3.401316, 14.3677, 42.8312, 62.887218, 96.734818])
        npt.assert_array_almost_equal(
            y.values,
            (y_test.loc[test_df["data"].isnull(), "imputed_data"]).values,
            decimal=4,
        )

    # Test 2: 3 highly correlated assets with less data, such that asset 'b' has no data imputed
    # Data frame of 3 assets with overlapping NaN occurrences and highly correlated data
    # Asset 'b' in particular shouldn't be able to get imputed
    test_df = pd.DataFrame(
        data={
            "time": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "01",
                "02",
                "03",
                "04",
                "05",
                "01",
                "02",
                "03",
                "04",
                "05",
            ],
            "data": [
                0,
                np.nan,
                np.nan,
                5,
                8,
                np.nan,
                23,
                33,
                np.nan,
                np.nan,
                20.5,
                41,
                np.nan,
                85,
                np.nan,
            ],
            "asset_id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "c", "c", "c", "c", "c"],
        },
        index=np.arange(15),
    ).set_index(["time", "asset_id"])
    y = pd.Series([1.589147, np.nan, np.nan, np.nan, np.nan, np.nan, 123.7000])
    y_test = imputing.impute_all_assets_by_correlation(
        test_df, "data", "data", r2_threshold=0.7
    ).to_frame()
    with subtests.test("3 highly correlated assets w/less data"):
        nan_ind = test_df.loc[test_df["data"].isnull()].index
        npt.assert_array_almost_equal(y, y_test.loc[nan_ind, "imputed_data"], decimal=4)

    # Test 3: 2 poorly correlated data sets, no data should be imputed
    # Data frame of two assets that are poorly correlated
    test_df = pd.DataFrame(
        data={
            "time": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
            ],
            "data": [0, np.nan, np.nan, 5, 8, 11, 14, 40, 40, np.nan, 20, np.nan, 80, 10],
            "asset_id": ["a", "a", "a", "a", "a", "a", "a", "c", "c", "c", "c", "c", "c", "c"],
        },
        index=np.arange(14),
    ).set_index(["time", "asset_id"])
    with subtests.test("2 poorly correlated datasets w/o imputation"):
        y_test = imputing.impute_all_assets_by_correlation(
            test_df, "data", "data", r2_threshold=0.7
        ).to_frame()
        npt.assert_array_almost_equal(test_df["data"], y_test["imputed_data"], decimal=4)
