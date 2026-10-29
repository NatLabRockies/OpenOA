import copy
import tempfile
from pathlib import Path

import yaml
import pytest
from pandas.testing import assert_frame_equal

from openoa import PlantData
from openoa.schema import ANALYSIS_REQUIREMENTS, ReanalysisMetaData
from openoa.schema.schema import create_schema, create_analysis_schema


from test.conftest import project_ENGIE, example_data_path_str  # isort: skip


@pytest.fixture
def project_df():
    return project_ENGIE.prepare(
        path=example_data_path_str, return_value="dataframes", use_cleansed=False
    )


@pytest.fixture
def plant(project_df):
    scada_df, meter_df, curtail_df, asset_df, reanalysis_dict = project_df
    plant = PlantData(
        analysis_type=None,
        metadata=example_data_path_str + "/../plant_meta.yml",
        scada=scada_df,
        meter=meter_df,
        curtail=curtail_df,
        asset=asset_df,
        reanalysis=reanalysis_dict,
    )
    return plant


@pytest.fixture
def partial_plant(project_df):
    scada_df, meter_df, curtail_df, asset_df, _ = project_df
    with Path(example_data_path_str + "/../plant_meta.yml").open() as f:
        meta_partial = yaml.safe_load(f)
    meta_partial.pop("reanalysis")
    plant = PlantData(
        analysis_type=None,
        metadata=meta_partial,
        scada=scada_df,
        meter=meter_df,
        curtail=curtail_df,
        asset=asset_df,
    )
    return plant


@pytest.fixture
def schemas():
    schema_path = Path(__file__).resolve().parents[2] / "openoa/schema"

    with (schema_path / "full_schema.yml").open() as f:
        full_schema = yaml.safe_load(f)

    with (schema_path / "base_electrical_losses_schema.yml").open() as f:
        el_schema = yaml.safe_load(f)

    with (schema_path / "base_monte_carlo_aep_schema.yml").open() as f:
        mc_aep_schema = yaml.safe_load(f)

    with (schema_path / "base_tie_schema.yml").open() as f:
        tie_schema = yaml.safe_load(f)

    with (schema_path / "scada_wake_losses_schema.yml").open() as f:
        wake_schema = yaml.safe_load(f)

    return full_schema, el_schema, mc_aep_schema, tie_schema, wake_schema


def test_analysis_type_values(plant):
    """Test the acceptance of the valid inputs, except None."""
    valid = [*ANALYSIS_REQUIREMENTS, "all", None]

    plant.analysis_type = valid
    assert plant.analysis_type == valid

    # Test a couple of edge cases to show that only valid inputs are allowed
    # Add a misspelling
    with pytest.raises(ValueError):
        plant.analysis_type = "Montecarloaep"

    # Add a completely wrong value
    with pytest.raises(ValueError):
        plant.analysis_type = "this is wrong"


def test_validate_AEP_plant(plant):
    """The example plant should validate for MonteCarloAEP analysis type."""
    plant.analysis_type = "MonteCarloAEP"
    plant.validate()

    # Ensure that after validating a non-None analysis type, that the None is removed
    assert None not in plant.analysis_type


def test_does_not_validate_all(plant):
    """The example plant should not validate for MonteCarloAEP analysis type."""
    with pytest.raises(ValueError):
        plant.analysis_type = "all"
        plant.validate()


def test_update_columns(plant):
    """Tests that the column names are successfully mapped to the standardized names."""
    # Put the plant analysis type back in working order
    plant.analysis_type = "MonteCarloAEP"
    plant.validate()

    # Get the OpenOA standardized column names where the default isn't used
    scada_original = {v for k, v in plant.metadata.scada.col_map.items() if k != v}
    assert len(scada_original.intersection(plant.scada.columns)) == 0

    meter_original = {v for k, v in plant.metadata.meter.col_map.items() if k != v}
    assert len(meter_original.intersection(plant.meter.columns)) == 0

    asset_original = {v for k, v in plant.metadata.asset.col_map.items() if k != v}
    assert len(asset_original.intersection(plant.asset.columns)) == 0

    curtail_original = {v for k, v in plant.metadata.curtail.col_map.items() if k != v}
    assert len(curtail_original.intersection(plant.curtail.columns)) == 0

    for name in plant.reanalysis:
        re_original = {v for k, v in plant.metadata.reanalysis[name].col_map.items() if k != v}
        assert len(re_original.intersection(plant.reanalysis[name].columns)) == 0


def test_to_CSV(plant):
    """Save this plant to a temporary directory, load it in, and make sure the data matches."""
    data_path = tempfile.mkdtemp()
    plant.to_csv(save_path=data_path, with_openoa_col_names=True)
    plant_loaded = PlantData(
        metadata=f"{data_path}/metadata.yml",
        scada=f"{data_path}/scada.csv",
        meter=f"{data_path}/meter.csv",
        curtail=f"{data_path}/curtail.csv",
        asset=f"{data_path}/asset.csv",
        reanalysis={
            "era5": f"{data_path}/reanalysis_era5.csv",
            "merra2": f"{data_path}/reanalysis_merra2.csv",
        },
    )

    assert_frame_equal(
        plant.scada,
        plant_loaded.scada,
        "SCADA dataframe did not survive CSV save/loading process",
    )
    assert_frame_equal(
        plant.meter,
        plant_loaded.meter,
        "Meter dataframe did not survive CSV save/loading process",
    )
    assert_frame_equal(
        plant.curtail,
        plant_loaded.curtail,
        "Curtail dataframe did not survive CSV save/loading process",
    )
    assert_frame_equal(
        plant.asset,
        plant_loaded.asset,
        "Asset dataframe did not survive CSV save/loading process",
    )
    assert_frame_equal(
        plant.reanalysis["era5"],
        plant_loaded.reanalysis["era5"],
        "ERA5 dataframe did not survive CSV save/loading process",
    )
    assert_frame_equal(
        plant.reanalysis["merra2"],
        plant_loaded.reanalysis["merra2"],
        "MERRA2 dataframe did not survive CSV save/loading process",
    )


def test_reanalysis_defaults(partial_plant):
    assert partial_plant.metadata.reanalysis == {"product": ReanalysisMetaData()}


def test_reanalysis_missing_metadata(project_df):
    """Tests that when there are missing products in the reanalysis metadata, that
    a KeyError is raised early.
    """
    scada_df, meter_df, *_, reanalysis_dict = project_df
    with Path(example_data_path_str + "/../plant_meta.yml").open() as f:
        metadata = yaml.safe_load(f)

    # Raised when all missing
    meta_partial = copy.deepcopy(metadata)
    meta_partial.pop("reanalysis")
    with pytest.raises(KeyError):
        PlantData(scada=scada_df, meter=meter_df, reanalysis=reanalysis_dict)

    # Raised when there is only some missing
    meta_partial = copy.deepcopy(metadata)
    meta_partial["reanalysis"].pop("merra2")
    with pytest.raises(KeyError):
        PlantData(scada=scada_df, meter=meter_df, reanalysis=reanalysis_dict)


def test_full_schema(schemas):
    expected_full_schema, *_ = schemas
    full_schema = create_schema()
    assert expected_full_schema == full_schema


def test_analysis_schemas(subtests, schemas):
    """A direct comparison is not possible because the frequency ordering is different
    between the two dictionaries, so only check for matching required data types.
    """
    _, expected_el_schema, expected_mc_aep_schema, expected_tie_schema, expected_wake_schema = (
        schemas
    )

    with subtests.test("Validate electrical loss schema"):
        el_schema = create_analysis_schema("ElectricalLosses")
        assert expected_el_schema.keys() == el_schema.keys()
        for key, value in el_schema.items():
            # Check that the correct required columns are pulled
            assert expected_el_schema[key].keys() == value.keys()
            # Check for matching frequencies
            assert set(expected_el_schema[key]["frequency"]) == set(value["frequency"])

    with subtests.test("Validate AEP schema"):
        mc_aep_schema = create_analysis_schema("MonteCarloAEP")
        assert expected_mc_aep_schema.keys() == mc_aep_schema.keys()
        for key, value in mc_aep_schema.items():
            # Check that the correct required columns are pulled
            assert mc_aep_schema[key].keys() == value.keys()
            # Check for matching frequencies
            assert set(value["frequency"]) == set(mc_aep_schema[key]["frequency"])

    with subtests.test("Validate TIE schema"):
        tie_schema = create_analysis_schema("TurbineLongTermGrossEnergy")
        assert tie_schema.keys() == tie_schema.keys()
        for key, value in tie_schema.items():
            # Check that the correct required columns are pulled
            assert expected_tie_schema[key].keys() == value.keys()
            # Check for matching frequencies
            if key == "asset":
                assert "frequency" not in expected_tie_schema[key]
                continue
            assert set(expected_tie_schema[key]["frequency"]) == set(value["frequency"])

    with subtests.test("Validate wake loss schema"):
        wake_schema = create_analysis_schema("WakeLosses-scada")
        assert expected_wake_schema.keys() == wake_schema.keys()
        for key, value in wake_schema.items():
            # Check that the correct required columns are pulled
            assert expected_wake_schema[key].keys() == value.keys()
            # Check for matching frequencies
            if key == "asset":
                assert "frequency" not in expected_wake_schema[key]
                continue
            assert set(expected_wake_schema[key]["frequency"]) == set(value["frequency"])


def test_combined_schema():
    analysis_types = [
        "ElectricalLosses",
        "MonteCarloAEP",
        "TurbineLongTermGrossEnergy",
        "WakeLosses-scada",
        "StaticYawMisalignment",
    ]
    combined_schema = create_analysis_schema(analysis_types=analysis_types)

    correct_schema = {
        "scada": {
            "asset_id": {"name": "asset_id", "dtype": "str", "units": None},
            "WTUR_W": {"name": "WTUR_W", "dtype": "float", "units": "kW"},
            "WMET_HorWdSpd": {"name": "WMET_HorWdSpd", "dtype": "float", "units": "m/s"},
            "WMET_HorWdDir": {"name": "WMET_HorWdDir", "dtype": "float", "units": "m/s"},
            "WMET_HorWdDirRel": {"name": "WMET_HorWdDirRel", "dtype": "float", "units": "deg"},
            "WROT_BlPthAngVal": {"name": "WROT_BlPthAngVal", "dtype": "float", "units": "deg"},
            "frequency": [
                "h",
                "min",
                "s",
                "ms",
                "us",
                "ns",
            ],
        },
        "reanalysis": {
            "WMETR_HorWdSpd": {"name": "WMETR_HorWdSpd", "dtype": "float", "units": "m/s"},
            "WMETR_HorWdDir": {"name": "WMETR_HorWdDir", "dtype": "float", "units": "deg"},
            "WMETR_AirDen": {"name": "WMETR_AirDen", "dtype": "float", "units": "kg/m^3"},
            "frequency": [
                "h",
                "min",
                "s",
                "ms",
                "us",
                "ns",
            ],
        },
        "meter": {
            "MMTR_SupWh": {"name": "MMTR_SupWh", "dtype": "float", "units": "kWh"},
            "frequency": ["min", "MS", "ME", "D", "ns", "W", "us", "min", "s", "h", "ms"],
        },
        "curtail": {
            "IAVL_ExtPwrDnWh": {"name": "IAVL_ExtPwrDnWh", "dtype": "float", "units": "kWh"},
            "IAVL_DnWh": {"name": "IAVL_DnWh", "dtype": "float", "units": "kWh"},
            "frequency": ["min", "MS", "ME", "D", "ns", "W", "us", "min", "s", "h", "ms"],
        },
        "asset": {
            "latitude": {"name": "latitude", "dtype": "float", "units": "WGS84"},
            "longitude": {"name": "longitude", "dtype": "float", "units": "WGS84"},
            "rated_power": {"name": "rated_power", "dtype": "float", "units": "kW"},
        },
    }

    # A direct comparison is not possible because the frequency ordering is different
    # between the two dictionaries.
    # Check for matching required data types
    assert correct_schema.keys() == combined_schema.keys()
    for key, dict in combined_schema.items():
        # Check that the correct required columns are pulled
        assert correct_schema[key].keys() == dict.keys()
        # Check for matching frequencies
        if key == "asset":
            assert "frequenct" not in correct_schema[key]
            continue
        assert set(correct_schema[key]["frequency"]) == set(dict["frequency"])
