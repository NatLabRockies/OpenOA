import sys
import random
from pathlib import Path

import numpy as np
import pytest


examples_folder = Path(__file__).resolve().parents[1]
sys.path.append(examples_folder)
from examples import project_ENGIE, example_data_path_str  # ruff: ignore[E402]


ROOT = Path(__file__).parent


def pytest_addoption(parser):
    parser.addoption("--unit", action="store_true", default=False, help="run tests in test/unit/.")
    parser.addoption(
        "--regression", action="store_true", default=False, help="run tests in test/regression/."
    )


def pytest_configure(config):
    # Check for the options
    unit = config.getoption("--unit")
    regression = config.getoption("--regression")

    # Provide the appropriate directories
    unit_tests = [str(el) for el in (ROOT / "unit").iterdir() if el.suffix == ".py"]
    regression_tests = [str(el) for el in (ROOT / "regression").iterdir() if el.suffix == ".py"]

    # If both, run them all; if neither skip any modifications; otherwise run just the appropriate subset
    if regression and unit:
        config.args = unit_tests + regression_tests
    elif regression:
        config.args = regression_tests
    elif unit:
        config.args = unit_tests


def reset_prng():
    np.random.seed(42)
    random.seed(42)


@pytest.fixture(scope="module")
def project():
    """Load and prepare the ENGIE plant data for regression testing."""
    reset_prng()

    def _project(analysis_type: str | list[str]):
        project = project_ENGIE.prepare(example_data_path_str, use_cleansed=False)

        if analysis_type is not None:
            if isinstance(analysis_type, str):
                analysis_type = [analysis_type]
            project.analysis_type.extend(analysis_type)

        project.validate()
        return project

    return _project
