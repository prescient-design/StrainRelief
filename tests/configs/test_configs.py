from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from strain_relief.configs._configs import (
    ExperimentConfigurationError,
    _validate_batch,
    _validate_calculator,
    _validate_model,
    _validate_optimiser,
    _validate_paths,
    validate_config,
)


@pytest.fixture()
def cfg(tmp_path: Path) -> DictConfig:
    """Return a base configuration for testing."""
    input_file = tmp_path / "input" / "data.parquet"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    input_file.touch()

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file_1 = tmp_path / "model1.pt"
    model_file_1.touch()

    model_file_2 = tmp_path / "model2.pt"
    model_file_2.touch()

    with initialize(config_path="../../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={input_file}",
                f"io.output.parquet_path={output_dir / 'results.parquet'}",
                "io.input.include_charged=false",
                "calculator._target_=some.calculator.Class",
                f"calculator.model_paths={model_file_1}",
                "local_optimiser.fmax=5.0",
                "local_optimiser.fexit=10.0",
                "global_optimiser.fmax=10.0",
                "global_optimiser.fexit=20.0",
                "batch_size=100",
                "+energy_evaluation.calculator._target_=other.calculator.Class",
                f"+energy_evaluation.calculator.model_paths={model_file_2}",
            ],
        )

    return cfg


def test_validate_paths_success(cfg):
    """Test that a valid path configuration passes."""
    print(cfg)
    try:
        _validate_paths(cfg)
    except ExperimentConfigurationError:
        pytest.fail("_validate_paths raised ExperimentConfigurationError unexpectedly")


def test_validate_paths_input_not_exist(cfg):
    """Test that an error is raised if the input path does not exist."""
    cfg = deepcopy(cfg)
    cfg.io.input.parquet_path = "/tmp/non/existent/file.parquet"
    with pytest.raises(ExperimentConfigurationError, match="Input path .* does not exist"):
        _validate_paths(cfg)


def test_validate_paths_output_dir_not_exist(cfg):
    """Test that an error is raised if the output directory does not exist."""
    cfg = deepcopy(cfg)
    cfg.io.output.parquet_path = "/tmp/non/existent/dir/results.parquet"
    with pytest.raises(ExperimentConfigurationError, match="Output directory .* does not exist"):
        _validate_paths(cfg)


# --- Tests for _validate_model ---


def test_validate_model_success(cfg):
    """Test that a valid model configuration passes."""
    try:
        _validate_model(cfg)
    except ExperimentConfigurationError:
        pytest.fail("_validate_model raised ExperimentConfigurationError unexpectedly")


def test_validate_model_calculator_path_none(cfg):
    """Test error when main calculator model_paths is None."""
    cfg = deepcopy(cfg)
    cfg.calculator.model_paths = None
    with pytest.raises(ExperimentConfigurationError, match="Model path must be provided"):
        _validate_model(cfg)


def test_validate_model_calculator_path_not_exist(cfg):
    """Test error when main calculator model_paths does not exist."""
    cfg = deepcopy(cfg)
    cfg.calculator.model_paths = "/tmp/non/existent/model.pt"
    with pytest.raises(ExperimentConfigurationError, match="Model path .* does not exist"):
        _validate_model(cfg)


def test_validate_model_energy_eval_path_none(cfg):
    """Test error when energy_evaluation calculator model_paths is None."""
    cfg = deepcopy(cfg)
    cfg.energy_evaluation.calculator.model_paths = None
    with pytest.raises(ExperimentConfigurationError, match="Model path must be provided"):
        _validate_model(cfg)


def test_validate_model_energy_eval_path_not_exist(cfg):
    """Test error when energy_evaluation calculator model_paths does not exist."""
    cfg = deepcopy(cfg)
    cfg.energy_evaluation.calculator.model_paths = "/tmp/non/existent/model.pt"
    with pytest.raises(ExperimentConfigurationError, match="Model path .* does not exist"):
        _validate_model(cfg)


def test_validate_model_no_energy_eval(cfg):
    """Test success when no energy_evaluation config is present."""
    cfg = OmegaConf.to_container(deepcopy(cfg))
    # Use del to remove the key, simulating it not being in the config
    del cfg["energy_evaluation"]
    cfg = OmegaConf.create(cfg)
    try:
        _validate_model(cfg)
    except ExperimentConfigurationError:
        pytest.fail("_validate_model raised ExperimentConfigurationError unexpectedly")


# --- Tests for _validate_optimiser ---


def test_validate_optimiser_success(cfg):
    """Test that a valid optimiser configuration passes."""
    try:
        _validate_optimiser(cfg)
    except ExperimentConfigurationError:
        pytest.fail("_validate_optimiser raised ExperimentConfigurationError unexpectedly")


def test_validate_optimiser_local_invalid(cfg):
    """Test error when local optimiser fexit < fmax."""
    cfg = deepcopy(cfg)
    cfg.local_optimiser.fexit = 4.0
    cfg.local_optimiser.fmax = 5.0
    with pytest.raises(
        ExperimentConfigurationError, match="Local optimiser fexit must be greater than fmax"
    ):
        _validate_optimiser(cfg)


def test_validate_optimiser_global_invalid(cfg):
    """Test error when global optimiser fexit < fmax."""
    cfg = deepcopy(cfg)
    cfg.global_optimiser.fexit = 9.0
    cfg.global_optimiser.fmax = 10.0
    with pytest.raises(
        ExperimentConfigurationError, match="Global optimiser fexit must be greater than fmax"
    ):
        _validate_optimiser(cfg)


# --- Tests for _validate_calculator ---


def test_validate_calculator_success(cfg):
    """Test that a valid calculator configuration passes."""
    try:
        _validate_calculator(cfg)
    except ExperimentConfigurationError:
        pytest.fail("_validate_calculator raised ExperimentConfigurationError unexpectedly")


def test_validate_calculator_same_calculators(cfg):
    """Test error when energy_evaluation calculator is same as main calculator."""
    cfg = deepcopy(cfg)
    cfg.energy_evaluation.calculator = cfg.calculator
    with pytest.raises(
        ExperimentConfigurationError, match="Energy evaluation calculator is the same"
    ):
        _validate_calculator(cfg)


def test_validate_calculator_no_energy_eval(cfg):
    """Test success when no energy_evaluation config is present."""
    cfg = OmegaConf.to_container(deepcopy(cfg))
    del cfg["energy_evaluation"]
    cfg = OmegaConf.create(cfg)
    try:
        _validate_calculator(cfg)
    except ExperimentConfigurationError:
        pytest.fail("_validate_calculator raised ExperimentConfigurationError unexpectedly")


# --- Tests for _validate_batch ---


def test_validate_batch_minus_one(cfg):
    """Test that batch_size = -1 is converted to a large number."""
    cfg = deepcopy(cfg)
    cfg.batch_size = -1

    updated_cfg = _validate_batch(cfg)

    assert updated_cfg.batch_size == 1_000_000_000
    assert cfg.batch_size == 1_000_000_000  # Check if modified in place


def test_validate_batch_positive_number(cfg):
    """Test that a positive batch_size remains unchanged."""
    cfg = deepcopy(cfg)
    cfg.batch_size = 50

    updated_cfg = _validate_batch(cfg)

    assert updated_cfg.batch_size == 50
    assert cfg.batch_size == 50


# --- Tests for validate_config (main function) ---


def test_validate_config_success(cfg):
    """Test the main validation function with a completely valid config."""
    try:
        updated_cfg = validate_config(cfg)
        # Check if batch logic was applied (though not changing in this case)
        assert updated_cfg.batch_size == 100
    except ExperimentConfigurationError:
        pytest.fail("validate_config raised ExperimentConfigurationError unexpectedly")


def test_validate_config_failure(cfg):
    """Test that the main validation function fails if a sub-validation fails."""
    cfg = deepcopy(cfg)
    # Introduce an error (invalid path)
    cfg.io.input.parquet_path = "/tmp/non/existent/file.parquet"

    with pytest.raises(ExperimentConfigurationError, match="Input path .* does not exist"):
        validate_config(cfg)


def test_validate_config_batch_update(cfg):
    """Test that the main validation function returns the modified config."""
    cfg = deepcopy(cfg)
    cfg.batch_size = -1

    updated_cfg = validate_config(cfg)

    assert updated_cfg.batch_size == 1_000_000_000
