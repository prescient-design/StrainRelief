"""Module to validate the hydra config."""

from pathlib import Path

from loguru import logger
from omegaconf import DictConfig


class ExperimentConfigurationError(Exception):
    """Exception to raise when the experiment configuration is invalid."""

    pass


def _validate_paths(cfg: DictConfig):
    """Ensures input and output paths are valid."""
    input_path = Path(cfg.io.input.parquet_path) if cfg.io.input.parquet_path is not None else None

    if input_path and not input_path.exists():
        raise ExperimentConfigurationError(f"Input path {input_path} does not exist")

    output_path = (
        Path(cfg.io.output.parquet_path) if cfg.io.output.parquet_path is not None else None
    )

    if output_path is None:
        logger.warning("No output path provided, results will not be saved to disk")
    elif not output_path.parent.exists():
        raise ExperimentConfigurationError(f"Output directory {output_path.parent} does not exist")


def _validate_model(cfg: DictConfig):
    """Ensures model paths is provdied if a NNP model is being used."""
    if hasattr(cfg.calculator, "model_paths"):
        if cfg.calculator.model_paths is None:
            raise ExperimentConfigurationError("Model path must be provided if using a NNP")

        if "s3" not in cfg.calculator.model_paths and not Path(cfg.calculator.model_paths).exists():
            raise ExperimentConfigurationError(
                f"Model path {cfg.calculator.model_paths} does not exist"
            )

    if cfg.get("energy_evaluation", None):
        if cfg.energy_evaluation.calculator.model_paths is None:
            raise ExperimentConfigurationError("Model path must be provided if using a NNP")
        elif not Path(cfg.energy_evaluation.calculator.model_paths).exists():
            raise ExperimentConfigurationError(
                f"Model path {cfg.energy_evaluation.calculator.model_paths} does not exist"
            )


def _validate_optimiser(cfg: DictConfig):
    """Ensures optimiser parameters are valid."""
    if cfg.local_optimiser.fexit < cfg.local_optimiser.fmax:
        raise ExperimentConfigurationError("Local optimiser fexit must be greater than fmax")
    if cfg.global_optimiser.fexit < cfg.global_optimiser.fmax:
        raise ExperimentConfigurationError("Global optimiser fexit must be greater than fmax")


def _validate_calculator(cfg: DictConfig):
    """Ensures calculator parameters are valid."""
    opt_calculator = cfg.calculator._target_
    e_calculator = (
        cfg.energy_evaluation.calculator._target_ if cfg.get("energy_evaluation", None) else None
    )

    if opt_calculator == e_calculator:
        raise ExperimentConfigurationError(
            "Energy evaluation calculator is the same as minimisation calculator. This is "
            "duplcating computation."
        )

    if (
        "neural_optimiser.calculators.MACECalculator" in [opt_calculator, e_calculator]
        and cfg.io.input.include_charged
    ):
        logger.warning("MACE (v0.3.14) currently has limited support for charged molecules.")

    if (
        "neural_optimiser.calculators.MMFF94Calculator" in [opt_calculator, e_calculator]
        and cfg.batch_size != 1
    ):
        raise ExperimentConfigurationError("MMFF94 calculator only supports batch size of 1.")


def _validate_batch(cfg: DictConfig) -> DictConfig:
    """Validate the batch size and set to a large number if -1 (i.e. no batching)."""
    if cfg.batch_size == -1:
        cfg.batch_size = 1_000_000_000
    return cfg


def validate_config(cfg: DictConfig) -> DictConfig:
    """Validate the config and make any necessary alterations to the parameters."""
    _validate_paths(cfg)
    _validate_model(cfg)
    _validate_calculator(cfg)
    _validate_optimiser(cfg)
    cfg = _validate_batch(cfg)

    return cfg
