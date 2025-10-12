"""Module to validate the hydra config."""

from pathlib import Path

from omegaconf import DictConfig


class ExperimentConfigurationError(Exception):
    """Exception to raise when the experiment configuration is invalid."""

    pass


def _validate_model(cfg: DictConfig):
    """Ensures model paths is provdied if a NNP model is being used."""
    if cfg.calculator.model_paths is None:
        raise ExperimentConfigurationError("Model path must be provided if using a NNP")

    if not Path(cfg.calculator.model_paths).exists():
        raise ExperimentConfigurationError(f"Model path {cfg.model.model_paths} does not exist")

    if cfg.get("energy_evaluation", None):
        if cfg.energy_evaluation.calculator.model_paths is None:
            raise ExperimentConfigurationError("Model path must be provided if using a NNP")
        elif not Path(cfg.energy_evaluation.calculator.model_paths).exists():
            raise ExperimentConfigurationError(
                f"Model path {cfg.energy_evaluation.model_paths} does not exist"
            )


def _validate_batch(cfg: DictConfig) -> DictConfig:
    """Validate the batch size and set to a large number if -1 (i.e. no batching)."""
    if cfg.batch_size == -1:
        cfg.batch_size = 1_000_000
    return cfg


def _validate_config(cfg: DictConfig) -> DictConfig:
    """Validate the config and make any necessary alterations to the parameters."""
    _validate_model(cfg)
    cfg = _validate_batch(cfg)

    return cfg
