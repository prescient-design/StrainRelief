"""Module to validate the hydra config."""

from pathlib import Path

from omegaconf import DictConfig


class ExperimentConfigurationError(Exception):
    """Exception to raise when the experiment configuration is invalid."""

    pass


def _validate_model(cfg: DictConfig):
    if (
        cfg.local_min.method in ["MACE", "FAIRChem"]
        or cfg.global_min.method in ["MACE", "FAIRChem"]
        or cfg.energy_eval.method in ["MACE", "FAIRChem"]
    ):
        if cfg.model.model_paths is None:
            raise ExperimentConfigurationError("Model path must be provided if using a NNP")

        if not Path(cfg.model.model_paths).exists():
            raise ExperimentConfigurationError(f"Model path {cfg.model.model_paths} does not exist")


def _validate_config(cfg: DictConfig) -> DictConfig:
    """
    Validate the config and make any necessary alterations to the parameters.
    """
    _validate_model(cfg)

    return cfg
