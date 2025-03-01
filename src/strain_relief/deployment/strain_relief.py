from dataclasses import dataclass

import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from strain_relief.cmdline._strain_relief import strain_relief


@dataclass
class DeploymentConfig:
    seed: int = -1
    threshold: float = 12.0
    num_confs: int = 20
    minimisation: str = "mace"
    max_iters: int = 125
    model_path: str = "s3://prescient-data-dev/strain_relief/models/MACE.model"


def deployment_function(dataframe: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Strain Relief calculates the ligand strain of docked poses and has a suite of different force fields with which to do this.
    This includes a MACE neural network potential.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe with:
            - ID column: id
            - mol column: mol_bytes (rdkit Mol object as bytes)

            For scoring and ranking models, the ID column is treated as a unique identifier.
            The output must preserve the ID column - it is used to map the output scores back to the input.

    config : DictConfig
        OmegaConf's DictConfig object with fields defined in DeploymentConfig

    Returns
    -------
    pd.DataFrame
        Output dataframe with:

            - id column: id (same as input)
            - new output columns: ligand_strain, passes_strain_filter

    """
    with initialize(version_base="1.1", config_path="../config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"seed={config.seed}",
                f"threshold={config.threshold}",
                "model=mace",
                f"model.model_paths={config.model_path}",
                f"conformers.numConfs={config.num_confs}",
                f"minimisation@local_min={config.minimisation}",
                f"minimisation@global_min={config.minimisation}",
                f"local_min.maxIters={config.max_iters}",
                f"global_min.maxIters={config.max_iters}",
                f"local_min.model_paths={config.model_path}",
                f"global_min.model_paths={config.model_path}",
                "local_min.fmax=0.50",
            ],
        )
        OmegaConf.resolve(cfg)
    df = strain_relief(dataframe, cfg)
    output_columns = dataframe.columns.tolist() + ["ligand_strain", "passes_strain_filter"]
    if "mol" in output_columns:
        output_columns.remove("mol")
    return df[output_columns]
