####################################################################################################
# This is the script for StrainRelief calculate ligand strain using a given force field            #
#                                                                                                  #
# ALGORITHM:                                                                                       #
# 1. Read in molecules(s) from df                                                                  #
# 2. Calculate the local minimum conformer by minimising the docked pose with a loose convergence  #
#    criteria                                                                                      #
# 2. Generate n conformers for each molecule                                                       #
# 3. Minimise each conformation and choose the lowest as an approximation for the global minimum   #
# 4. (ONLY IF USING A DIFFFERENT FF FOR ENERGIES) Predict energy of each conformation              #
# 5. Calculate ligand strain between local and global minimum and apply threshold                  #
#####################################################################################################

from copy import deepcopy
from timeit import default_timer as timer

import hydra
import pandas as pd
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

from strain_relief.conformers import generate_conformers
from strain_relief.energy_eval import predict_energy
from strain_relief.io import load_parquet, save_parquet, to_mols_dict
from strain_relief.minimisation import minimise_conformers


def strain_relief(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Calculate torsionsal strain energies using rkdit conformer generation.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe without rdkit.Mol objects.
    cfg: OmegaConf
        OmegaConf object containing the hydra configuration.

    Returns
    -------
    pd.DataFrame
    """
    start = timer()
    logging.info(f"STRAIN RELIEF RUN CONFIG: \n{OmegaConf.to_yaml(cfg)}")

    if (
        cfg.local_min.method == "MACE"
        or cfg.global_min.method == "MACE"
        or cfg.energy_eval.method == "MACE"
    ) and cfg.model.model_paths is None:
        raise ValueError("Model path must be provided if using MACE")

    # Load data
    docked = to_mols_dict(df, cfg.io.input.mol_col_name, cfg.io.input.id_col_name)
    local_minima = {id: deepcopy(mol) for id, mol in docked.items()}
    global_minimia = {id: deepcopy(mol) for id, mol in docked.items()}

    # Find the local minimum using a looser convergence criteria
    logging.info("Minimising docked conformer...")
    local_minima = minimise_conformers(local_minima, **cfg.local_min)

    # Generate conformers from the docked conformer
    global_minimia = generate_conformers(global_minimia, **cfg.conformers)

    # Find approximate global minimum from generated conformers
    logging.info("Minimising generated conformers...")
    global_minimia = minimise_conformers(global_minimia, **cfg.global_min)

    # Predict single point energies (if using a different method from minimisation)
    if (
        cfg.local_min.method != cfg.energy_eval.method
        or cfg.global_min.method != cfg.energy_eval.method
    ):
        logging.info("Predicting energies of local minima poses...")
        local_minima = predict_energy(local_minima, **cfg.energy_eval)
        logging.info("Predicting energies of generated conformers...")
        global_minimia = predict_energy(global_minimia, **cfg.energy_eval)

    # Save torsional strains
    md = save_parquet(df, docked, local_minima, global_minimia, cfg.threshold, **cfg.io.output)

    end = timer()
    logging.info(f"Ligand strain calculations took {end - start:.2f} seconds. \n")

    return md


@hydra.main(version_base=None, config_path="../hydra_config", config_name="default")
def main(cfg: DictConfig):
    df = load_parquet(**cfg.io.input)
    return strain_relief(df, cfg)


if __name__ == "__main__":
    torsions = main()
