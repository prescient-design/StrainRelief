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

from collections.abc import Sequence
from timeit import default_timer as timer

import hydra
import pandas as pd
import rich
import rich.syntax
import rich.tree
from loguru import logger
from neural_optimiser.calculators.base import Calculator
from neural_optimiser.conformers import Conformer, ConformerBatch
from neural_optimiser.optimisers.base import Optimiser
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem

from strain_relief.configs import _validate_config
from strain_relief.conformers import generate_conformers
from strain_relief.io import process_output, to_mols_dict
from strain_relief.optimisation import run_optimisation
from strain_relief.types import MolsDict


def compute_strain(
    cfg: DictConfig,
    df: pd.DataFrame | None = None,
    mols: Sequence[Chem.Mol | bytes] | None = None,
    ids: Sequence[int | str] | None = None,
) -> pd.DataFrame:
    """Calculate ligand strain energies using rkdit conformer generation.

    One (and only one) of df or mols must be provided. If mols are provided they must
    be all RDKit Mol objects or all bytes (Mol.ToBinary()).

    Parameters
    ----------
    cfg: DictConfig
        DictConfig object containing the hydra configuration.
    df: pd.DataFrame [Optional]
        Input dataframe without rdkit.Mol objects.
    mols: Sequence[Chem.Mol|bytes] [Optional]
        List of molecules (RDKit.Mol or bytes).
    ids: Sequence[int|str] [Optional]
        List of unique molecule ids.

    Returns
    -------
    pd.DataFrame
        Dataframe with strain energies and other metadata.
    """
    start = timer()

    # -------------- CONFIGURATION --------------

    _print_config_tree(cfg)
    _validate_config(cfg)  # TODO: more checks here

    # -------------- SET-UP --------------

    # Instantiate calculator
    logger.info("Instantiating calculator...")
    calculator: Calculator = hydra.utils.instantiate(
        cfg.calculator
    )  # TODO: add default_dtype to calculator functionality
    logger.info(calculator)

    # Instantiate energy evaluation calculator (if different from minimisation)
    if cfg.get("energy_evaluation", None):
        logger.info("Instantiating energy evaluation calculator...")
        energy_calculator: Calculator = hydra.utils.instantiate(cfg.energy_evaluation.calculator)
        logger.info(energy_calculator)

    # Instantiate local optimiser
    logger.info("Instantiating local optimiser...")
    local_optimiser: Optimiser = hydra.utils.instantiate(cfg.local_optimiser)
    logger.info(local_optimiser)

    # Instantiate global optimiser
    logger.info("Instantiating global optimiser...")
    global_optimiser: Optimiser = hydra.utils.instantiate(cfg.global_optimiser)
    logger.info(global_optimiser)

    local_optimiser.calculator = calculator
    global_optimiser.calculator = calculator

    # -------------- PROCESS MOLECULES --------------

    df = _parse_args(df=df, mols=mols, ids=ids)

    docked_mols: MolsDict = to_mols_dict(df, **cfg.io.input)  # move to conformer dir?
    docked_batch: ConformerBatch = ConformerBatch.from_data_list(
        [Conformer.from_rdkit(**docked_mols[id]) for id in docked_mols]
    )

    logger.info("Generating conformers for global minimum search...")
    generated_mols = generate_conformers(docked_mols, **cfg.conformers)
    generated_batch: ConformerBatch = ConformerBatch.cat(
        [ConformerBatch.from_rdkit(**generated_mols[id]) for id in generated_mols]
    )

    logger.info("Minimising docked conformers...")
    local_minima = run_optimisation(
        docked_batch.clone(), local_optimiser, cfg.batch_size, cfg.num_workers
    )

    logger.info("Minimising generated conformers...")
    global_minima = run_optimisation(
        generated_batch, global_optimiser, cfg.batch_size, cfg.num_workers
    )

    if cfg.get("energy_evaluation", None):  # TODO: update config for this to be optional
        logger.info("Predicting energies of local minima poses...")
        local_minima = energy_calculator.get_energy(local_minima)
        logger.info("Predicting energies of generated conformers...")
        global_minima = energy_calculator.get_energy(global_minima)

    # Save ligand strains
    md = process_output(
        df, docked_batch, local_minima, global_minima, cfg.threshold, **cfg.io.output
    )

    end = timer()
    logger.info(f"Ligand strain calculations took {end - start:.2f} seconds. \n")

    return md


def _parse_args(
    df: pd.DataFrame | None = None,
    mols: Sequence[Chem.Mol | bytes] | None = None,
    ids: Sequence[int | str] | None = None,
) -> pd.DataFrame:
    """Normalise input into a dataframe with columns ['id', 'mol_bytes'].

     Precedence:
    1. If df is provided it is copied and returned (mols / ids ignored).
    2. Otherwise mols must be provided and be homogeneous (all Chem.Mol or all bytes).
       If ids not supplied they are auto-generated (0..n-1). RDKit Mol objects can be
       converted to binary via Mol.ToBinary().
    """
    if df is None and mols is None:
        raise ValueError("Either df or mols must be provided")

    if df is not None:  # prevents input df from being updated
        if mols is not None:
            logger.warning("compute_strain received both df and mols; using df and ignoring mols.")
        return df.copy()

    if not ids:
        ids = list(range(len(mols)))

    mol_types = set(type(mol) for mol in mols)
    if len(mol_types) > 1:
        raise ValueError("All molecules must be of the same type (Chem.Mol or bytes)")

    if Chem.Mol in mol_types:
        mols = [mol.ToBinary() for mol in mols]

    return pd.DataFrame({"id": ids, "mol_bytes": mols})


def _print_config_tree(
    cfg: DictConfig,
    resolve: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Parameters
    ----------
    cfg: DictConfig
        The configuration to be printed.
    resolve: bool
        Whether to resolve interpolations in the configuration.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    # Generate config tree
    for field in cfg:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # Print config tree
    rich.print(tree)


if __name__ == "__main__":

    @hydra.main(version_base=None, config_path="../../hydra_config", config_name="default")
    def main(cfg: DictConfig) -> pd.DataFrame:
        cfg.calculator.model_paths = "./models/MACE_SPICE2_NEUTRAL.model"
        cfg.io.input.parquet_path = None
        cfg.conformers.numConfs = 1
        cfg.local_optimiser.fmax = 0.5

        from omegaconf import OmegaConf

        OmegaConf.resolve(cfg)

        df = pd.read_parquet("./data/example_ligboundconf_input.parquet")
        return compute_strain(cfg, df=df)

    main()
