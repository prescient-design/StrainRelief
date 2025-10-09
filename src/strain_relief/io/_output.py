import numpy as np
import pandas as pd
from loguru import logger as logging
from rdkit import Chem

from strain_relief.constants import (
    ENERGY_PROPERTY_NAME,
    ID_COL_NAME,
    MOL_COL_NAME,
    MOL_KEY,
)
from strain_relief.types import MolsDict


def _process_molecule_data(
    id: str,
    local_min_mol: Chem.Mol,
    global_min_mol: Chem.Mol,
    threshold: float,
) -> dict:
    """Helper function to process data for a single molecule."""
    local_min_energy: float = float(np.nan)
    local_min_conf: float = float(np.nan)

    if local_min_mol.GetNumConformers() != 0:
        local_min_energy = local_min_mol.GetConformer().GetDoubleProp(ENERGY_PROPERTY_NAME)
        local_min_conf = local_min_mol.ToBinary()

    global_min_energy: float = float(np.nan)
    global_min_conf: float = float(np.nan)
    conf_energies: list[float] = []

    if global_min_mol.GetNumConformers() != 0:
        conf_energies = [
            conf.GetDoubleProp(ENERGY_PROPERTY_NAME) for conf in global_min_mol.GetConformers()
        ]
        conf_idxs = [conf.GetId() for conf in global_min_mol.GetConformers()]
        min_idx = np.argmin(conf_energies)
        global_min_energy = conf_energies[min_idx]
        global_min_conf = Chem.Mol(global_min_mol, confId=conf_idxs[min_idx]).ToBinary()

    strain: float = float(np.nan)
    if not np.isnan(local_min_energy) and not np.isnan(global_min_energy):
        strain = local_min_energy - global_min_energy
        if strain < 0:
            logging.warning(
                f"{strain:.2f} kcal/mol ligand strain for molecule {id}. Negative ligand strain."
            )
        else:
            logging.debug(f"{strain:.2f} kcal/mol ligand strain for molecule {id}")
    else:
        logging.warning(f"Strain cannot be calculated for molecule {id}")

    return {
        "id": id,
        "local_min_mol": local_min_conf,
        "local_min_e": local_min_energy,
        "global_min_mol": global_min_conf,
        "global_min_e": global_min_energy,
        "ligand_strain": strain,
        "passes_strain_filter": strain <= threshold if threshold is not None else np.nan,
        "nconfs_converged": len(conf_energies),
    }


def save_parquet(
    input_df: pd.DataFrame,
    docked_mols: MolsDict,
    local_min_mols: MolsDict,
    global_min_mols: MolsDict,
    threshold: float,
    parquet_path: str,
    id_col_name: str | None = None,
    mol_col_name: str | None = None,
) -> pd.DataFrame:
    """Creates a df of results and saves to a parquet file using mol.ToBinary().

    Parameters
    ----------
    input_df: pd.DataFrame
        Input DataFrame containing the StrainRelief's original input.
    docked_mols: MolsDict
        Nested dictionary containing the poses of docked molecules.
    local_min_mols: MolsDict
        Nested dictionary containing the poses of locally minimised molecules using strain_relief.
    global_min_mols: MolsDict
        Nested dictionary containing the poses of globally minimised molecules using strain_relief.
    threshold: float
        Threshold for the ligand strain filter.
    parquet_path: str
        Path to the output parquet file.
    id_col_name: str [Optional]
        Name of the column containing the molecule IDs.
    mol_col_name: str [Optional]
        Name of the column containing the RDKit.Mol objects.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the docked and minimum poses of molecules and energies.
    """
    if id_col_name is None:
        id_col_name = ID_COL_NAME
    if mol_col_name is None:
        mol_col_name = MOL_COL_NAME

    dicts = []
    for mol_id in docked_mols.keys():
        dicts.append(
            _process_molecule_data(
                mol_id,
                local_min_mols[mol_id][MOL_KEY],
                global_min_mols[mol_id][MOL_KEY],
                threshold,
            )
        )

    # Define columns upfront to ensure correct order and handle empty DataFrame creation
    result_columns = [
        "id",
        "local_min_mol",
        "local_min_e",
        "global_min_mol",
        "global_min_e",
        "ligand_strain",
        "passes_strain_filter",
        "nconfs_converged",
    ]
    results = pd.DataFrame(dicts, columns=result_columns)

    if not results[results.ligand_strain < 0].empty:
        logging.warning(
            f"{len(results[results.ligand_strain < 0])} molecules have a negative ligand strain, "
            "meaning the initial conformer is lower energy than all generated conformers."
        )
    if not results[results.ligand_strain.isna()].empty:
        logging.warning(
            f"{len(results[results.ligand_strain.isna()])} molecules have no conformers generated "
            "for either the initial or minimised pose, so strain cannot be calculated."
        )

    total_n_confs: int = results["nconfs_converged"].sum() if not results.empty else 0

    if total_n_confs > 0 and not results.empty:
        logging.info(
            f"{total_n_confs:,} configurations converged across {len(results):,} molecules "
            f"(avg. {total_n_confs / len(results):.2f} per molecule)"
        )
    else:
        logging.error(
            "Ligand strain calculation failed for all molecules or no molecules were processed."
        )

    # Merge and drop original molecule column
    final_results = input_df.merge(results, left_on=id_col_name, right_on="id", how="outer")
    final_results.drop(columns=[mol_col_name], inplace=True)

    if parquet_path is not None:
        final_results.to_parquet(parquet_path)
        logging.info(f"Data saved to {parquet_path}")
    else:
        logging.info("Output file not provided, data not saved.")

    return final_results
