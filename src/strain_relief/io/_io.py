import numpy as np
import pandas as pd
from loguru import logger as logging
from rdkit import Chem

from strain_relief.constants import CHARGE_COL_NAME, ENERGY_PROPERTY_NAME, ID_COL_NAME, MOL_COL_NAME


def load_parquet(
    parquet_path: str,
    id_col_name: str | None = None,
    mol_col_name: str | None = None,
) -> pd.DataFrame:
    """Load a parquet file containing molecules.

    Parameters
    ----------
    parquet_path: str
        Path to the parquet file containing the molecules.
    id_col_name: str
        Name of the column containing the molecule IDs.
    mol_col_name: str
        Name of the column containing the RDKit.Mol objects OR binary string.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the molecules.
    """
    if mol_col_name is None:
        mol_col_name = MOL_COL_NAME
    if id_col_name is None:
        id_col_name = ID_COL_NAME

    logging.info("Loading data...")
    df = pd.read_parquet(parquet_path)
    logging.info(f"Loaded {len(df)} posed molecules")

    _check_columns(df, mol_col_name, id_col_name)
    df = _calculate_charge(df, mol_col_name)

    return df


def to_mols_dict(df: pd.DataFrame, mol_col_name: str, id_col_name: str) -> dict:
    """Converts a DataFrame to a dictionary of RDKit.Mol objects.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing molecules.
    mol_col_name: str
        Name of the column containing the RDKit.Mol objects OR binary strings.
    id_col_name: str
        Name of the column containing the molecule IDs.

    Returns
    -------
    dict
        Dictionary containing the molecule IDs and RDKit.Mol objects.
    """
    if mol_col_name is None:
        mol_col_name = MOL_COL_NAME
    if id_col_name is None:
        id_col_name = ID_COL_NAME

    if mol_col_name not in df.columns:  # needed for deployment code
        df[mol_col_name] = df["mol_bytes"].apply(Chem.Mol)

    if CHARGE_COL_NAME not in df.columns:  # needed for deployment code
        df = _calculate_charge(df, mol_col_name)

    return {r[id_col_name]: r[mol_col_name] for _, r in df[df[CHARGE_COL_NAME] == 0].iterrows()}


def _check_columns(df: pd.DataFrame, mol_col_name: str, id_col_name: str):
    """Check if the required columns are present in the dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing molecules.
    mol_col_name: str
        Name of the column containing the RDKit.Mol objects OR binary strings.
    id_col_name: str
        Name of the column containing the molecule IDs.
    """
    if "mol_bytes" not in df.columns:
        raise ValueError("Column 'mol_bytes' not found in dataframe")
    df[mol_col_name] = df["mol_bytes"].apply(Chem.Mol)
    logging.info(f"RDKit.Mol column is '{mol_col_name}'")

    if id_col_name not in df.columns:
        raise ValueError(f"Column '{id_col_name}' not found in dataframe")
    if not df[id_col_name].is_unique:
        raise ValueError(f"ID column ({id_col_name}) contains duplicate values")
    logging.info(f"ID column is '{id_col_name}'")


def _calculate_charge(df: pd.DataFrame, mol_col_name: str) -> pd.DataFrame:
    """Calculate charge of molecules.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing molecules.

    Returns
    -------
        DataFrame with charge column.
    """
    df[CHARGE_COL_NAME] = df[mol_col_name].apply(lambda x: int(Chem.GetFormalCharge(x)))
    if all(df[CHARGE_COL_NAME] != 0):
        logging.error(
            # raise ValueError(
            "All molecules are charged. StrainRelief only calculates ligand strain for neutral "
            "molecules."
        )
    elif any(df[CHARGE_COL_NAME] != 0):
        logging.info(
            f"Dataset contains {len(df[df[CHARGE_COL_NAME] != 0])} charged molecules. Ligand "
            "strains will not be calculated for these."
        )
    return df


def _process_molecule_data(
    id: str,
    local_min_mol: Chem.Mol,
    global_min_mol: Chem.Mol,
    threshold: float,
) -> dict:
    """Helper function to process data for a single molecule."""
    local_min_energy, local_min_conf = np.nan, np.nan

    if local_min_mol.GetNumConformers() != 0:
        local_min_energy = local_min_mol.GetConformer().GetDoubleProp(ENERGY_PROPERTY_NAME)
        local_min_conf = local_min_mol.ToBinary()

    global_min_energy, global_min_conf, conf_energies = np.nan, np.nan, []
    if global_min_mol.GetNumConformers() != 0:
        conf_energies = [
            conf.GetDoubleProp(ENERGY_PROPERTY_NAME) for conf in global_min_mol.GetConformers()
        ]
        conf_idxs = [conf.GetId() for conf in global_min_mol.GetConformers()]
        min_idx = np.argmin(conf_energies)
        global_min_energy = conf_energies[min_idx]
        global_min_conf = Chem.Mol(global_min_mol, confId=conf_idxs[min_idx]).ToBinary()

    strain = np.nan
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
    docked_mols: dict,
    local_min_mols: dict,
    global_min_mols: dict,
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
    docked_mols: dict
        Dictionary containing the poses of docked molecules.
    local_min_mols: dict
        Dictionary containing the poses of locally minimised molecules using strain_relief.
    global_min_mols: dict
        Dictionary containing the poses of globally minimised molecules using strain_relief.
    threshold: float
        Threshold for the ligand strain filter.
    parquet_path: str
        Path to the output parquet file.
    id_col_name: str
        Name of the column containing the molecule IDs.
    mol_col_name: str
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
                local_min_mols[mol_id],
                global_min_mols[mol_id],
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

    total_n_confs = results["nconfs_converged"].sum() if not results.empty else 0

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
