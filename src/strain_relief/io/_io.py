import logging

import numpy as np
import pandas as pd
from rdkit import Chem

from strain_relief.constants import ENERGY_PROPERTY_NAME, ID_COL_NAME, MOL_COL_NAME


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

    logging.info(f"Loading data...")
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

    if "charge" not in df.columns:  # needed for deployment code
        if mol_col_name not in df.columns:
            df[mol_col_name] = df["mol_bytes"].apply(Chem.Mol)
        df = _calculate_charge(df, mol_col_name)

    return {r[id_col_name]: r[mol_col_name] for _, r in df[df.charge == 0].iterrows()}


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
        raise ValueError(f"Column 'mol_bytes' not found in dataframe")
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
    md: MolDataset
        MolDataset containing molecules.

    Returns
    -------
        MolDataset containing only neutral molecules.
    """
    df["charge"] = df[mol_col_name].apply(lambda x: Chem.GetFormalCharge(x))
    if len(df[df["charge"] != 0]) > 0:
        logging.info(
            f"Dataset contains {len(df[df['charge'] != 0])} charged molecules. Ligand strains will not be calculated for these."
        )
    return df


def save_parquet(
    input_df: pd.DataFrame,
    docked_mols: dict,
    local_min_mols: dict,
    global_min_mols: dict,
    threshold: float,
    output_file: str,
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
    output_file: str
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

    for id in docked_mols.keys():
        docked_mol = docked_mols[id]
        local_min_mol = local_min_mols[id]
        global_min_mol = global_min_mols[id]

        # Initial energy if local_min_mol has conformers
        if local_min_mol.GetNumConformers() != 0:
            local_min_energy = local_min_mol.GetConformer().GetDoubleProp(ENERGY_PROPERTY_NAME)
            local_min_conf = local_min_mol.ToBinary()
        else:
            local_min_energy, local_min_conf = np.nan, np.nan

        # Minimised energy and conformer if global_min_mol has conformers
        if global_min_mol.GetNumConformers() != 0:
            conf_energies = [
                conf.GetDoubleProp(ENERGY_PROPERTY_NAME) for conf in global_min_mol.GetConformers()
            ]
            conf_idxs = [conf.GetId() for conf in global_min_mol.GetConformers()]
            min_idx = np.argmin(conf_energies)
            global_min_energy = conf_energies[min_idx]
            global_min_conf = Chem.Mol(global_min_mol, confId=conf_idxs[min_idx]).ToBinary()
        else:
            global_min_energy, global_min_conf, conf_energies = np.nan, np.nan, []

        # Ligand strain if both local_min_mol and global_min_mol have conformers
        if local_min_mol.GetNumConformers() != 0 and global_min_mol.GetNumConformers() != 0:
            strain = local_min_energy - global_min_energy
            if strain < 0:
                logging.warning(
                    f"{strain} kcal/mol ligand strain for molecule {id}. Negative ligand strain."
                )
            else:
                logging.debug(f"{strain} kcal/mol ligand strain for molecule {id}")
        else:
            strain = np.nan
            logging.warning(f"Strain cannot be calculated for molecule {id}")

        dicts.append(
            {
                "id": id,
                "docked_mol": docked_mol.ToBinary(),
                "local_min_mol": local_min_conf,
                "local_min_e": local_min_energy,
                "global_min_mol": global_min_conf,
                "global_min_e": global_min_energy,
                "ligand_strain": strain,
                "passes_strain_filter": strain <= threshold,
                "nconfs_converged": len(conf_energies),
            }
        )

    results = pd.DataFrame(dicts)

    if len(results[results.ligand_strain < 0]) > 0:
        logging.warning(
            f"{len(results[results.ligand_strain < 0])} molecules have a negative ligand strain i.e. the initial conformer is lower energy than all generated conformers."
        )
    if len(results[results.ligand_strain.isna()]) > 0:
        logging.warning(
            f"{len(results[results.ligand_strain.isna()])} molecules have no conformers generated for either the initial or minimised pose."
        )

    results = input_df.merge(results, left_on=id_col_name, right_on="id", how="outer")
    results.drop(columns=[mol_col_name], inplace=True)

    if output_file is not None:
        results.to_parquet(output_file)
        logging.info(f"Data saved to {output_file}")
    else:
        logging.info("Output file not provided, data not saved")

    return results
