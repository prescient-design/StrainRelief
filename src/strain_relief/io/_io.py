import numpy as np
import pandas as pd
from loguru import logger as logging
from rdkit import Chem

from strain_relief.constants import (
    CHARGE_COL_NAME,
    CHARGE_KEY,
    ENERGY_PROPERTY_NAME,
    ID_COL_NAME,
    MOL_COL_NAME,
    MOL_KEY,
    SPIN_COL_NAME,
    SPIN_KEY,
)
from strain_relief.types import MolsDict


def load_parquet(
    parquet_path: str,
    include_charged: bool | None = None,
    id_col_name: str | None = None,
    mol_col_name: str | None = None,
) -> pd.DataFrame:
    """Load a parquet file containing molecules.

    Parameters
    ----------
    parquet_path: str
        Path to the parquet file containing the molecules.
    include_charged: bool [Optional]
        If False, filters out charged molecules.
    id_col_name: str [Optional]
        Name of the column containing the molecule IDs.
    mol_col_name: str [Optional]
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
    df = _calculate_charge(df, mol_col_name, include_charged)
    df = _calculate_spin(df, mol_col_name)
    return df


def to_mols_dict(
    df: pd.DataFrame,
    mol_col_name: str,
    id_col_name: str,
    include_charged: bool,
    parquet_path: str | None = None,
) -> MolsDict:
    """Converts a DataFrame to a dictionary of RDKit.Mol objects.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing molecules.
    mol_col_name: str
        Name of the column containing the RDKit.Mol objects OR binary strings.
    id_col_name: str
        Name of the column containing the molecule IDs.
    include_charged: bool
        If False, filters out charged molecules.
    parquet_path: str
        [PLACEHOLDER] Needed for simplicity of arg parsing.

    Returns
    -------
    MolsDict
        Dictionary containing the molecule IDs, RDKit.Mol objects, charges and spins.
    """
    if mol_col_name is None:
        mol_col_name = MOL_COL_NAME
    if id_col_name is None:
        id_col_name = ID_COL_NAME

    if mol_col_name not in df.columns:  # needed for deployment code
        df[mol_col_name] = df["mol_bytes"].apply(Chem.Mol)

    if CHARGE_COL_NAME not in df.columns:
        df = _calculate_charge(df, mol_col_name, include_charged)

    if SPIN_COL_NAME not in df.columns:
        df = _calculate_spin(df, mol_col_name)

    for _, r in df.iterrows():
        logging.debug(
            f"Mol ID: {r[id_col_name]}, Charge: {r[CHARGE_COL_NAME]}, Spin: {r[SPIN_COL_NAME]}"
        )

    return {
        r[id_col_name]: {
            MOL_KEY: r[mol_col_name],
            CHARGE_KEY: int(r[CHARGE_COL_NAME]),
            SPIN_KEY: int(r[SPIN_COL_NAME]),
        }
        for _, r in df.iterrows()
    }


def _check_columns(df: pd.DataFrame, mol_col_name: str, id_col_name: str) -> None:
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
        raise ValueError(f"'mol_bytes' not found in dataframe columns {df.columns}")
    df[mol_col_name] = df["mol_bytes"].apply(Chem.Mol)
    logging.info(f"RDKit.Mol column is '{mol_col_name}'")

    if id_col_name not in df.columns:
        raise ValueError(f"Column '{id_col_name}' not found in dataframe, {df.columns}")
    if not df[id_col_name].is_unique:
        raise ValueError(f"ID column ({id_col_name}) contains duplicate values")
    logging.info(f"ID column is '{id_col_name}'")


def _calculate_charge(df: pd.DataFrame, mol_col_name: str, include_charged: bool) -> pd.DataFrame:
    """Calculate charge of molecules.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing molecules.
    mol_col_name: str
        Name of the column containing the RDKit.Mol objects OR binary strings.
    include_charged: bool
        If False, filters out charged molecules.

    Returns
    -------
        DataFrame with charge column.
    """
    if CHARGE_COL_NAME not in df.columns:
        df[CHARGE_COL_NAME] = df[mol_col_name].apply(lambda x: int(Chem.GetFormalCharge(x)))
    logging.info(f"Dataset contains {len(df[df[CHARGE_COL_NAME] != 0])} charged molecules.")

    if not include_charged:
        df = df[df[CHARGE_COL_NAME] == 0]
        if len(df) == 0:
            logging.error("No neutral molecules found after charge filtering.")
        else:
            logging.info(f"Dataset contains {len(df)} neutral molecules after charge filtering.")

    return df


def _calculate_spin(df: pd.DataFrame, mol_col_name: str) -> pd.DataFrame:
    """Calculate spin multiplicity of molecules.

    The spin multiplicity is calculated from the number of free radical electrons using Hund's rule
    of maximum multiplicity defined as 2S + 1 where S is the total electron spin. The total spin is
    1/2 the number of free radical electrons in a molecule using Hund's rule.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing molecules.
    mol_col_name: str
        Name of the column containing the RDKit.Mol objects OR binary strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with spin multiplicity column.
    """

    def hunds_rule(mol: Chem.Mol) -> int:
        """Calculate spin multiplicity using Hund's rule."""
        num_radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
        total_electronic_spin = num_radical_electrons / 2
        spin_multiplicity = 2 * total_electronic_spin + 1
        return int(spin_multiplicity)

    df[SPIN_COL_NAME] = df[mol_col_name].apply(lambda x: hunds_rule(x))

    return df


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
