import pandas as pd
from loguru import logger as logging
from rdkit import Chem

from strain_relief.constants import (
    CHARGE_COL_NAME,
    CHARGE_KEY,
    ID_COL_NAME,
    MOL_COL_NAME,
    MOL_KEY,
    SPIN_COL_NAME,
    SPIN_KEY,
)
from strain_relief.data_types import MolsDict


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
            ID_COL_NAME: r[id_col_name],
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
        charge = Chem.GetFormalCharge(mol)
        num_electrons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge
        num_radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())

        is_odd_electron = num_electrons % 2 != 0

        unpaired_electrons = num_radical_electrons
        if is_odd_electron and unpaired_electrons == 0:
            unpaired_electrons = 1

        total_spin_s = unpaired_electrons / 2.0
        multiplicity = int((2 * total_spin_s) + 1)
        return int(multiplicity)

    df[SPIN_COL_NAME] = df[mol_col_name].apply(lambda x: hunds_rule(x))

    return df
