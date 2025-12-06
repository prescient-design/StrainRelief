from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem
from strain_relief import test_dir
from strain_relief.constants import CHARGE_COL_NAME, SPIN_COL_NAME
from strain_relief.io._input import (
    _calculate_charge,
    _calculate_spin,
    _check_columns,
    load_parquet,
    to_mols_dict,
)


@pytest.mark.parametrize(
    "parquet_path, id_col_name",
    [
        (test_dir / "data" / "ligboundconf.parquet", None),
        (test_dir / "data" / "target.parquet", "SMILES"),
    ],
)
def test_load_parquet(parquet_path: Path, id_col_name: str | None):
    """Test loading parquet files with and without specifying id column name."""
    df = load_parquet(parquet_path=parquet_path, id_col_name=id_col_name, include_charged=True)
    assert len(df) > 0


def test_include_charged_false():
    """Test loading parquet with include_charged=False results in empty DataFrame."""
    df = load_parquet(parquet_path=test_dir / "data" / "all_charged.parquet", include_charged=False)
    assert df.empty


def test_calculate_charge():
    """Test charge calculation on molecules."""
    df = pd.DataFrame({"mol": [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("C[O-]")]})
    df = _calculate_charge(df, "mol", True)
    assert df[CHARGE_COL_NAME].to_list() == [0, -1]


def test_calculate_spin():
    """Test spin calculation on molecules."""
    df = pd.DataFrame({"mol": [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("C[CH]")]})
    df = _calculate_spin(df, "mol")
    assert df[SPIN_COL_NAME].to_list() == [1, 3]


def test_to_mols_dict():
    """Test conversion to molecules dictionary."""
    df = pd.DataFrame({"mol": [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("C[O-]")], "id": [1, 2]})
    df = _calculate_charge(df, "mol", False)
    mols = to_mols_dict(df=df, mol_col_name="mol", id_col_name="id", include_charged=False)
    assert len(mols) == 1


def test_check_columns_success(sample_mol_bytes):
    """Test _check_columns succeeds with valid inputs."""
    bytes_c, bytes_o = sample_mol_bytes
    df = pd.DataFrame({"mol_bytes": [bytes_c, bytes_o], "my_id": [1, 2]})
    mol_col = "mol"
    id_col = "my_id"

    _check_columns(df, mol_col, id_col)

    assert mol_col in df.columns
    assert isinstance(df[mol_col].iloc[0], Chem.Mol)
    assert df[mol_col].iloc[0].GetAtomWithIdx(0).GetSymbol() == "C"
    assert df[mol_col].iloc[1].GetAtomWithIdx(0).GetSymbol() == "O"


def test_check_columns_no_mol_bytes():
    """Test _check_columns raises error if 'mol_bytes' is missing."""
    df = pd.DataFrame({"my_id": [1, 2]})
    with pytest.raises(ValueError, match="'mol_bytes' not found"):
        _check_columns(df, "mol", "my_id")


def test_check_columns_no_id_col(sample_mol_bytes):
    """Test _check_columns raises error if id_col_name is missing."""
    bytes_c, _ = sample_mol_bytes
    df = pd.DataFrame({"mol_bytes": [bytes_c]})

    id_col = "missing_id"
    with pytest.raises(ValueError, match=f"Column '{id_col}' not found"):
        _check_columns(df, "mol", id_col)


def test_check_columns_duplicate_ids(sample_mol_bytes):
    """Test _check_columns raises error for duplicate IDs."""
    bytes_c, bytes_o = sample_mol_bytes
    id_col = "my_id"
    df = pd.DataFrame(
        {
            "mol_bytes": [bytes_c, bytes_o],
            id_col: [1, 1],  # Duplicate IDs
        }
    )

    match_str = f"ID column \\({id_col}\\) contains duplicate values"
    with pytest.raises(ValueError, match=match_str):
        _check_columns(df, "mol", id_col)
