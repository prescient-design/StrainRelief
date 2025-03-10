from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem
from strain_relief import test_dir
from strain_relief.io._io import (
    _calculate_charge,
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
    df = load_parquet(parquet_path=parquet_path, id_col_name=id_col_name)
    assert len(df) > 0


def test_calculate_charge():
    df = pd.DataFrame({"mol": [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("C[O-]")]})
    df = _calculate_charge(df, "mol")
    assert df.charge.to_list() == [0, -1]


def test_to_mols_dict():
    df = pd.DataFrame({"mol": [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("C[O-]")], "id": [1, 2]})
    df = _calculate_charge(df, "mol")
    mols = to_mols_dict(df, "mol", "id")
    assert len(mols) == 1


@pytest.mark.skip(reason="Test not implemented")
def test_check_columns():
    pass


@pytest.mark.skip(reason="Test not implemented")
def test_save_data():
    pass
