import pytest
from hydra import compose, initialize
from rdkit import Chem
from rdkit.Chem import AllChem
from strain_relief import test_dir
from strain_relief.compute_strain import _parse_args, compute_strain
from strain_relief.io import load_parquet

CALCULATED_COLUMNS = [
    "id",
    "local_min_mol",
    "local_min_e",
    "global_min_mol",
    "global_min_e",
    "ligand_strain",
    "passes_strain_filter",
]


def test_compute_strain_from_mols(device: str):
    """Test strain computation from a list of molecules."""
    with initialize(version_base="1.1", config_path="../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"calculator.model_paths={test_dir}/models/MACE.model",
                "experiment=pytest",
                f"device={device}",
            ],
        )

    mols = []
    for smiles in ["C", "CC"]:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mols.append(mol)
    df = _parse_args(mols=mols)
    results = compute_strain(df=df, cfg=cfg)

    assert len(results) == 2
    nans_in_col = [c for c in CALCULATED_COLUMNS if results[c].isna().sum() != 0]
    assert nans_in_col == [], f"Columns with NaN values: {nans_in_col}"


def test_parse_args_from_df():
    """Test _parse_args with a DataFrame input."""
    df = load_parquet(
        parquet_path=test_dir / "data" / "target.parquet",
        id_col_name="SMILES",
        include_charged=True,
    )
    df2 = _parse_args(df=df)
    assert df.equals(df2)


@pytest.mark.parametrize(
    "mols,ids",
    [
        ([Chem.MolFromSmiles("C"), Chem.MolFromSmiles("CC")], [0, 1]),
        ([Chem.MolFromSmiles("C").ToBinary(), Chem.MolFromSmiles("CC").ToBinary()], None),
    ],
)
def test_parse_args_from_mols(mols, ids):
    """Test _parse_args with a list of molecules and optional IDs."""
    df = _parse_args(mols=mols, ids=ids)
    assert len(df) == 2
    assert df.id.to_list() == [0, 1]
