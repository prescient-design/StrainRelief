import pytest
from hydra import compose, initialize
from rdkit import Chem
from rdkit.Chem import AllChem
from strain_relief import test_dir
from strain_relief.compute_strain import _parse_args, compute_strain
from strain_relief.io import load_parquet


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
    for mol in [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("CC")]:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mols.append(mol)
    df = _parse_args(mols=mols)
    compute_strain(df=df, cfg=cfg)


def test_compute_strain_empty_df(device: str):
    """Test strain computation on a DataFrame with no neutral molecules."""
    with initialize(version_base="1.1", config_path="../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/all_charged.parquet",
                f"calculator.model_paths={test_dir}/models/MACE.model",
                "experiment=pytest",
                f"device={device}",
            ],
        )
    df = load_parquet(
        parquet_path=cfg.io.input.parquet_path, id_col_name="id", include_charged=False
    )
    results = compute_strain(df=df, cfg=cfg)
    assert len(results) == 2
    assert results["ligand_strain"].isna().all()
    assert results["passes_strain_filter"].isna().all()


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
