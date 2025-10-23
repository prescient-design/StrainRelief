import pytest
from hydra import compose, initialize
from rdkit import Chem
from strain_relief import test_dir
from strain_relief.compute_strain import _parse_args, compute_strain
from strain_relief.io import load_parquet


def test_compute_strain_from_mols():
    assert True is False


def test_compute_strain_empty_df(device: str):
    with initialize(version_base="1.1", config_path="../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/all_charged.parquet",
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


def test_parse_args():
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
def test_parse_args_mols(mols, ids):
    df = _parse_args(mols=mols, ids=ids)
    assert len(df) == 2
    assert df.id.to_list() == [0, 1]
