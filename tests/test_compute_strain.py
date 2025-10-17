import pytest
from hydra import compose, initialize
from rdkit import Chem
from strain_relief import test_dir
from strain_relief.compute_strain import _parse_args, compute_strain
from strain_relief.io import load_parquet


@pytest.mark.integration
@pytest.mark.parametrize("eval_method", ["mmff94", "mmff94s"])
@pytest.mark.parametrize("min_method", ["mmff94", "mmff94s"])
def test_strain_relief(min_method: str, eval_method: str):
    with initialize(version_base="1.1", config_path="../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/target.parquet",
                "io.input.id_col_name=SMILES",
                f"calculator={min_method}",
                "optimiser@local_optimiser=bfgs",
                "optimiser@global_optimiser=bfgs",
                f"energy_eval={eval_method}",
                "conformers.numConfs=1",
            ],
        )
    df = load_parquet(
        parquet_path=cfg.io.input.parquet_path, id_col_name="SMILES", include_charged=True
    )
    compute_strain(df=df, cfg=cfg)


@pytest.mark.integration
@pytest.mark.gpu
def test_strain_relief_w_mace():
    with initialize(version_base="1.1", config_path="../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/target.parquet",
                "io.input.id_col_name=SMILES",
                "optimiser@local_optimiser=bfgs",
                "optimiser@global_optimiser=bfgs",
                "calculator=mace",
                "local_min.fmax=0.50",
                f"local_optimiser.model_paths={test_dir}/models/MACE.model",
                f"global_optimiser.model_paths={test_dir}/models/MACE.model",
                f"model.model_paths={test_dir}/models/MACE.model",
                "conformers.numConfs=1",
            ],
        )
    df = load_parquet(
        parquet_path=cfg.io.input.parquet_path, id_col_name="SMILES", include_charged=True
    )
    compute_strain(df=df, cfg=cfg)


@pytest.mark.integration
@pytest.mark.gpu
def test_strain_relief_w_esen(esen_model_path: str):
    with initialize(version_base="1.1", config_path="../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/target.parquet",
                "io.input.id_col_name=SMILES",
                "optimiser@local_optimiser=bfgs",
                "optimiser@global_optimiser=bfgs",
                "calculator=fairchem",
                "local_optimiser.fmax=0.50",
                f"local_optimiser.model_paths={test_dir}/models/eSEN.pt",
                f"global_optimiser.model_paths={test_dir}/models/eSEN.pt",
                f"model.model_paths={test_dir}/models/eSEN.pt",
                "conformers.numConfs=1",
            ],
        )
    df = load_parquet(
        parquet_path=cfg.io.input.parquet_path, id_col_name="SMILES", include_charged=True
    )
    compute_strain(df=df, cfg=cfg)


@pytest.mark.integration
def test_strain_relief_all_charged():
    with initialize(version_base="1.1", config_path="../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/all_charged.parquet",
                "io.input.id_col_name=id",
                "optimiser@local_optimiser=bfgs",
                "optimiser@global_optimiser=bfgs",
                "calculator=mace",
                "conformers.numConfs=1",
            ],
        )
    df = load_parquet(
        parquet_path=cfg.io.input.parquet_path, id_col_name="id", include_charged=False
    )
    results = compute_strain(df=df, cfg=cfg)
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
