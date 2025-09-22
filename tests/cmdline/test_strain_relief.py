import pytest
from hydra import compose, initialize
from strain_relief import test_dir
from strain_relief.cmdline._strain_relief import main, strain_relief
from strain_relief.io import load_parquet


@pytest.mark.integration
@pytest.mark.parametrize("eval_method", ["mmff94", "mmff94s"])
@pytest.mark.parametrize("min_method", ["mmff94", "mmff94s"])
def test_strain_relief(min_method: str, eval_method: str):
    with initialize(version_base="1.1", config_path="../../src/strain_relief/hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/target.parquet",
                "io.input.id_col_name=SMILES",
                f"minimisation@local_min={min_method}",
                f"minimisation@global_min={min_method}",
                f"energy_eval={eval_method}",
                "conformers.numConfs=1",
            ],
        )
    df = load_parquet(
        parquet_path=cfg.io.input.parquet_path, id_col_name="SMILES", include_charged=True
    )
    strain_relief(df, cfg)


@pytest.mark.integration
@pytest.mark.gpu
def test_strain_relief_w_mace():
    with initialize(version_base="1.1", config_path="../../src/strain_relief/hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/target.parquet",
                "io.input.id_col_name=SMILES",
                "minimisation@local_min=mace",
                "minimisation@global_min=mace",
                "local_min.fmax=0.50",
                "model=mace",
                f"local_min.model_paths={test_dir}/models/MACE.model",
                f"global_min.model_paths={test_dir}/models/MACE.model",
                f"model.model_paths={test_dir}/models/MACE.model",
                "conformers.numConfs=1",
            ],
        )
    df = load_parquet(
        parquet_path=cfg.io.input.parquet_path, id_col_name="SMILES", include_charged=True
    )
    strain_relief(df, cfg)


@pytest.mark.integration
@pytest.mark.gpu
def test_strain_relief_w_esen(esen_model_path: str):
    with initialize(version_base="1.1", config_path="../../src/strain_relief/hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/target.parquet",
                "io.input.id_col_name=SMILES",
                "minimisation@local_min=fairchem",
                "minimisation@global_min=fairchem",
                "local_min.fmax=0.50",
                "model=fairchem",
                f"local_min.model_paths={test_dir}/models/eSEN.pt",
                f"global_min.model_paths={test_dir}/models/eSEN.pt",
                f"model.model_paths={test_dir}/models/eSEN.pt",
                "conformers.numConfs=1",
            ],
        )
    df = load_parquet(
        parquet_path=cfg.io.input.parquet_path, id_col_name="SMILES", include_charged=True
    )
    strain_relief(df, cfg)


@pytest.mark.integration
def test_strain_relief_filter_charged():
    with initialize(version_base="1.1", config_path="../../src/strain_relief/hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/all_charged.parquet",
                "io.input.id_col_name=id",
                "io.input.include_charged=False",
                "minimisation@local_min=mmff94s",
                "minimisation@global_min=mmff94s",
                "conformers.numConfs=1",
            ],
        )
    df = load_parquet(
        parquet_path=cfg.io.input.parquet_path,
        id_col_name="id",
        include_charged=cfg.io.input.include_charged,
    )
    results = strain_relief(df, cfg)

    print(results["ligand_strain"])
    print(results["passes_strain_filter"])

    assert results["ligand_strain"].isna().all()
    assert results["passes_strain_filter"].isna().all()


@pytest.mark.integration
def test_strain_relief_include_charged():
    with initialize(version_base="1.1", config_path="../../src/strain_relief/hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/all_charged.parquet",
                "io.input.id_col_name=id",
                "io.input.include_charged=True",
                "minimisation@local_min=mmff94s",
                "minimisation@global_min=mmff94s",
                "conformers.numConfs=1",
            ],
        )
    df = load_parquet(
        parquet_path=cfg.io.input.parquet_path, id_col_name="id", include_charged=True
    )
    strain_relief(df, cfg)


@pytest.mark.integration
@pytest.mark.parametrize(
    "parquet, id_col_name",
    [
        (f"{test_dir}/data/target.parquet", "SMILES"),
        (f"{test_dir}/data/ligboundconf.parquet", "id"),
    ],
)
def test_main(parquet: str, id_col_name: str):
    with initialize(version_base="1.1", config_path="../../src/strain_relief/hydra_config"):
        overrides = [
            f"io.input.parquet_path={parquet}",
            f"io.input.id_col_name={id_col_name}",
            "minimisation@local_min=mmff94s",
            "minimisation@global_min=mmff94s",
            "conformers.numConfs=1",
        ]
        cfg = compose(config_name="default", overrides=overrides)
    main(cfg)
