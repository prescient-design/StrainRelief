import pytest
from hydra import compose, initialize
from strain_relief import compute_strain, test_dir
from strain_relief.cmdline._strain_relief import main
from strain_relief.io import load_parquet


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
    compute_strain(df=df, cfg=cfg)


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
