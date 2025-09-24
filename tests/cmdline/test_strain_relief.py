import pytest
from hydra import compose, initialize
from strain_relief import test_dir
from strain_relief.cmdline._strain_relief import main


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
