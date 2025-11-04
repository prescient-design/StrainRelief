import pytest
import torch
from hydra import compose, initialize
from strain_relief import test_dir
from strain_relief.cmdline._strain_relief import main

CALCULATED_COLUMNS = [
    "id",
    "local_min_mol",
    "local_min_e",
    "global_min_mol",
    "global_min_e",
    "ligand_strain",
    "passes_strain_filter",
]


@pytest.mark.integration
@pytest.mark.parameterized("device", ["cpu", "cuda"])
@pytest.mark.parameterized("parquet id_col_name", [("target", "SMILES"), ("ligboundconf", "id")])
def test_compute_strain_cpu(mace_model_path: str, device: str, parquet: str, id_col_name: str):
    """Test strain relief computation on CPU."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping CUDA test.")
    with initialize(version_base="1.1", config_path="../../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/{parquet}.parquet",
                f"calculator.model_paths={mace_model_path}",
                f"io.input.id_col_name={id_col_name}",
                "conformers.numConfs=5",
                "experiment=mace",
                f"device={device}",
            ],
        )
    df = main(cfg)

    assert len(df) == 2
    nans_in_col = [c for c in CALCULATED_COLUMNS if df[c].isna().sum() != 0]
    assert nans_in_col == [], f"Columns with NaN values: {nans_in_col}"
