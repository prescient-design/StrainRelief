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
def test_compute_strain_cpu():
    """Test strain relief computation on CPU."""
    with initialize(version_base="1.1", config_path="../../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/target.parquet",
                f"calculator.model_paths={test_dir}/models/MACE.model",
                "io.input.id_col_name=SMILES",
                "conformers.numConfs=5",
                "experiment=mace",
                "device=cpu",
            ],
        )
    df = main(cfg)

    assert len(df) == 2
    nans_in_col = [c for c in CALCULATED_COLUMNS if df[c].isna().sum() != 0]
    assert nans_in_col == [], f"Columns with NaN values: {nans_in_col}"


@pytest.mark.integration
def test_compute_strain_cuda():
    """Test strain relief computation on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping CUDA test.")
    with initialize(version_base="1.1", config_path="../../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/target.parquet",
                f"calculator.model_paths={test_dir}/models/MACE.model",
                "io.input.id_col_name=SMILES",
                "conformer.numCOnfs=5",
                "experiment=mace",
                "device=cuda",
            ],
        )
    df = main(cfg)

    assert len(df) == 2
    nans_in_col = [c for c in CALCULATED_COLUMNS if df[c].isna().sum() != 0]
    assert nans_in_col == [], f"Columns with NaN values: {nans_in_col}"
