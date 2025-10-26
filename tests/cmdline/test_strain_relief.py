import pytest
import torch
from hydra import compose, initialize
from strain_relief import test_dir
from strain_relief.cmdline._strain_relief import main

pytest.mark.integration


def test_compute_strain_cpu():
    """Test strain relief computation on CPU."""
    with initialize(version_base="1.1", config_path="../../hydra_config"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"io.input.parquet_path={test_dir}/data/target.parquet",
                f"calculator.model_paths={test_dir}/models/MACE.model",
                "io.input.id_col_name=SMILES",
                "experiment=pytest",
                "device=cpu",
            ],
        )
    main(cfg)


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
                "experiment=pytest",
                "device=cuda",
            ],
        )
    main(cfg)
