from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from strain_relief import test_dir
from strain_relief.deployment.strain_relief import DeploymentConfig, deployment_function
from strain_relief.io import load_parquet


@pytest.fixture
def df_w_processing():
    df = load_parquet(
        parquet_path=str(test_dir / "data" / "target.parquet"),
        id_col_name="SMILES",
    )
    df["id"] = df["SMILES"].astype(str)
    return df


@pytest.fixture
def df_wo_processing():
    df = pd.read_parquet(test_dir / "data" / "target.parquet")
    df["id"] = df["SMILES"].astype(str)
    return df


@pytest.mark.gpu
@pytest.mark.parametrize("fixture", ["df_w_processing", "df_wo_processing"])
def test_strain_relief(request, fixture: pd.DataFrame):
    dataframe = request.getfixturevalue(fixture)
    output_columns = dataframe.columns.tolist() + ["ligand_strain", "passes_strain_filter"]
    if "mol" in output_columns:
        output_columns.remove("mol")

    config = DeploymentConfig(
        minimisation="mace",
        num_confs=1,
    )
    config = OmegaConf.create(asdict(config))
    out = deployment_function(dataframe=dataframe, config=config)

    assert isinstance(out, pd.DataFrame)
    assert all(
        col in out.columns for col in output_columns
    ), f"Expected columns: {output_columns} to be present but got {out.columns}."
    # Test that id values have not changed but allow for missing values
    assert all(id_value in set(dataframe["id"]) for id_value in out["id"].values)
    assert out["id"].duplicated().sum() == 0, "Output contains duplicate IDs"
    assert all(
        isinstance(val, float | np.floating) or np.isnan(val) for val in out["ligand_strain"]
    ), "All values in 'ligand_strain' should be float or NaN"
    assert all(
        isinstance(val, bool) or np.isnan(val) for val in out["passes_strain_filter"]
    ), "All values in 'passes_strain_filter' should be bool"
