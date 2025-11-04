import pandas as pd
import pytest
import torch
from neural_optimiser.conformers import ConformerBatch
from strain_relief.io._output import extract_minimum_conformer, process_output


def test_extract_minimum_conformer(minimised_batch: ConformerBatch):
    """Test extracting minimum energy conformers from a batch."""
    map = {getattr(c, "name"): c for c in extract_minimum_conformer(minimised_batch, "name")}
    expected_map = {"H2O": 0.2, "NH3": 0.3}

    for name, conformer in map.items():
        expected_energy = expected_map[name]
        assert torch.isclose(conformer.energies, torch.tensor(expected_energy))


def test_extract_minimum_conformer_missing_attr(minimised_batch: ConformerBatch):
    """Test extracting minimum energy conformers with missing attribute."""
    with pytest.raises(AttributeError):
        extract_minimum_conformer(minimised_batch, molecule_attr="missing")


def test_process_output_missing_attr(minimised_batch: ConformerBatch):
    missing_attr = minimised_batch.clone()
    del missing_attr.name

    with pytest.raises(AttributeError):
        process_output(
            docked=missing_attr,
            local_min=minimised_batch,
            global_min=minimised_batch,
            molecule_attr="name",
            parquet_path=None,
            input_df=None,
            threshold=None,
        )


def test_process_output_success(minimised_batch: ConformerBatch, tmp_path):
    df = process_output(
        input_df=pd.DataFrame({"id": ["H2O", "NH3"], "mol": [None, None]}),
        docked=minimised_batch,
        local_min=minimised_batch,
        global_min=minimised_batch,
        threshold=1.0,
        molecule_attr="name",
        id_col_name="id",
        mol_col_name="mol",
        parquet_path=str(tmp_path / "output.parquet"),
    )
    assert len(df) == 2
