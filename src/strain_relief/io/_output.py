import numpy as np
import pandas as pd
import torch
from loguru import logger as logging
from neural_optimiser.conformers import Conformer
from torch_geometric.data import Batch, Data

from strain_relief.constants import (
    EV_TO_KCAL_PER_MOL,
    ID_COL_NAME,
    MOL_COL_NAME,
)


def extract_minimum_conformer(batch: Batch | Data, molecule_attr: str) -> list[Conformer]:
    """Extract the minimum energy converged conformer for each molecule in the batch.

    Parameters
    ----------
    batch: Batch | Data
        Batch or Data object containing the conformers.
    molecule_attr: str
        Attribute name for the molecule indices in the batch.

    Returns
    -------
    ConformerBatch
        ConformerBatch containing only the minimum energy conformers.
    """
    if not hasattr(batch, molecule_attr):
        raise AttributeError(f"Batch does not have attribute '{molecule_attr}'")

    molecule_idxs = set(getattr(batch, molecule_attr))

    minimum_conformers: list[Conformer] = []

    for mol_idx in molecule_idxs:
        conformers = [
            batch.conformer(i)
            for i in range(batch.n_conformers)
            if getattr(batch, molecule_attr)[i] == mol_idx
        ]
        converged = [conf for conf in conformers if conf.converged]

        if len(converged) == 0:
            logging.warning(f"No conformers converged for molecule index {mol_idx}. Skipping.")
            continue

        minimum_conformers.append(min(converged, key=lambda c: float(c.energies)))

    return minimum_conformers


def process_output(
    input_df: pd.DataFrame,
    docked: Data | Batch,
    local_min: Data | Batch,
    global_min: Data | Batch,
    threshold: float,
    parquet_path: str,
    molecule_attr: str | None = None,
    id_col_name: str | None = None,
    mol_col_name: str | None = None,
) -> pd.DataFrame:
    """Process the output of the strain relief calculation and save to a parquet file.

    Parameters
    ----------
    input_df: pd.DataFrame
        Input DataFrame containing the StrainRelief's original input.
    docked: Data | Batch
        Data or Batch object containing the poses of docked molecules.
    local_min: Data | Batch
        Data or Batch object containing the poses of locally minimised molecules.
    global_min: Data | Batch
        Data or Batch object containing the poses of globally minimised molecules.
    threshold: float
        Threshold for the ligand strain filter.
    parquet_path: str
        Path to the output parquet file.
    molecule_attr: str [Optional]
        Attribute name for the molecule indices in the batch.
    id_col_name: str [Optional]
        Name of the column containing the molecule IDs.
    mol_col_name: str [Optional]
        Name of the column containing the RDKit.Mol objects.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the docked and minimum poses of molecules and energies.
    """
    if id_col_name is None:
        id_col_name = ID_COL_NAME
    if mol_col_name is None:
        mol_col_name = MOL_COL_NAME
    if molecule_attr is None:
        molecule_attr = ID_COL_NAME

    if not all(hasattr(b, molecule_attr) for b in (docked, local_min, global_min)):
        raise AttributeError(f"Batch does not have attribute '{molecule_attr}'")

    def energy(c: Conformer | None) -> float:
        return float(c.energies) if c is not None else float("nan")

    def mol_bytes(c: Conformer | None):
        return c.to_rdkit().ToBinary() if c is not None else np.nan

    molecule_idxs = set(getattr(docked, molecule_attr))

    # Compute minimum conformers once and build fast lookup maps
    local_min_map = {
        getattr(c, molecule_attr): c for c in extract_minimum_conformer(local_min, molecule_attr)
    }
    global_min_map = {
        getattr(c, molecule_attr): c for c in extract_minimum_conformer(global_min, molecule_attr)
    }

    rows: list[dict] = []
    for mol_id in molecule_idxs:
        # For if ids were auto-generated
        if isinstance(mol_id, torch.Tensor):
            mol_id = int(mol_id.item())

        lconf = local_min_map.get(mol_id)
        gconf = global_min_map.get(mol_id)

        rows.append(
            {
                "id": mol_id,
                "local_min_mol": mol_bytes(lconf),
                "local_min_e": energy(lconf),
                "global_min_mol": mol_bytes(gconf),
                "global_min_e": energy(gconf),
            }
        )

    # Define columns upfront to ensure correct order and handle empty DataFrame creation
    result_columns = [
        "id",
        "local_min_mol",
        "local_min_e",
        "global_min_mol",
        "global_min_e",
    ]
    results = pd.DataFrame(rows, columns=result_columns)

    results["local_min_e"] *= EV_TO_KCAL_PER_MOL
    results["global_min_e"] *= EV_TO_KCAL_PER_MOL

    results["ligand_strain"] = results["local_min_e"] - results["global_min_e"]
    results["passes_strain_filter"] = results["ligand_strain"] <= threshold

    _log_results(results, global_min)

    # Merge and drop original molecule column
    final_results = input_df.merge(results, left_on=id_col_name, right_on="id", how="outer")
    final_results.drop(columns=[mol_col_name], inplace=True)

    if parquet_path is not None:
        final_results.to_parquet(parquet_path)
        torch.save(local_min, parquet_path.replace(".parquet", "_local_min.pt"))
        torch.save(global_min, parquet_path.replace(".parquet", "_global_min.pt"))
        logging.info(f"Batches and outputs saved to {parquet_path}")
    else:
        logging.info("Output file not provided, data not saved.")

    return final_results


def _log_results(results: pd.DataFrame, global_min: Data | Batch) -> None:
    """Helper method to log output summary statistics."""
    # Log individual calculated ligand strains
    for id, strain in zip(results.id, results.ligand_strain):
        if strain is np.nan:
            logging.warning(f"Ligand strain could not be calculated for molecule {id}.")
        elif strain > 0:
            logging.debug(f"{strain:.2f} kcal/mol ligand strain for molecule {id}")
        else:
            logging.warning(
                f"{strain:.2f} kcal/mol ligand strain for molecule {id}. Negative ligand strain."
            )

    # Log any negative or NaN ligand strains
    if not results[results.ligand_strain < 0].empty:
        logging.warning(
            f"{len(results[results.ligand_strain < 0])} molecules have a negative ligand strain, "
            "meaning the initial conformer is lower energy than all generated conformers."
        )
    if not results[results.ligand_strain.isna()].empty:
        logging.warning(
            f"{len(results[results.ligand_strain.isna()])} molecules have no conformers generated "
            "for either the initial or minimised pose, so strain cannot be calculated."
        )

    # Log overall convergence statistics
    total_n_confs: int = global_min.converged.sum()
    if total_n_confs > 0 and not results.empty:
        logging.info(
            f"{total_n_confs:,} configurations converged across {len(results):,} molecules "
            f"(avg. {total_n_confs / len(results):.2f} per molecule)"
        )
    else:
        logging.error(
            "Ligand strain calculation failed for all molecules or no molecules were processed."
        )
