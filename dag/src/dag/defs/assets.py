import math
import os

import dagster as dg
import pandas as pd
from neural_optimiser.conformers import Conformer, ConformerBatch
from rdkit import Chem
from strain_relief.conformers import generate_conformers
from strain_relief.constants import MOL_COL_NAME
from strain_relief.data_types import MolsDict
from strain_relief.io import load_parquet, process_output, to_mols_dict
from strain_relief.optimisation import run_optimisation

from ._metadata import (
    histogram_md,
    metrics_table_md,
    per_ligand_conformer_counts,
    scatter_md,
    table_schema_md,
)
from .configs import (
    ConformerConfig,
    GlobalOptimisationConfig,
    InputConfig,
    LocalOptimisationConfig,
    OutputConfig,
    PlotConfig,
)
from .partitions import CHUNK_SIZE, chunk_partitions
from .resources import DATA_DIR, GlobalOptimiserResource, LocalOptimiserResource


def _chunk(docked_mols: dict, partition_key: str) -> dict:
    """Row-range slice of the (master-ordered) docked_mols for a chunk partition."""
    start = int(partition_key) * CHUNK_SIZE
    return dict(list(docked_mols.items())[start : start + CHUNK_SIZE])


@dg.asset(io_manager_key="pandas_io_manager")
def input_df(config: InputConfig) -> dg.MaterializeResult:
    """Load the input parquet of docked molecules.

    Drop the live RDKit ``mol`` column before returning: it cannot be serialised to parquet.
    Molecules persist as ``mol_bytes`` and are reconstructed downstream.
    """
    df = load_parquet(parquet_path=config.parquet_path)
    df = df.drop(columns=[config.mol_col_name or MOL_COL_NAME], errors="ignore")
    return dg.MaterializeResult(
        value=df,
        metadata={
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "schema": table_schema_md(df),
        },
    )


@dg.asset
def docked_mols(
    context: dg.AssetExecutionContext, input_df: pd.DataFrame, config: InputConfig
) -> dict:
    """Convert the input DataFrame to a MolsDict and register the chunk partitions."""
    mols: MolsDict = to_mols_dict(input_df, **config.model_dump())
    n_chunks = math.ceil(len(mols) / CHUNK_SIZE)
    context.instance.add_dynamic_partitions(
        chunk_partitions.name, [str(i) for i in range(n_chunks)]
    )
    return mols


@dg.asset(partitions_def=chunk_partitions, io_manager_key="pytorch_io_manager")
def conformers(
    context: dg.AssetExecutionContext, docked_mols: dict, config: ConformerConfig
) -> dg.MaterializeResult:
    """Generate conformers for this chunk's molecules."""
    docked_mols = _chunk(docked_mols, context.partition_key)
    generated_mols = generate_conformers(docked_mols, **config.model_dump())
    generated_batch: ConformerBatch = ConformerBatch.cat(
        [ConformerBatch.from_rdkit(**generated_mols[id]) for id in generated_mols]
    )
    total, _ = per_ligand_conformer_counts(generated_batch)
    counts = list(total.values())
    return dg.MaterializeResult(
        value=generated_batch,
        metadata={
            "num_ligands": len(total),
            "total_conformers": int(sum(counts)),
            "avg_conformers_per_ligand": float(sum(counts) / len(total)) if total else 0.0,
            "conformers_per_ligand_hist": histogram_md(
                counts,
                discrete=True,
                xlabel="Conformers per Ligand",
                title="Conformers Generated per Ligand",
            ),
        },
    )


@dg.asset(partitions_def=chunk_partitions, io_manager_key="pytorch_io_manager")
def local_minima(
    context: dg.AssetExecutionContext,
    local_optimiser: LocalOptimiserResource,
    docked_mols: dict,
    config: LocalOptimisationConfig,
) -> dg.MaterializeResult:
    """Locally optimised (minimised) conformers from this chunk's docked poses."""
    docked_mols = _chunk(docked_mols, context.partition_key)
    docked_batch = ConformerBatch.from_data_list(
        [Conformer.from_rdkit(**docked_mols[id]) for id in docked_mols]
    )
    minima = run_optimisation(
        docked_batch,
        local_optimiser.get_optimiser(),
        config.batch_size,
        config.num_workers,
        config.device,
    )
    total, conv = per_ligand_conformer_counts(minima)
    n_no_converged = sum(1 for lid in total if conv.get(lid, 0) == 0)
    return dg.MaterializeResult(
        value=minima,
        metadata={
            "num_ligands": len(total),
            "ligands_with_no_converged_conformers": n_no_converged,
        },
    )


@dg.asset(partitions_def=chunk_partitions, io_manager_key="pytorch_io_manager")
def global_minima(
    global_optimiser: GlobalOptimiserResource,
    conformers: ConformerBatch,
    config: GlobalOptimisationConfig,
) -> dg.MaterializeResult:
    """Globally optimised (minimised) conformers from this chunk's generated conformers."""
    minima = run_optimisation(
        conformers,
        global_optimiser.get_optimiser(),
        config.batch_size,
        config.num_workers,
        config.device,
    )
    total, conv = per_ligand_conformer_counts(minima)
    converged_counts = [conv.get(lid, 0) for lid in total]
    n_no_converged = sum(1 for c in converged_counts if c == 0)
    return dg.MaterializeResult(
        value=minima,
        metadata={
            "num_ligands": len(total),
            "ligands_with_no_converged_conformers": n_no_converged,
            "converged_conformers_per_ligand_hist": histogram_md(
                converged_counts,
                discrete=True,
                xlabel="Converged Conformers per Ligand",
                title="Converged Conformers per Ligand (Global Min.)",
            ),
        },
    )


@dg.asset(
    io_manager_key="pandas_io_manager",
    ins={
        "local_minima": dg.AssetIn(partition_mapping=dg.AllPartitionMapping()),
        "global_minima": dg.AssetIn(partition_mapping=dg.AllPartitionMapping()),
    },
)
def aggregate_results(
    input_df: pd.DataFrame,
    docked_mols: dict,
    local_minima: object,  # ConformerBatch (1 chunk) or list[ConformerBatch] (fan-in over chunks)
    global_minima: object,
    config: OutputConfig,
) -> dg.MaterializeResult:
    """Aggregate results from local and global minima and save to output parquet file."""
    # Fan-in: the pytorch io manager returns a list of per-chunk batches; cat back to the full set.
    if isinstance(local_minima, list):
        local_minima = ConformerBatch.cat(local_minima)
    if isinstance(global_minima, list):
        global_minima = ConformerBatch.cat(global_minima)
    docked_batch = ConformerBatch.from_data_list(
        [Conformer.from_rdkit(**docked_mols[id]) for id in docked_mols]
    )
    if MOL_COL_NAME not in input_df.columns:
        input_df[MOL_COL_NAME] = input_df["mol_bytes"].apply(Chem.Mol)

    cfg = config.model_dump()
    os.makedirs(DATA_DIR, exist_ok=True)
    cfg["parquet_path"] = f"{DATA_DIR}/{os.path.basename(cfg['parquet_path'] or 'output.parquet')}"

    output_df = process_output(input_df, docked_batch, local_minima, global_minima, **cfg)

    strain = output_df["ligand_strain"]
    n_total = len(output_df)
    n_pass = int(output_df["passes_strain_filter"].sum())
    return dg.MaterializeResult(
        value=output_df,
        metadata={
            "num_molecules": n_total,
            "molecules_passing_filter": n_pass,
            "threshold_kcal_per_mol": config.threshold,
            "pass_rate": float(n_pass / n_total) if n_total else 0.0,
            "negative_strains": int((strain < 0).sum()),
            "nan_strains": int(strain.isna().sum()),
            "total_converged_conformers": int(global_minima.converged.sum()),
            "schema": table_schema_md(output_df),
            "strain_hist": histogram_md(
                strain.dropna(),
                xlabel="Ligand Strain (kcal/mol)",
                title="Ligand Strain Distribution",
            ),
        },
    )


@dg.asset
def plot_results(aggregate_results: pd.DataFrame, config: PlotConfig) -> dg.MaterializeResult:
    """Compare predicted ligand strain against ground-truth strain (if available)."""
    df = aggregate_results
    true_col = config.true_strain_col

    if not true_col or true_col not in df.columns:
        return dg.MaterializeResult(
            metadata={
                "note": dg.MetadataValue.md(
                    f"No ground-truth strain column (true_strain_col={true_col!r}). Set "
                    "`plot_results.config.true_strain_col` to a column in the input data to "
                    "enable the predicted-vs-true scatterplot and accuracy metrics."
                )
            }
        )

    valid = df[[true_col, "ligand_strain"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) < 2:
        return dg.MaterializeResult(
            metadata={
                "note": dg.MetadataValue.md(
                    f"Only {len(valid)} valid (true, predicted) pair(s); need >= 2 for metrics."
                ),
                "n_compared": len(valid),
            }
        )

    true, pred = valid[true_col], valid["ligand_strain"]
    err = pred - true
    ss_tot = float(((true - true.mean()) ** 2).sum())
    metrics = {
        "MAE": float(err.abs().mean()),
        "RMSE": float((err**2).mean() ** 0.5),
        "R2": (1 - float((err**2).sum()) / ss_tot) if ss_tot else float("nan"),
        "Spearman rho": float(true.corr(pred, method="spearman")),
    }
    return dg.MaterializeResult(
        metadata={
            "n_compared": len(valid),
            "metrics": metrics_table_md(metrics),
            "scatter": scatter_md(
                true.tolist(),
                pred.tolist(),
                xlabel="true strain (kcal/mol)",
                ylabel="predicted strain (kcal/mol)",
                title="Predicted vs true ligand strain",
            ),
        }
    )
