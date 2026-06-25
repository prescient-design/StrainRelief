from typing import Annotated, Literal

import dagster as dg
from neural_optimiser.calculators import FAIRChemCalculator, MACECalculator, MMFF94Calculator
from neural_optimiser.calculators.base import Calculator
from neural_optimiser.optimisers import BFGS
from neural_optimiser.optimisers.base import Optimiser
from pydantic import Field

# ---------- Assets Configs ----------


class InputConfig(dg.Config):
    """Config for preprocess_data -> to_mols_dict()."""

    parquet_path: str
    include_charged: bool = True
    id_col_name: str | None = None
    mol_col_name: str | None = None


class ConformerConfig(dg.Config):
    """Config for conformers -> generate_conformers()."""

    randomSeed: int = -1
    numConfs: int = 10
    maxAttempts: int = 200
    pruneRmsThresh: float = 0.1
    clearConfs: bool = False
    numThreads: int = 0


class LocalOptimisationConfig(dg.Config):
    """Config for local_minima -> run_optimisation()."""

    batch_size: int
    num_workers: int
    device: str


class GlobalOptimisationConfig(dg.Config):
    """Config for global_minima -> run_optimisation()."""

    batch_size: int
    num_workers: int
    device: str


class OutputConfig(dg.Config):
    """Config for aggregate_results -> process_output()."""

    threshold: float
    parquet_path: str | None = None  # filename only; the run-scoped dir is set in the asset
    save_batch: bool = False
    molecule_attr: str | None = None
    id_col_name: str | None = None
    mol_col_name: str | None = None


class PlotConfig(dg.Config):
    """Config for plot_results."""

    # Column in the input data holding the ground-truth strain (kcal/mol). When set, the
    # predicted (ligand_strain) vs true scatterplot and accuracy metrics are produced.
    true_strain_col: str | None = None


# ---------- Resources Configs ----------


class MACEConfig(dg.Config):
    kind: Literal["mace"] = "mace"
    model_paths: str
    device: str = "cpu"
    default_dtype: Literal["float32", "float64"] = "float32"

    def build(self) -> Calculator:
        return MACECalculator(
            model_paths=self.model_paths, device=self.device, default_dtype=self.default_dtype
        )


class MMFF94Config(dg.Config):
    kind: Literal["mmff94"] = "mmff94"

    def build(self) -> Calculator:
        return MMFF94Calculator()


class FAIRChemConfig(dg.Config):
    kind: Literal["fairchem"] = "fairchem"
    model_paths: str
    device: str = "cpu"
    task_name: str = "omol"
    default_dtype: Literal["float32", "float64"] = "float32"

    def build(self) -> Calculator:
        return FAIRChemCalculator(
            model_paths=self.model_paths,
            device=self.device,
            task_name=self.task_name,
            default_dtype=self.default_dtype,
        )


CalculatorConfig: dg.Config = Annotated[
    MACEConfig | MMFF94Config | FAIRChemConfig, Field(discriminator="kind")
]


class BFGSConfig(dg.Config):
    kind: Literal["bfgs"] = "bfgs"
    max_step: float = 0.04
    steps: int = 250
    fmax: float | None = None
    fexit: float | None = None

    def build(self) -> Optimiser:
        return BFGS(max_step=self.max_step, steps=self.steps, fmax=self.fmax, fexit=self.fexit)


class _BaseOptimiserConfig(dg.Config):
    """Placeholder so OptimiserConfig is a discriminated union (i.e. renders as a selector,
    uniform with CalculatorConfig). The base Optimiser is abstract, so this is never selected;
    it exists only until a second concrete optimiser (e.g. LBFGS) is added."""

    kind: Literal["base"] = "base"

    def build(self) -> Optimiser:
        raise NotImplementedError(
            "Select a concrete optimiser (e.g. bfgs); 'base' is a placeholder."
        )


OptimiserConfig: dg.Config = Annotated[
    BFGSConfig | _BaseOptimiserConfig, Field(discriminator="kind")
]
