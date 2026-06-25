import os
from pathlib import Path
from typing import ClassVar

import dagster as dg
import pandas as pd
import torch
from neural_optimiser.calculators.base import Calculator
from neural_optimiser.optimisers.base import Optimiser

from .configs import (
    BFGSConfig,
    CalculatorConfig,
    MACEConfig,
    OptimiserConfig,
)

DATA_DIR = str(Path(__file__).resolve().parents[4] / "data")
_CALCULATOR_CACHE: dict[str, Calculator] = {}


def _get_or_build_calculator(config: CalculatorConfig) -> Calculator:
    """Load-once calculator cache; keyed on resolved config."""
    key = config.model_dump_json()
    if key not in _CALCULATOR_CACHE:
        _CALCULATOR_CACHE[key] = config.build()
    return _CALCULATOR_CACHE[key]


class CalculatorResource(dg.ConfigurableResource):
    """Holds the calculator selection; `get_calculator()` is load-once (see cache above)."""

    spec: CalculatorConfig

    def get_calculator(self) -> Calculator:
        return _get_or_build_calculator(self.spec)


class OptimiserResource(dg.ConfigurableResource):
    """Base: builds a BFGS optimiser and attaches the shared calculator.

    Mirrors compute_strain.py, which instantiates one calculator and assigns it to both
    optimisers.
    """

    calculator: CalculatorResource
    spec: OptimiserConfig

    def get_calculator(self) -> Calculator:
        return self.calculator.get_calculator()

    def get_optimiser(self) -> Optimiser:
        optimiser = self.spec.build()
        optimiser.calculator = self.get_calculator()
        return optimiser


class LocalOptimiserResource(OptimiserResource):
    """Local optimiser (hydra default.yaml: local_optimiser); defaults bound at registration."""


class GlobalOptimiserResource(OptimiserResource):
    """Global optimiser (hydra default.yaml: global_optimiser); defaults bound at registration."""


class _ChunkedIOManager(dg.ConfigurableIOManager):
    """IO at {base_path}/{asset}/result[_{chunk}].{EXT}.

    Paths are keyed on asset name + chunk (no run id), so a fan-in asset reads every chunk
    regardless of which run produced it — needed when chunks are materialised as separate
    (parallel) runs. ``load_input`` returns a single object for an identity dependency, or the
    combined result for a fan-in over chunks.
    """

    base_path: str
    EXT: ClassVar[str]

    def _file(self, asset: str, chunk: str | None) -> str:
        suffix = f"_{chunk}" if chunk is not None else ""
        return f"{self.base_path}/{asset}/result{suffix}.{self.EXT}"

    def handle_output(self, context: dg.OutputContext, obj: object):
        chunk = context.asset_partition_key if context.has_asset_partitions else None
        path = self._file(context.asset_key.path[-1], chunk)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._dump(obj, path)

    def load_input(self, context: dg.InputContext) -> object:
        asset = context.asset_key.path[-1]
        chunks = context.asset_partition_keys if context.has_asset_partitions else [None]
        objs = [self._load(self._file(asset, c)) for c in chunks]
        return objs[0] if len(objs) == 1 else self._combine(objs)

    def _dump(self, obj, path: str) -> None:
        ...

    def _load(self, path: str):
        ...

    def _combine(self, objs: list):
        return objs


class PyTorchIOManager(_ChunkedIOManager):
    EXT: ClassVar[str] = "pt"

    def _dump(self, obj, path):
        torch.save(obj, path)

    def _load(self, path):
        return torch.load(path, weights_only=False)


class ParquetIOManager(_ChunkedIOManager):
    EXT: ClassVar[str] = "parquet"

    def _dump(self, obj, path):
        obj.to_parquet(path)

    def _load(self, path):
        return pd.read_parquet(path)

    def _combine(self, objs):
        return pd.concat(objs, ignore_index=True)


@dg.definitions
def resources():
    calculator = CalculatorResource(  # default
        spec=MACEConfig(model_paths="../models/MACE_SPICE2_NEUTRAL.model"),
    )
    return dg.Definitions(
        executor=dg.in_process_executor,  # required for load-once calculator cache
        resources={
            # IO Managers specify how the OUTPUT of an asset is handled.
            # The INPUT is handled by the upstream asset's output manager.
            "io_manager": dg.FilesystemIOManager(),  # default if not specified.
            "pandas_io_manager": ParquetIOManager(base_path=DATA_DIR),
            "pytorch_io_manager": PyTorchIOManager(base_path=DATA_DIR),
            # Calculator + optimisers (config-driven; calculator shared).
            "calculator": calculator,
            "local_optimiser": LocalOptimiserResource(
                calculator=calculator, spec=BFGSConfig(fmax=0.50, fexit=5)
            ),
            "global_optimiser": GlobalOptimiserResource(
                calculator=calculator, spec=BFGSConfig(fmax=0.05, fexit=25)
            ),
        },
    )
