from typing import Literal

from mace.calculators import MACECalculator


def esen_calculator(
    model_paths: str,
    device: str = Literal["cpu", "cuda"],
    default_dtype: Literal["float32", "float64"] = "float32",
    **kwargs,
):
    try:  # Required while mace-torch and fairchem-core have conflicting e3nn dependencies.
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
    except ImportError:
        raise ImportError(
            "fairchem is required for esen_calculator(). "
            "Install with: pip install --force-reinstall e3nn==0.5 fairchem-core"
        )

    esen_predictor = load_predict_unit(path=model_paths, device=device)
    calculator = FAIRChemCalculator(esen_predictor, task_name="omol", **kwargs)

    if default_dtype == "float32":
        if hasattr(calculator, "predictor") and hasattr(calculator.predictor, "model"):
            calculator.predictor.model = calculator.predictor.model.float()

    return calculator


def mace_calculator(
    model_paths: str,
    device: str = Literal["cpu", "cuda"],
    default_dtype: Literal["float32", "float64"] = "float64",
    **kwargs,
):
    return MACECalculator(
        model_paths=model_paths, device=device, default_dtype=default_dtype, **kwargs
    )
