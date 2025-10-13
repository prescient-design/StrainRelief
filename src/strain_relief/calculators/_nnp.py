from typing import Any, Literal

from ase.calculators.calculator import Calculator


def fairchem_calculator(
    model_paths: str,
    device: str = Literal["cpu", "cuda"],
    default_dtype: Literal["float32", "float64"] = "float32",
    **kwargs: Any,
) -> Calculator:
    """Create a FAIRChemCalculator using a pre-trained model.

    Parameters
    ----------
    model_paths : str
        Path to the pre-trained model.
    device : Literal["cpu", "cuda"]
        Device to run the model on.
    default_dtype : Literal["float32", "float64"], optional
        Default data type for the model. Default is "float32".

    Returns
    -------
    Calculator
        An instance of the FAIRChemCalculator.
    """
    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
    except ImportError:
        raise ImportError(
            "fairchem is required for fairchem_calculator(). "
            "Install with: pip install --force-reinstall e3nn==0.5 fairchem-core"
        )

    predictor = load_predict_unit(path=model_paths, device=device)
    calculator = FAIRChemCalculator(predictor, task_name="omol", **kwargs)

    if default_dtype == "float32":
        if hasattr(calculator, "predictor") and hasattr(calculator.predictor, "model"):
            calculator.predictor.model = calculator.predictor.model.float()

    return calculator


def mace_calculator(
    model_paths: str,
    device: str = Literal["cpu", "cuda"],
    default_dtype: Literal["float32", "float64"] = "float64",
    **kwargs: Any,
) -> Calculator:
    """Create a MACECalculator using a pre-trained model.

    Parameters
    ----------
    model_paths : str
        Path to the pre-trained model.
    device : Literal["cpu", "cuda"]
        Device to run the model on.
    default_dtype : Literal["float32", "float64"], optional
        Default data type for the model. Default is "float64".

    Returns
    -------
    Calculator
        An instance of the MACECalculator.
    """
    try:
        from mace.calculators import MACECalculator
    except ImportError:
        raise ImportError(
            "mace is required for mace_calculator(). "
            "Install with: pip install mace-torch>=0.3.13"
        )
    return MACECalculator(
        model_paths=model_paths, device=device, default_dtype=default_dtype, **kwargs
    )
