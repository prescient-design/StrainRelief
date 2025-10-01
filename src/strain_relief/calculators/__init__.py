from collections.abc import Callable

from ._mmff94 import RDKitMMFFCalculator
from ._nnp import fairchem_calculator, mace_calculator

CALCULATORS_DICT: dict[str, Callable] = {
    "MMFF94": RDKitMMFFCalculator,
    "MMFF94s": RDKitMMFFCalculator,
    "FAIRChem": fairchem_calculator,
    "MACE": mace_calculator,
}

__all__ = ["RDKitMMFFCalculator", "fairchem_calculator", "mace_calculator", "CALCULATORS_DICT"]
