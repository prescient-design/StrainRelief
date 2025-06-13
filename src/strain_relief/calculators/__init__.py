from ._mmff94 import RDKitMMFFCalculator
from ._nnp import esen_calculator, mace_calculator

CALCULATORS_DICT = {
    "MMFF94": RDKitMMFFCalculator,
    "MMFF94s": RDKitMMFFCalculator,
    "eSEN": esen_calculator,
    "MACE": mace_calculator,
}

__all__ = ["RDKitMMFFCalculator", "esen_calculator", "mace_calculator", "CALCULATORS_DICT"]
