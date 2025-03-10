import numpy as np
from strain_relief.constants import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    EV_TO_HARTREE,
    EV_TO_KCAL_PER_MOL,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
    KCAL_PER_MOL_TO_EV,
    KCAL_PER_MOL_TO_HARTREE,
)


def test_units():
    assert np.isclose(HARTREE_TO_KCAL_PER_MOL * KCAL_PER_MOL_TO_HARTREE, 1)
    assert np.isclose(HARTREE_TO_EV * EV_TO_HARTREE, 1)
    assert np.isclose(EV_TO_KCAL_PER_MOL * KCAL_PER_MOL_TO_EV, 1)
    assert np.isclose(BOHR_TO_ANGSTROM * ANGSTROM_TO_BOHR, 1)
