import numpy as np
import pytest
from rdkit import Chem

from strain_relief.utils import remove_non_converged


@pytest.mark.parametrize(
    "results, expected",
    [([(0, -10), (1, -5)], np.array([[0, -10]])), ([(1, -5), (1, -5)], np.empty((0, 2)))],
)
def test_remove_non_converged(
    mol_w_confs: Chem.Mol, results: list[tuple[int, float]], expected: list[tuple[int, float]]
):
    mol = mol_w_confs
    results = remove_non_converged(mol, "id", results)

    assert mol.GetNumConformers() == len(results)
    assert all([not_converged == 0 for not_converged, E in results])
    assert np.array_equal(results, expected)
