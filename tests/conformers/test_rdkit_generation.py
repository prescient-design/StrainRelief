import numpy as np
import pytest
from rdkit import Chem

from strain_relief.conformers import generate_conformers


@pytest.mark.parametrize("fixture", ["mol", "mol_wo_bonds"])
def test_generate_conformers(request, fixture: Chem.Mol):
    mol = request.getfixturevalue(fixture)
    initial_num_conformers = mol.GetNumConformers()
    initial_conformer = mol.GetConformer(0).GetPositions()

    mols = generate_conformers({"0": mol, "1": mol})
    first_mol = mols["0"]

    final_num_conformers = first_mol.GetNumConformers()
    final_conformer = first_mol.GetConformer(0).GetPositions()

    assert final_num_conformers > initial_num_conformers
    assert np.array_equal(final_conformer, initial_conformer)

    n_confs = [mol.GetNumConformers() for mol in mols.values()]
    # If DetermineBonds() fails only 2 confs generated, original and nan.
    assert any([n > 2 for n in n_confs])


@pytest.mark.parametrize("fixture", ["mol_w_confs", "mol_wo_bonds_w_confs"])
def test_generate_conformers_multiple_initial_confs(request, fixture: Chem.Mol):
    mol = request.getfixturevalue(fixture)
    with pytest.raises(ValueError):
        generate_conformers({"0": mol})
