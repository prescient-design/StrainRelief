import numpy as np
import pytest
from strain_relief.conformers import generate_conformers
from strain_relief.constants import MOL_KEY
from strain_relief.types import MolPropertiesDict


@pytest.mark.parametrize("fixture", ["mol", "mol_wo_bonds"])
def test_generate_conformers(request, fixture: MolPropertiesDict):
    mol = request.getfixturevalue(fixture)
    initial_mol = mol[MOL_KEY]

    initial_num_conformers = initial_mol.GetNumConformers()
    initial_conformer = initial_mol.GetConformer(0).GetPositions()

    mols = generate_conformers({"0": mol, "1": mol})
    first_mol = mols["0"][MOL_KEY]

    final_num_conformers = first_mol.GetNumConformers()
    final_conformer = first_mol.GetConformer(0).GetPositions()

    assert final_num_conformers > initial_num_conformers
    assert np.array_equal(final_conformer, initial_conformer)

    n_confs = [mol_properties[MOL_KEY].GetNumConformers() for mol_properties in mols.values()]
    # If DetermineBonds() fails only 2 confs generated, original and nan.
    assert any([n > 2 for n in n_confs])


@pytest.mark.parametrize("fixture", ["mol_w_confs", "mol_wo_bonds_w_confs"])
def test_generate_conformers_multiple_initial_confs(request, fixture: dict):
    mol = request.getfixturevalue(fixture)
    with pytest.raises(ValueError):
        generate_conformers({"0": mol})
