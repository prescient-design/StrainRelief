import pytest
from rdkit import Chem

from strain_relief.energy_eval._uff import UFF_energy, _UFF_energy


@pytest.mark.parametrize("fixture", ["mols", "mols_wo_bonds"])
def test_UFF_energy(request, fixture: dict[str, Chem.Mol]):
    mols = request.getfixturevalue(fixture)
    result = UFF_energy(mols, {})
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == len(mols)

    for id, mol in result.items():
        assert isinstance(mol, dict)
        assert len(mol) == mols[id].GetNumConformers()

        for conf_id, energy in mol.items():
            assert isinstance(conf_id, int)
            assert isinstance(energy, float)


@pytest.mark.parametrize("fixture", ["mol_w_confs", "mol_wo_bonds_w_confs"])
def test__UFF_energy(request, fixture: Chem.Mol):
    mol = request.getfixturevalue(fixture)
    result = _UFF_energy(mol, "id", {})
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == mol.GetNumConformers()

    for conf_id, energy in result.items():
        assert isinstance(conf_id, int)
        assert isinstance(energy, float)
