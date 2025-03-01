import pytest
from rdkit import Chem

from strain_relief.energy_eval._mmff94 import MMFF94_energy, _MMFF94_energy


@pytest.mark.parametrize("fixture", ["mols", "mols_wo_bonds"])
@pytest.mark.parametrize("force_field", ["MMFF94", "MMFF94s"])
def test_MMFF94_energy(request, fixture: dict[str, Chem.Mol], force_field: str):
    mols = request.getfixturevalue(fixture)
    result = MMFF94_energy(mols, {"mmffVariant": force_field}, {})
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == len(mols)

    for id, mol in result.items():
        assert isinstance(mol, dict)
        assert len(mol) == mols[id].GetNumConformers()

        for conf_id, energy in mol.items():
            assert isinstance(conf_id, int)
            assert isinstance(energy, float)


@pytest.mark.parametrize("force_field", ["MMFF94", "MMFF94s"])
def test__MMFF94_energy(mol_w_confs: Chem.Mol, force_field: str):
    mol = mol_w_confs
    result = _MMFF94_energy(mol, "id", {"mmffVariant": force_field}, {})
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == mol.GetNumConformers()

    for conf_id, energy in result.items():
        assert isinstance(conf_id, int)
        assert isinstance(energy, float)
