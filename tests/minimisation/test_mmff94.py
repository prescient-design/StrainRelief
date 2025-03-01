import numpy as np
import pytest
from rdkit import Chem

from strain_relief.minimisation._mmff94 import MMFF94_min, _MMFF94_min


@pytest.mark.parametrize("fixture", ["mols", "mols_wo_bonds"])
@pytest.mark.parametrize("force_field", ["MMFF94", "MMFF94s"])
def test_MMFF94_min(request, fixture: dict[str, Chem.Mol], force_field: str):
    mols = request.getfixturevalue(fixture)
    results, mols = MMFF94_min(mols, {"mmffVariant": force_field}, {}, {"maxIts": 1})

    for id in mols.keys():
        conf_energies = results[id]
        assert len(conf_energies) == mols[id].GetNumConformers()


@pytest.mark.parametrize("maxIts", [1, 1000])
@pytest.mark.parametrize("force_field", ["MMFF94", "MMFF94s"])
def test__MMFF94_min(mol_w_confs: Chem.Mol, maxIts: int, force_field: str):
    mol = mol_w_confs
    mol.AddConformer(mol.GetConformer(0), assignId=True)

    energies = _MMFF94_min(mol, "id", {"mmffVariant": force_field}, {}, {"maxIts": maxIts})

    assert mol.GetNumConformers() == len(energies)
