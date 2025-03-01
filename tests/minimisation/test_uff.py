import numpy as np
import pytest
from rdkit import Chem

from strain_relief.minimisation._uff import UFF_min, _UFF_min


def test_UFF_min(mols: dict[str, Chem.Mol]):
    mols = mols
    results, mols = UFF_min(mols, {"maxIters": 1})

    for id in mols.keys():
        conf_energies = results[id]
        assert len(conf_energies) == mols[id].GetNumConformers()


@pytest.mark.parametrize("maxIters", [1, 1000])
def test__UFF_min(mol_w_confs: Chem.Mol, maxIters: int):
    mol = mol_w_confs
    mol.AddConformer(mol.GetConformer(0), assignId=True)

    energies = _UFF_min(mol, "id", {"maxIters": maxIters})

    assert mol.GetNumConformers() == len(energies)
