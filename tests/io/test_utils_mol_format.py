from copy import deepcopy

import pytest
from ase import Atoms
from rdkit import Chem
from strain_relief.io import ase_to_rdkit, rdkit_to_ase


@pytest.fixture(scope="function")
def ase_atoms() -> list[tuple[int, Atoms]]:
    symbols = ["H", "O", "H"]
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    return [(0, Atoms(symbols=symbols, positions=positions))]


def test_rdkit_to_ase(mol_w_confs: Chem.Mol):
    rdkit_mol = mol_w_confs
    ase_atoms = rdkit_to_ase(rdkit_mol)
    assert len(ase_atoms[0][1]) == rdkit_mol.GetNumAtoms()

    # Check that the atom positions are the same
    for conf_id, conf in ase_atoms:
        for i, atom in enumerate(conf):
            assert atom.position[0] == rdkit_mol.GetConformer(conf_id).GetAtomPosition(i).x
            assert atom.position[1] == rdkit_mol.GetConformer(conf_id).GetAtomPosition(i).y
            assert atom.position[2] == rdkit_mol.GetConformer(conf_id).GetAtomPosition(i).z


def test_ase_to_rdkit(ase_atoms: list[tuple[int, Atoms]]):
    rdkit_mol = ase_to_rdkit(ase_atoms)
    assert len(ase_atoms[0][1]) == rdkit_mol.GetNumAtoms()

    # Check that the atom positions are the same
    for conf_id, conf in ase_atoms:
        for i, atom in enumerate(conf):
            assert atom.position[0] == rdkit_mol.GetConformer(conf_id).GetAtomPosition(i).x
            assert atom.position[1] == rdkit_mol.GetConformer(conf_id).GetAtomPosition(i).y
            assert atom.position[2] == rdkit_mol.GetConformer(conf_id).GetAtomPosition(i).z


def test_rdkit_to_ase_to_rdkit(mol_w_confs: Chem.Mol):
    rdkit_mol = deepcopy(mol_w_confs)
    ase_atoms = rdkit_to_ase(rdkit_mol)
    rdkit_mol = ase_to_rdkit(ase_atoms)

    # Check that the atom positions are the same
    for conf_id, conf in ase_atoms:
        for i, _ in enumerate(conf):
            assert (
                mol_w_confs.GetConformer(conf_id).GetAtomPosition(i).x
                == rdkit_mol.GetConformer(conf_id).GetAtomPosition(i).x
            )
            assert (
                mol_w_confs.GetConformer(conf_id).GetAtomPosition(i).y
                == rdkit_mol.GetConformer(conf_id).GetAtomPosition(i).y
            )
            assert (
                mol_w_confs.GetConformer(conf_id).GetAtomPosition(i).z
                == rdkit_mol.GetConformer(conf_id).GetAtomPosition(i).z
            )
