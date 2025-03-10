import numpy as np
import pytest
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from strain_relief import test_dir
from strain_relief.calculators import RDKitMMFFCalculator
from strain_relief.constants import KCAL_PER_MOL_TO_EV


def test_RDKitMMFFCalculator():
    # Example usage
    atoms = read(test_dir / "data" / "water.xyz")
    calc = RDKitMMFFCalculator()
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # Now double check the results with RDKit
    raw_mol = Chem.MolFromXYZFile(str(test_dir / "data" / "water.xyz"))
    conn_mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(conn_mol, charge=0)
    mmff_props = AllChem.MMFFGetMoleculeProperties(conn_mol)
    mmff_forcefield = AllChem.MMFFGetMoleculeForceField(conn_mol, mmff_props)
    rdkit_energy = mmff_forcefield.CalcEnergy()
    rdkit_grad = mmff_forcefield.CalcGrad()

    assert abs(energy - rdkit_energy) < 1e-6, f"Energy are not equal: {energy} != {rdkit_energy}"
    assert np.array_equal(
        np.array(forces).reshape(3, 3), np.array(rdkit_grad).reshape(3, 3) * -KCAL_PER_MOL_TO_EV
    )


@pytest.mark.skip(reason="Not yet implemented")
def test_RDKitMMFFCalculator_dynamics():
    pass
