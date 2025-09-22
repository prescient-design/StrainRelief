import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from strain_relief.constants import KCAL_PER_MOL_TO_EV
from strain_relief.io import ase_to_rdkit


def mmff94_calculator(MMFFGetMoleculeProperties: dict, MMFFGetMoleculeForceField: dict, **kwargs):
    return RDKitMMFFCalculator(MMFFGetMoleculeProperties, MMFFGetMoleculeForceField, **kwargs)


class RDKitMMFFCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self, MMFFGetMoleculeProperties: dict = {}, MMFFGetMoleculeForceField: dict = {}, **kwargs
    ):
        """
        RDKit MMFF94(s) ASE Calculator

        Parameters
        ----------
        MMFFGetMoleculeProperties : Dict, optional
            Additional keyword arguments for MMFFGetMoleculeProperties, by default {}
        MMFFGetMoleculeForceField : Dict, optional
            Additional keyword arguments for MMFFGetMoleculeForceField, by default {}
        kwargs
            Additional keyword arguments for Calculator
        """
        Calculator.__init__(self, **kwargs)
        self.MMFFGetMoleculeProperties = MMFFGetMoleculeProperties
        self.MMFFGetMoleculeForceField = MMFFGetMoleculeForceField
        self.bond_info = None
        self.smiles = None

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        """Calculate properties.

        Energies are in kcal/mol and forces are in eV/Å.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            The atoms object to calculate the energy and forces for, by default None
        properties : list, optional
            The properties to calculate, by default ["energy", "forces"]
        system_changes : int, optional
            The system changes to calculate, by default all_changes
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        charge = atoms.info.get("charge", 0)

        mol = ase_to_rdkit([(0, atoms)])

        # Determine bonds for each new molecule. Bond information remains constant during MD.
        new_smiles = Chem.MolToSmiles(mol)
        if new_smiles != self.smiles:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
            self.bond_info = [
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
                for bond in mol.GetBonds()
            ]
            self.smiles = new_smiles
        else:
            for BeginAtomIdx, EndAtomIdx, BondType in self.bond_info:
                mol.AddBond(BeginAtomIdx, EndAtomIdx, BondType)
        Chem.SanitizeMol(mol)

        # Calculate MMFF energy
        mp = AllChem.MMFFGetMoleculeProperties(mol, **self.MMFFGetMoleculeProperties)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, **self.MMFFGetMoleculeForceField)
        energy = ff.CalcEnergy()
        grad = ff.CalcGrad()

        self.results["energy"] = energy
        self.results["forces"] = np.array(grad).reshape(-1, 3) * -KCAL_PER_MOL_TO_EV
