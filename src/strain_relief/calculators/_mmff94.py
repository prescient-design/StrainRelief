from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from strain_relief.constants import KCAL_PER_MOL_TO_EV
from strain_relief.io import ase_to_rdkit


def mmff94_calculator(
    MMFFGetMoleculeProperties: Mapping | None = None,
    MMFFGetMoleculeForceField: Mapping | None = None,
    **kwargs: Any,
) -> "RDKitMMFFCalculator":
    return RDKitMMFFCalculator(
        MMFFGetMoleculeProperties=MMFFGetMoleculeProperties,
        MMFFGetMoleculeForceField=MMFFGetMoleculeForceField,
        **kwargs,
    )


class RDKitMMFFCalculator(Calculator):
    implemented_properties: list[str] = ["energy", "forces"]

    def __init__(
        self,
        MMFFGetMoleculeProperties: Mapping | None = None,
        MMFFGetMoleculeForceField: Mapping | None = None,
        **kwargs: Any,
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
        self.MMFFGetMoleculeProperties: Mapping[str, Any] = dict(MMFFGetMoleculeProperties or {})
        self.MMFFGetMoleculeForceField: Mapping[str, Any] = dict(MMFFGetMoleculeForceField or {})
        self.bond_info: list[tuple[int, int, Any]] | None = None
        self.smiles: str | None = None

    def calculate(
        self,
        atoms: Atoms,
        properties: Sequence[str] = ("energy", "forces"),
        system_changes: Any = all_changes,
    ) -> None:
        """Calculate properties.

        Energies are in kcal/mol and forces are in eV/Å.

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to calculate the energy and forces for, by default None
        properties : list, optional
            The properties to calculate, by default ["energy", "forces"]
        system_changes : int, optional
            The system changes to calculate, by default all_changes
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        charge: int = atoms.info.get("charge", 0)

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
            if self.bond_info:
                for BeginAtomIdx, EndAtomIdx, BondType in self.bond_info:
                    mol.AddBond(BeginAtomIdx, EndAtomIdx, BondType)
        Chem.SanitizeMol(mol)

        # Calculate MMFF energy
        mp = AllChem.MMFFGetMoleculeProperties(mol, **self.MMFFGetMoleculeProperties)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, **self.MMFFGetMoleculeForceField)
        energy: float = ff.CalcEnergy()
        grad = ff.CalcGrad()

        self.results["energy"] = energy
        self.results["forces"] = np.array(grad, dtype=float).reshape(-1, 3) * -KCAL_PER_MOL_TO_EV
