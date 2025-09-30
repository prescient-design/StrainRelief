from collections.abc import Mapping
from typing import Any

from loguru import logger as logging
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdForceFieldHelpers

from strain_relief.constants import CHARGE_KEY, MOL_KEY
from strain_relief.types import ConfEnergiesDict, EnergiesDict, MolsDict


def MMFF94_energy(
    mols: MolsDict,
    method: str,
    MMFFGetMoleculeProperties: Mapping[str, Any],
    MMFFGetMoleculeForceField: Mapping[str, Any],
) -> EnergiesDict:
    """Calculate the MMFF94(s) energy for all conformers of all molecules.

    Parameters
    ----------
    mols : MolsDict
        A dictionary of molecules.
    method : str
        [PLACEHOLDER] Needed for NNP_energy compatibility.
    MMFFGetMoleculeProperties : Dict
        Additional keyword arguments for MMFFGetMoleculeProperties.
    MMFFGetMoleculeForceField : Dict
        Additional keyword arguments for MMFFGetMoleculeForceField.

    Returns
    -------
    EnergiesDict
        A dictionary of dictionaries of conformer energies for each molecule.

        mol_energies = {
            "mol_id": {
                "conf_id": energy
            }
        }
    """
    mol_energies = {}
    for id, mol_properties in mols.items():
        mol = mol_properties[MOL_KEY]
        charge = mol_properties[CHARGE_KEY]
        if mol.GetNumBonds() == 0:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        mol_energies[id] = _MMFF94_energy(
            mol, id, MMFFGetMoleculeProperties, MMFFGetMoleculeForceField
        )
    return mol_energies


def _MMFF94_energy(
    mol: Chem.Mol,
    id: str,
    MMFFGetMoleculeProperties: Mapping[str, Any],
    MMFFGetMoleculeForceField: Mapping[str, Any],
) -> ConfEnergiesDict:
    """Calculate the MMFF94 energy for all conformers of a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        A molecule.
    id : str
        ID of the molecule. Used for logging.
    MMFFGetMoleculeProperties : Dict
        Additional keyword arguments for MMFFGetMoleculeProperties.
    MMFFGetMoleculeForceField : Dict
        Additional keyword arguments for MMFFGetMoleculeForceField.

    Returns
    -------
    ConfEnergiesDict
        A dictionary of conformer energies.

        conf_energies = {
            "conf_id": energy
    """
    conformer_energies = {}
    for conf in mol.GetConformers():
        print(mol, MMFFGetMoleculeProperties)
        mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, **MMFFGetMoleculeProperties)
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, mp, confId=conf.GetId(), **MMFFGetMoleculeForceField
        )
        conformer_energies[conf.GetId()] = ff.CalcEnergy()
        logging.debug(
            f"{id}: Minimised conformer {conf.GetId()} "
            f"energy = {conformer_energies[conf.GetId()]} kcal/mol"
        )
    return conformer_energies
