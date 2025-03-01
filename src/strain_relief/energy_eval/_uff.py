import logging

# TODO: importing AllChem prevents TypeError: No Python class registered for C++ class ForceFields::PyForceField ???
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers


def UFF_energy(mols: dict[str : Chem.Mol], UFFGetMoleculeForceField: dict) -> dict[dict]:
    """Calculate the UFF energy for all conformers of all molecules.

    Parameters
    ----------
    mols : dict[str:Chem.Mol]
        A dictionary of molecules.
    UFFGetMoleculeForceField : dict
        Keyword arguments for the UFF force field to use for energy calculation.

    Returns
    -------
    dict[str: dict[int: float]]
        A dictionary of dictionaries of conformer energies for each molecule.

        mol_energies = {
            "mol_id": {
                "conf_id": energy
            }
        }
    """
    mol_energies = {}
    for id, mol in mols.items():
        mol_energies[id] = _UFF_energy(mol, id, UFFGetMoleculeForceField)
    return mol_energies


def _UFF_energy(mol: Chem.Mol, id: str, UFFGetMoleculeForceField: dict) -> dict[int:float]:
    """Calculate the UFF energy for all conformers of a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        A molecule.
    id : str
        ID of the molecule. Used for logging.
    UFFGetMoleculeForceField : dict
        Keyword arguments for the UFF force field to use for energy calculation.

    Returns
    -------
    dict[int: float]
        A dictionary of conformer energies.

        conf_energies = {
            "conf_id": energy
    """
    conformer_energies = {}
    for conf in mol.GetConformers():
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(
            mol, confId=conf.GetId(), **UFFGetMoleculeForceField
        )
        conformer_energies[conf.GetId()] = ff.CalcEnergy()
        logging.debug(
            f"{id}: Minimised conformer {conf.GetId()} energy = {conformer_energies[conf.GetId()]} kcal/mol"
        )
    return conformer_energies
