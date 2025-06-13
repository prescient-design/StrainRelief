from loguru import logger as logging
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdForceFieldHelpers


def MMFF94_energy(
    mols: dict[str : Chem.Mol],
    method: str,
    MMFFGetMoleculeProperties: dict,
    MMFFGetMoleculeForceField: dict,
) -> dict[dict]:
    """Calculate the MMFF94(s) energy for all conformers of all molecules.

    Parameters
    ----------
    mols : dict[str:Chem.Mol]
        A dictionary of molecules.
    method : str
        [PLACEHOLDER] Needed for NNP_energy compatibility.
    MMFFGetMoleculeProperties : dict
        Additional keyword arguments for MMFFGetMoleculeProperties.
    MMFFGetMoleculeForceField : dict
        Additional keyword arguments for MMFFGetMoleculeForceField.

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
        if mol.GetNumBonds() == 0:
            rdDetermineBonds.DetermineBonds(mol)
        mol_energies[id] = _MMFF94_energy(
            mol, id, MMFFGetMoleculeProperties, MMFFGetMoleculeForceField
        )
    return mol_energies


def _MMFF94_energy(
    mol: Chem.Mol, id: str, MMFFGetMoleculeProperties: dict, MMFFGetMoleculeForceField: dict
) -> dict[int:float]:
    """Calculate the MMFF94 energy for all conformers of a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        A molecule.
    id : str
        ID of the molecule. Used for logging.
    MMFFGetMoleculeProperties : dict
        Additional keyword arguments for MMFFGetMoleculeProperties.
    MMFFGetMoleculeForceField : dict
        Additional keyword arguments for MMFFGetMoleculeForceField.

    Returns
    -------
    dict[int: float]
        A dictionary of conformer energies.

        conf_energies = {
            "conf_id": energy
    """
    conformer_energies = {}
    for conf in mol.GetConformers():
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
