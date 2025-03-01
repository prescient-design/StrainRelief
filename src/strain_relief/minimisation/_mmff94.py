import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdForceFieldHelpers

from strain_relief.utils import remove_non_converged


def MMFF94_min(
    mols: dict[str : Chem.Mol],
    MMFFGetMoleculeProperties: dict,
    MMFFGetMoleculeForceField: dict,
    Minimize: dict,
) -> tuple[dict[str : dict[str:float]], dict[str : Chem.Mol]]:
    """Minimise all conformers of all molecules using UFF.

    Parameters
    ----------
    mols : dict[str:Chem.Mol]
        Dictionary of molecules to minimise.
    MMFFGetMoleculeProperties: dict
        Additional keyword arguments to pass to the MMFFGetMoleculeProperties function.
    MMFFGetMoleculeForceField: dict
        Additional keyword arguments to pass to the MMFFGetMoleculeForceField function.
    Minimize: dict
        Additional keyword arguments to pass to the Minimize function (e.g. maxIters).

    Returns
    -------
    energies, mols : dict[str:dict[str: float]], dict[str:Chem.Mol]
        energies is a dict of final energy of each molecular conformer in eV (i.e. 0 = converged).
        mols contains the dictionary of molecules with the conformers minimised.

        energies = {
            "mol_id": {
                "conf_id": energy
            }
        }
    """
    energies = {}
    for id, mol in mols.items():
        if mol.GetNumBonds() == 0:
            rdDetermineBonds.DetermineBonds(mol)
        energies[id] = _MMFF94_min(
            mol, id, MMFFGetMoleculeProperties, MMFFGetMoleculeForceField, Minimize
        )
    return energies, mols


def _MMFF94_min(
    mol: Chem.Mol,
    id: str,
    MMFFGetMoleculeProperties: dict,
    MMFFGetMoleculeForceField: dict,
    Minimize: dict,
) -> dict[int:float]:
    """Minimise a conformers of a single molecule using UFF.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to minimise.
    id : str
        ID of the molecule. Used for logging.
    MMFFGetMoleculeProperties: dict
        Additional keyword arguments to pass to the MMFFGetMoleculeProperties function.
    MMFFGetMoleculeForceField: dict
        Additional keyword arguments to pass to the MMFFGetMoleculeForceField function.
    Minimize: dict
        Additional keyword arguments to pass to the Minimize function (e.g. maxIters).

    Returns
    -------
    dict[int: float]
        The final energy of each sucessfully converged conformer in the molecule in kcal/mol.
        {conf_id, energy}
    """
    results = []
    for conf in mol.GetConformers():
        mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, **MMFFGetMoleculeProperties)
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, mp, confId=conf.GetId(), **MMFFGetMoleculeForceField
        )
        results.append((ff.Minimize(**Minimize), ff.CalcEnergy()))
    # Remove non-converged conformers
    energies = [E for (converged, E) in remove_non_converged(mol, id, results)]
    energies = {conf.GetId(): E for conf, E in zip(mol.GetConformers(), energies)}
    return energies
