from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from strain_relief.utils import remove_non_converged


def UFF_min(
    mols: dict[str : Chem.Mol], UFFOptimizeMoleculeConfs: dict
) -> tuple[dict[str : dict[str:float]], dict[str : Chem.Mol]]:
    """Minimise all conformers of all molecules using UFF.

    Parameters
    ----------
    mols : dict[str:Chem.Mol]
        Dictionary of molecules to minimise.
    UFFOptimizeMoleculeConfs: dict
        Additional keyword arguments to pass to the minimisation function (e.g. maxIters, numThreads).

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
    energies = {id: _UFF_min(mol, id, UFFOptimizeMoleculeConfs) for id, mol in mols.items()}
    return energies, mols


def _UFF_min(mol: Chem.Mol, id: str, UFFOptimizeMoleculeConfs: dict) -> dict[int:float]:
    """Minimise a conformers of a single molecule using UFF.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to minimise.
    id : str
        ID of the molecule. Used for logging.
    UFFOptimizeMoleculeConfs: dict
        Additional keyword arguments to pass to the minimisation function (e.g. maxIters, numThreads).

    Returns
    -------
    dict[int: float]
        The final energy of each sucessfully converged conformer in the molecule in kcal/mol.
        {conf_id, energy}
    """
    results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol, **UFFOptimizeMoleculeConfs)
    # Remove non-converged conformers
    energies = [E for (converged, E) in remove_non_converged(mol, id, results)]
    energies = {conf.GetId(): E for conf, E in zip(mol.GetConformers(), energies)}
    return energies
