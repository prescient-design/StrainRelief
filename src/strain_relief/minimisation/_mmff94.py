from strain_relief.calculators import RDKitMMFFCalculator
from strain_relief.minimisation.utils_minimisation import method_min
from strain_relief.types import EnergiesDict, MolsDict


def MMFF94_min(
    mols: MolsDict,
    method: str,
    MMFFGetMoleculeProperties: dict,
    MMFFGetMoleculeForceField: dict,
    maxIters: int,
    fmax: float,
    fexit: float,
) -> tuple[EnergiesDict, MolsDict]:
    """Minimise all conformers of a Chem.Mol using MMFF94(s).

    Parameters
    ----------
    mols : MolsDict
        Dictionary of molecules to minimise.
    method : str
        [PLACEHOLDER] Needed for NNP_min compatibility.
    MMFFGetMoleculeProperties: dict
        Additional keyword arguments to pass to the MMFFGetMoleculeProperties function.
    MMFFGetMoleculeForceField: dict
        Additional keyword arguments to pass to the MMFFGetMoleculeForceField function.
    maxIters : int
        Maximum number of iterations for the minimisation.
    fmax : float
        Convergence criteria, converged when max(forces) < fmax.
    fexit : float
        Exit criteria, exit when max(forces) > fexit.

    energies, mols : EnergiesDict, MolsDict
        energies is a dict of final energy of each molecular conformer in eV (i.e. 0 = converged).
        mols contains a nested dictionary of molecules with the conformers minimised.

        energies = {
            "mol_id": {
                "conf_id": energy
            }
        }
    """
    calculator = RDKitMMFFCalculator(
        MMFFGetMoleculeProperties=MMFFGetMoleculeProperties,
        MMFFGetMoleculeForceField=MMFFGetMoleculeForceField,
    )
    energies, mols = method_min(mols, calculator, maxIters, fmax, fexit)

    return energies, mols
