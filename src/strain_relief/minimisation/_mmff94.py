from strain_relief.calculators import RDKitMMFFCalculator
from strain_relief.minimisation.utils_minimisation import method_min


def MMFF94_min(
    mols: dict[str:dict],
    method: str,
    MMFFGetMoleculeProperties: dict,
    MMFFGetMoleculeForceField: dict,
    maxIters: int,
    fmax: float,
    fexit: float,
) -> tuple[dict[str : dict[str:float]], dict[str:dict]]:
    """Minimise all conformers of a Chem.Mol using MMFF94(s).

    Parameters
    ----------
    mols : dict[str:dict]
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

    energies, mols : dict[str:dict[str: float]], dict[str:dict]
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
