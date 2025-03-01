import logging
import tempfile
from typing import Literal

import ase
from mace.calculators import MACECalculator
from rdkit import Chem

from strain_relief.constants import EV_TO_KCAL_PER_MOL, HARTREE_TO_KCAL_PER_MOL
from strain_relief.io.utils import copy_from_s3
from strain_relief.minimisation.utils import StrainReliefBFGS
from strain_relief.utils import ase_to_rdkit, rdkit_to_ase, remove_non_converged


def MACE_min(
    mols: dict[str : Chem.Mol],
    model_paths: str,
    maxIters: int,
    fmax: float,
    fexit: float,
    default_dtype: str,
    device=Literal["cpu", "cuda"],
    mace_energy_units: Literal["eV", "Hartrees", "kcal/mol"] = "eV",
) -> tuple[dict[str : dict[str:float]], dict[str : Chem.Mol]]:
    """Minimise all conformers of a Chem.Mol using MACE.

    Parameters
    ----------
    mols : dict[str:Chem.Mol]
        Dictionary of molecules to minimise.
    model_path : str
        Path to the MACE model to use for energy calculation.
    maxIters : int
        Maximum number of iterations for the minimisation.
    fmax : float
        Convergence criteria, converged when max(forces) < fmax.
    fexit : float
        Exit criteria, exit when max(forces) > fexit.
    default_dtype : str
        The default data type to use for energy calculation.
    device : Literal["cpu", "cuda"]
        The device to use for energy calculation.
    mace_energy_units: Literal["eV", "Hartrees", "kcal/mol"]
        The units output from the energy calculation.

    energies, mols : dict[str:dict[str: float]], dict[str:Chem.Mol]
        energies is a dict of final energy of each molecular conformer in eV (i.e. 0 = converged).
        mols contains the dictionary of molecules with the conformers minimised.

        energies = {
            "mol_id": {
                "conf_id": energy
            }
        }
    """
    if model_paths.startswith("s3://"):
        local_path = tempfile.mktemp(suffix=".model")
        copy_from_s3(model_paths, local_path)
        model_paths = local_path

    if mace_energy_units == "eV":
        conversion_factor = EV_TO_KCAL_PER_MOL
        logging.info("MACE model outputs energies in eV. Converting to kcal/mol.")
    elif mace_energy_units == "Hartrees":
        conversion_factor = HARTREE_TO_KCAL_PER_MOL
        logging.info("MACE model outputs energies in Hartrees. Converting to kcal/mol.")
    elif mace_energy_units == "kcal/mol":
        conversion_factor = 1
        logging.info("MACE model outputs energies in kcal/mol. No conversion needed.")

    calculator = MACECalculator(model_paths=model_paths, device=device, default_dtype=default_dtype)

    energies = {}
    for id, mol in mols.items():
        energies[id], mols[id] = _MACE_min(
            mol, id, calculator, maxIters, fmax, fexit, conversion_factor
        )
    return energies, mols


def _MACE_min(
    mol: Chem.Mol,
    id: str,
    calculator: ase.calculators,
    maxIters: int,
    fmax: float,
    fexit: float,
    conversion_factor: float,
) -> dict[int:float]:
    """Minimise a conformers of a single molecule using MACE.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to minimise.
    calculator : ase.calculators
        The ASE calculator to use for energy calculation.
    maxIters : int
        The maximum number of iterations for the minimisation.
    fmax : float
        Convergence criteria, converged when max(forces) < fmax.
    fexit : float
        Exit criteria, exit when max(forces) > fexit.
    conversion_factor: float
        Scale factor to convert energy to kcal/mol.

    Returns
    -------
    dict[int: float]
        The final energy of each sucessfully converged conformer in the molecule in kcal/mol.
        {conf_id, energy}
    """
    results = []
    conf_id_and_conf_min = []

    conf_id_and_conf = rdkit_to_ase(mol)

    for conf_id, conf in conf_id_and_conf:
        new_conf, converged, energy = run_minimisation(conf, calculator, maxIters, fmax, fexit)
        results.append(tuple([converged, energy]))
        conf_id_and_conf_min.append(tuple([conf_id, new_conf]))

    mol = ase_to_rdkit(conf_id_and_conf_min)

    energies = [E * conversion_factor for (converged, E) in remove_non_converged(mol, id, results)]
    energies = {conf.GetId(): E for conf, E in zip(mol.GetConformers(), energies)}
    return energies, mol


def run_minimisation(
    atoms: ase.Atoms,
    calculator: ase.calculators,
    maxIters: int,
    fmax: float = 0.05,
    fexit: float = 250,
) -> tuple[ase.Atoms, int, float]:
    """Run the minimisation of a single conformer using the given calculator.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to minimise.
    calculator : ase.calculators
        The ASE calculator to use for energy calculation.
    maxIters : int
        The maximum number of iterations for the minimisation.
    fmax : float
        Convergence criteria, converged when max(forces) < fmax.
    fexit : float
        Exit criteria, exit when max(forces) > fexit.

    Returns
    -------
    atoms : ase.Atoms
        The ASE Atoms object after minimisation.
    int
        The convergence status of the minimisation (0 = converged).
    float
        The final energy of the minimised conformer
        (Note: this is in eV be default as MACE is trained on eV).
    """
    atoms.calc = calculator
    dyn = StrainReliefBFGS(atoms)
    converged = dyn.run(fmax=fmax, fexit=fexit, steps=maxIters)
    return (
        atoms,
        int(not converged),
        atoms.get_potential_energy(),
    )  # doesn't have to recalculate energy
