import tempfile
from typing import Literal

from loguru import logger as logging

from strain_relief.calculators import CALCULATORS_DICT
from strain_relief.constants import EV_TO_KCAL_PER_MOL, HARTREE_TO_KCAL_PER_MOL
from strain_relief.io.utils_s3 import copy_from_s3
from strain_relief.minimisation.utils_minimisation import method_min
from strain_relief.types import EnergiesDict, MolsDict


def NNP_min(
    mols: MolsDict,
    method: Literal["MACE", "FAIRChem"],
    calculator_kwargs: dict,
    model_paths: str,
    maxIters: int,
    fmax: float,
    fexit: float,
    energy_units: Literal["eV", "Hartrees", "kcal/mol"] = "eV",
) -> tuple[EnergiesDict, MolsDict]:
    """Minimise all conformers of a Chem.Mol using a NNP.

    Parameters
    ----------
    mols : MolsDict
        Dictionary of molecules to minimise.
    method : Literal["MACE", "FAIRChem"]
        The NNP to use for MD calculation.
    calculator_kwargs : dict
        Additional keyword arguments to pass to the NNP calculator.
        For example, for MACE, this should include `model_path`, `device` and `default_dtype`.
    model_path : str
        Path to the NNP model to use for MD calculation.
    maxIters : int
        Maximum number of iterations for the minimisation.
    fmax : float
        Convergence criteria, converged when max(forces) < fmax.
    fexit : float
        Exit criteria, exit when max(forces) > fexit.
    energy_units: Literal["eV", "Hartrees", "kcal/mol"]
        The units output from the energy calculation.

    energies, mols : EnergiesDict, MolsDict
        energies is a dict of final energy of each molecular conformer in eV (i.e. 0 = converged).
        mols contains the dictionary of molecules with the conformers minimised.

        energies = {
            "mol_id": {
                "conf_id": energy
            }
        }
    """
    # Check if model_paths is an S3 path and copy to local if so
    if model_paths.startswith("s3://"):
        local_path = tempfile.mktemp(suffix=".model")
        copy_from_s3(model_paths, local_path)
        model_paths = local_path

    # Set up conversion factor based on energy units
    if energy_units == "eV":
        conversion_factor = EV_TO_KCAL_PER_MOL
        logging.info(f"{method} model outputs energies in eV. Converting to kcal/mol.")
    elif energy_units == "Hartrees":
        conversion_factor = HARTREE_TO_KCAL_PER_MOL
        logging.info(f"{method} model outputs energies in Hartrees. Converting to kcal/mol.")
    elif energy_units == "kcal/mol":
        conversion_factor = 1
        logging.info(f"{method} model outputs energies in kcal/mol. No conversion needed.")

    # Initialise the calculator
    if method not in CALCULATORS_DICT:
        raise ValueError(f"method must be in {CALCULATORS_DICT.keys()}")
    calculator = CALCULATORS_DICT[method](**calculator_kwargs)

    energies, mols = method_min(mols, calculator, maxIters, fmax, fexit, conversion_factor)

    return energies, mols
