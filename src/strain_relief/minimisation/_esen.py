import tempfile
from typing import Literal

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from loguru import logger as logging
from rdkit import Chem

from strain_relief.constants import EV_TO_KCAL_PER_MOL, HARTREE_TO_KCAL_PER_MOL
from strain_relief.io.utils_s3 import copy_from_s3
from strain_relief.minimisation.utils_minimisation import method_min


def eSEN_min(
    mols: dict[str : Chem.Mol],
    model_paths: str,
    maxIters: int,
    fmax: float,
    fexit: float,
    device=Literal["cpu", "cuda"],
    energy_units: Literal["eV", "Hartrees", "kcal/mol"] = "eV",
) -> tuple[dict[str : dict[str:float]], dict[str : Chem.Mol]]:
    """Minimise all conformers of a Chem.Mol using eSEN.

    Parameters
    ----------
    mols : dict[str:Chem.Mol]
        Dictionary of molecules to minimise.
    model_path : str
        Path to the eSEN model to use for energy calculation.
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
    energy_units: Literal["eV", "Hartrees", "kcal/mol"]
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

    if energy_units == "eV":
        conversion_factor = EV_TO_KCAL_PER_MOL
        logging.info("eSEN model outputs energies in eV. Converting to kcal/mol.")
    elif energy_units == "Hartrees":
        conversion_factor = HARTREE_TO_KCAL_PER_MOL
        logging.info("eSEN model outputs energies in Hartrees. Converting to kcal/mol.")
    elif energy_units == "kcal/mol":
        conversion_factor = 1
        logging.info("eSEN model outputs energies in kcal/mol. No conversion needed.")

    esen_predictor = load_predict_unit(path=model_paths, device=device)
    calculator = FAIRChemCalculator(esen_predictor, task_name="omol")

    energies, mols = method_min(mols, calculator, maxIters, fmax, fexit, conversion_factor)

    return energies, mols
