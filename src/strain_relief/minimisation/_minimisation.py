from timeit import default_timer as timer
from typing import Literal

from loguru import logger as logging
from rdkit import Chem

from strain_relief.constants import ENERGY_PROPERTY_NAME
from strain_relief.minimisation import MACE_min, MMFF94_min, eSEN_min

METHODS_DICT = {"MACE": MACE_min, "eSEN": eSEN_min, "MMFF94": MMFF94_min, "MMFF94s": MMFF94_min}


def minimise_conformers(
    mols: dict[str : Chem.Mol], method: Literal["MACE", "eSEN", "MMFF94s", "MMFF94"], **kwargs
) -> dict[str : Chem.Mol]:
    """Minimise all conformers of all molecules using a force field.

    Parameters
    ----------
    mols : dict[str:Chem.Mol]
        Dictionary of molecules to minimise.
    method : Literal["MACE", "eSEN", "MMFF94s", "MMFF94"]
        Method to use for minimisation.
    kwargs : dict
        Additional keyword arguments to pass to the minimisation function.

    Returns
    -------
    mols : dict[str:Chem.Mol]
        List of molecules with the conformers minimised.
    """
    start = timer()

    if method not in METHODS_DICT:
        raise ValueError(f"method must be in {METHODS_DICT.keys()}")

    logging.info(f"Minimising conformers using {method} and removing non-converged conformers...")
    # Select method and run minimisation
    min_method = METHODS_DICT[method]
    energies, mols = min_method(mols, **kwargs)

    # Store the predicted energies as a property on each conformer
    for id, mol in mols.items():
        [
            mol.GetConformer(conf_id).SetDoubleProp(ENERGY_PROPERTY_NAME, energy)
            for conf_id, energy in energies[id].items()
        ]
    logging.info(
        f"Predicted energies stored as '{ENERGY_PROPERTY_NAME}' property on each conformer"
    )

    no_confs = sum([mol.GetNumConformers() == 0 for mol in mols.values()])
    if no_confs > 0:
        logging.warning(f"{no_confs} molecules have 0 converged confomers after minimisation.")

    end = timer()
    logging.info(f"Conformers minimisation took {end - start:.2f} seconds. \n")

    return mols
