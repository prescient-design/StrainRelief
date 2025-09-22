from timeit import default_timer as timer
from typing import Literal

from loguru import logger as logging

from strain_relief.constants import ENERGY_PROPERTY_NAME, MOL_KEY
from strain_relief.energy_eval import MMFF94_energy, NNP_energy
from strain_relief.types import MolsDict

METHODS_DICT = {
    "MACE": NNP_energy,
    "FAIRChem": NNP_energy,
    "MMFF94": MMFF94_energy,
    "MMFF94s": MMFF94_energy,
}


def predict_energy(
    mols: MolsDict, method: Literal["MACE", "FAIRChem", "MMFF94", "MMFF94s"], **kwargs
):
    """Predict the energy of all conformers of molecules in mols using a specified method.

    Parameters
    ----------
    mols : MolsDict
        Nested dictionary of molecules.
    method : Literal["MACE", "FAIRChem", "MMFF94", "MMFF94s"]
        The method to use for energy prediction.
    **kwargs
        Additional keyword arguments to pass to the energy prediction method.

    Returns
    -------
    MolsDict
        Nested dictionary of molecules with the predicted energies stored as a property on each
        conformer.
    """
    start = timer()

    if method not in METHODS_DICT:
        raise ValueError(f"method must be in {METHODS_DICT.keys()}")

    logging.info(f"Predicting energies using {method}")
    # Select method and run energy evaluation
    energy_method = METHODS_DICT[method]
    energies = energy_method(mols, method, **kwargs)

    # Store the predicted energies as a property on each conformer
    for id, mol_properties in mols.items():
        [
            mol_properties[MOL_KEY]
            .GetConformer(conf_id)
            .SetDoubleProp(ENERGY_PROPERTY_NAME, energy)
            for conf_id, energy in energies[id].items()
        ]
    logging.info(
        f"Predicted energies stored as '{ENERGY_PROPERTY_NAME}' property on each conformer"
    )

    end = timer()
    logging.info(f"Energy prediction took {end - start:.2f} seconds. \n")

    return mols
