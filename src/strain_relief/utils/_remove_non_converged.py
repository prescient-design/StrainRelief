import logging

import numpy as np
from rdkit import Chem


def remove_non_converged(
    mol: Chem.Mol, id: str, results: list[tuple[int, float]]
) -> list[tuple[int, float]]:
    """Remove non-converged conformers from a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to remove non-converged conformers from.
    id : str
        ID of the molecule. Used for logging.
    results : list[tuple[int, float]]
        A binary not_converged flag and the final energy of the molecule in kcal/mol (i.e. 0 = converged) for each conformer.

    Results
    -------
    np.array[tuple[int, float]]
    """
    not_converged = np.array(
        [True if not_converged == 1 else False for (not_converged, _) in results]
    )
    if not_converged.sum() == 0:
        logging.debug(f"All conformers converged sucessfully for {id}")
    else:
        logging.debug(
            f"{mol.GetNumConformers() - not_converged.sum()}/{mol.GetNumConformers()} conformers converged sucessfully for {id}"
        )
    confs_to_remove = np.array([conf.GetId() for conf in mol.GetConformers()])[not_converged]
    for conf_id in confs_to_remove:
        mol.RemoveConformer(int(conf_id))

    results = np.array(results)[~not_converged]
    return results
