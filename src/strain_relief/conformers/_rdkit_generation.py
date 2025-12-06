from collections import Counter
from timeit import default_timer as timer
from typing import Any

import numpy as np
from loguru import logger as logging
from rdkit.Chem import AllChem, rdDetermineBonds

from strain_relief.constants import CHARGE_KEY, MOL_KEY
from strain_relief.data_types import MolsDict


def generate_conformers(
    mols: MolsDict,
    randomSeed: int = -1,
    numConfs: int = 10,
    maxAttempts: int = 200,
    pruneRmsThresh: float = 0.1,
    clearConfs: bool = False,
    numThreads: int = 0,
    **kwargs: Any,
) -> MolsDict:
    """Generate conformers for a molecule. The 0th conformer is the original molecule.

    This function uses RDKit's ETKDGv2 method to generate conformers with the execption of
    clearConfs=False.

    Parameters
    ----------
    mols : MolsDict
            Nested dictionary of molecules for which to generate conformers.
    randomSeed : int, optional
            The random seed to use. The default is -1.
    numConfs : int, optional
            The number of conformers to generate. The default is 100.
    maxAttempts : int, optional
            The maximum number of attempts to try embedding. The default is 1000.
    pruneRmsThresh : float, optional
            The RMS threshold to prune conformers. The default is 0.1.
    numThreads : int, optional
            The number of threads to use while embedding. This only has an effect if the
            RDKit was built with multi-thread support. If set to zero, the max supported
            by the system will be used. The default is 0.

    Returns
    -------
    MolsDict
        Nested dictionary of molecules with multiple conformers.
    """
    start: float = timer()
    n_conformers: np.ndarray = np.array(
        [mol_properties[MOL_KEY].GetNumConformers() for mol_properties in mols.values()]
    )
    if not np.all((n_conformers == 1) | (n_conformers == 0)):
        logging.error(f"Conformer counts: {dict(Counter(n_conformers))}")
        raise ValueError("Some molecules have more than one conformer before conformer generation.")

    logging.info("Generating conformers...")

    for id, mol_properties in mols.items():
        mol = mol_properties[MOL_KEY]
        charge = mol_properties[CHARGE_KEY]
        if mol.GetNumBonds() == 0:
            logging.debug(f"Adding bonds to {id}")
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        AllChem.EmbedMultipleConfs(
            mol,
            randomSeed=randomSeed,
            numConfs=numConfs,
            maxAttempts=maxAttempts,
            pruneRmsThresh=pruneRmsThresh,
            clearConfs=clearConfs,
            numThreads=numThreads,
            **kwargs,
        )
        logging.debug(f"{mol.GetNumConformers()} conformers generated for {id}")

    n_conformers = np.array(
        [mol_properties[MOL_KEY].GetNumConformers() for mol_properties in mols.values()]
    )
    logging.info(
        f"{np.sum(n_conformers == numConfs + 1)} molecules with {numConfs + 1} conformers each"
    )
    logging.info(f"Avg. number of conformers is {np.mean(n_conformers):.1f}")
    logging.info(
        f"Min. number of conformers is {np.min(n_conformers) if len(n_conformers) > 0 else np.nan}"
    )

    end: float = timer()
    logging.info(f"Conformer generation took {end - start:.2f} seconds. \n")

    return mols
