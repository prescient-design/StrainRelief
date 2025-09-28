from collections import Counter
from timeit import default_timer as timer
from typing import Literal

import numpy as np
from loguru import logger as logging
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds


def generate_conformers(
    device: Literal["cuda", "cpu"],
    mols: dict[str : Chem.Mol],
    randomSeed: int = -1,
    numConfs: int = 10,
    maxAttempts: int = 200,
    pruneRmsThresh: float = 0.1,
    clearConfs: bool = False,
    numThreads: int = 0,
    **kwargs,
) -> dict[str : Chem.Mol]:
    """Generate conformers for a molecule. The 0th conformer is the original molecule.

    This function uses RDKit's ETKDGv2 method to generate conformers with the execption of
    clearConfs=False.

    Parameters
    ----------
    mols : dict[str:Chem.Mol]
            Dictionary of molecules for which to generate conformers.
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
    dict[str:Chem.Mol]
            List of molecules with multiple conformers.
    """
    start = timer()
    # Check that each molecule only has one conformer before generation.
    n_conformers = np.array([mol.GetNumConformers() for mol in mols.values()])
    if not np.all((n_conformers == 1) | (n_conformers == 0)):
        logging.error(f"Conformer counts: {dict(Counter(n_conformers))}")
        raise ValueError("Some molecules have more than one conformer before conformer generation.")

    logging.info("Generating conformers...")

    if device == "cuda":
        _generate_conformers_cuda(
            mols=mols,
            randomSeed=randomSeed,
            numConfs=numConfs,
            maxAttempts=maxAttempts,
            pruneRmsThresh=pruneRmsThresh,
            clearConfs=clearConfs,
            numThreads=numThreads,
            **kwargs,
        )
    elif device == "cpu":
        _generate_conformers_cpu(
            mols=mols,
            randomSeed=randomSeed,
            numConfs=numConfs,
            maxAttempts=maxAttempts,
            pruneRmsThresh=pruneRmsThresh,
            clearConfs=clearConfs,
            numThreads=numThreads,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown device: {device}")

    n_conformers = np.array([mol.GetNumConformers() for mol in mols.values()])
    logging.info(
        f"{np.sum(n_conformers == numConfs + 1)} molecules with {numConfs + 1} conformers each"
    )
    logging.info(f"Avg. number of conformers is {np.mean(n_conformers):.1f}")
    logging.info(
        f"Min. number of conformers is {np.min(n_conformers) if len(n_conformers) > 0 else np.nan}"
    )

    end = timer()
    logging.info(f"Conformer generation took {end - start:.2f} seconds. \n")

    return mols


def _generate_conformers_cuda(mols, **kwargs):
    """nvMolKit based conformer generation on GPU."""
    for id, mol in mols.items():
        if mol.GetNumBonds() == 0:
            logging.debug(f"Adding bonds to {id}")
            rdDetermineBonds.DetermineBonds(mol)
        AllChem.EmbedMultipleConfs(
            mol,
            **kwargs,
        )
        logging.debug(f"{mol.GetNumConformers()} conformers generated for {id}")
    return mols


def _generate_conformers_cpu(mols, **kwargs):
    """RDKit based conformer generation on CPU."""
    for id, mol in mols.items():
        if mol.GetNumBonds() == 0:
            logging.debug(f"Adding bonds to {id}")
            rdDetermineBonds.DetermineBonds(mol)
        AllChem.EmbedMultipleConfs(
            mol,
            **kwargs,
        )
        logging.debug(f"{mol.GetNumConformers()} conformers generated for {id}")
    return mols
