from collections import Counter
from timeit import default_timer as timer
from typing import Literal

import numpy as np
from loguru import logger as logging
from rdkit.Chem import AllChem, rdDetermineBonds

from strain_relief.constants import CHARGE_KEY, MOL_KEY
from strain_relief.types import MolsDict


def generate_conformers(
    mols: MolsDict,
    EmbedMultipleConfs: dict,
    device: Literal["cpu", "cuda"],
) -> MolsDict:
    """Generate conformers for a molecule. The 0th conformer is the original molecule.

    This function uses RDKit's ETKDGv2 method to generate conformers with the execption of
    clearConfs=False.

    Parameters
    ----------
    mols : MolsDict
            Nested dictionary of molecules for which to generate conformers.
    EmbedMultipleConfs : dict
        Additional keyword arguments to pass to the EmbedMultipleConfs function.
        For example: `numConfs`, `maxAttempts`, `pruneRmsThresh` and `randomSeed`.
    device : Literal["cpu", "cuda"]
        Device to run the conformer generation on (determines whether to use RDKit or nvMolKit).

    Returns
    -------
    MolsDict
        Nested dictionary of molecules with multiple conformers.
    """
    start = timer()
    # Check that each molecule only has one conformer before generation.
    n_conformers = np.array(
        [mol_properties[MOL_KEY].GetNumConformers() for mol_properties in mols.values()]
    )
    if not np.all((n_conformers == 1) | (n_conformers == 0)):
        logging.error(f"Conformer counts: {dict(Counter(n_conformers))}")
        raise ValueError("Some molecules have more than one conformer before conformer generation.")

    logging.info("Generating conformers...")

    # Add bonds if missing
    for id, mol_properties in mols.items():
        mol = mol_properties[MOL_KEY]
        charge = mol_properties[CHARGE_KEY]
        if mol.GetNumBonds() == 0:
            logging.debug(f"Adding bonds to {id}")
            rdDetermineBonds.DetermineBonds(mol, charge=charge)

    # Generate conformers
    if device == "cuda":
        _generate_conformers_cuda(mols, **EmbedMultipleConfs)
    elif device == "cpu":
        _generate_conformers_cpu(mols, **EmbedMultipleConfs)
    else:
        raise ValueError(f"Unknown device: {device}")

    n_conformers = np.array(
        [mol_properties[MOL_KEY].GetNumConformers() for mol_properties in mols.values()]
    )
    numConfs = EmbedMultipleConfs["numConfs"] if "numConfs" in EmbedMultipleConfs else 10
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
    logging.info("Generating conformers with GPU enabled nvMolKit...")
    try:
        from nvmolkit.embedMolecules import EmbedMolecules as nvMolKitEmbed
    except ImportError:
        raise ImportError(
            "nvMolKit is required for GPU based conformer generation. "
            "Install from https://github.com/NVIDIA-Digital-Bio/nvMolKit "
            "or set cfg.conformers.device = 'cpu' to use RDKit conformer generation."
        )

    mol_list = [mol_properties[MOL_KEY] for mol_properties in mols.values()]
    nvMolKitEmbed(mol_list, **kwargs)
    for i, id in enumerate(mols.keys()):
        mols[id][MOL_KEY] = mol_list[i]
        logging.debug(f"{mols[id][MOL_KEY].GetNumConformers()} conformers generated for {id}")
    return mols


def _generate_conformers_cpu(mols, **kwargs):
    """RDKit based conformer generation on CPU."""
    logging.info("Generating conformers with CPU enabled RDKit...")
    for id, mol_properties in mols.items():
        mol = mol_properties[MOL_KEY]
        AllChem.EmbedMultipleConfs(
            mol,
            **kwargs,
        )
        logging.debug(f"{mol.GetNumConformers()} conformers generated for {id}")
    return mols
