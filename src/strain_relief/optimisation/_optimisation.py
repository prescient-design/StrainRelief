from collections import defaultdict
from timeit import default_timer as timer

import torch
from loguru import logger
from neural_optimiser.conformers import ConformerBatch
from neural_optimiser.datasets.base import ConformerDataLoader, ConformerDataset
from neural_optimiser.optimisers.base import Optimiser


def run_optimisation(
    conformers: ConformerBatch, optimiser: Optimiser, batch_size: int, num_workers: int, device: str
) -> ConformerBatch:
    """Helper method to run batch optimisation using a DataLoader.

    Parameters
    ----------
    conformers: ConformerBatch
        The batch of conformers to be optimised.
    optimiser: Optimiser
        The optimiser to use for the optimisation.
    batch_size: int
        The batch size to use for the DataLoader.
    num_workers: int
        The number of workers to use for the DataLoader.
    device: str
        The device to use for the optimisation.

    Returns
    -------
    ConformerBatch
        A single ConformerBatch containing all the optimised results.
    """
    start = timer()

    dataset = ConformerDataset(conformers.to_data_list())
    dataloader = ConformerDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    minimised = []
    for i, batch in enumerate(dataloader):
        if len(dataloader) > 1:
            logger.info(f"Optimising batch {i}/{len(dataloader)}")
        batch = batch.to(device)
        optimiser.run(batch)
        del batch.pos_dt, batch.forces_dt, batch.energies_dt
        batch = batch.to("cpu")  # free up GPU memory
        minimised.append(batch)

    all_conformers = ConformerBatch.cat(minimised)
    _log_optimisation(all_conformers, optimiser)

    end = timer()
    logger.info(f"Conformers minimisation took {end - start:.2f} seconds. \n")

    return all_conformers


def _log_optimisation(conformers: ConformerBatch, optimiser: Optimiser) -> None:
    """Helper method to log optimisation summary statistics."""
    no_converged = 0

    # Group conformers by molecule ID in a single pass
    confs_by_mol = defaultdict(list)
    for idx in range(conformers.n_conformers):
        mol_id = conformers.id[idx]
        confs_by_mol[mol_id].append(conformers.conformer(idx))

    for mol_id, confs in confs_by_mol.items():
        # Molecule level logging
        n_converged = sum([conf.converged for conf in confs])
        if n_converged == len(confs):
            logger.info(
                f"Molecule ID {mol_id}: All {n_converged} conformers converged after minimisation."
            )
        else:
            logger.info(
                f"Molecule ID {mol_id} has {n_converged} converged conformers after minimisation."
            )
        if n_converged == 0:
            no_converged += 1

        # Conformer level logging
        for i, conf in enumerate(confs):
            fmax = torch.linalg.vector_norm(conf.forces, dim=1).max()
            if conf.converged:
                logger.debug(
                    f"Molecule ID {conf.id}, Conformer {i} converged: "
                    f"Steps={conf.converged_step}, fmax={fmax:.4f}, E={conf.energies:.2f}."
                )
            elif fmax > optimiser.fexit:
                logger.debug(
                    f"Molecule ID {conf.id}, Conformer {i} failed: "
                    f"Steps={optimiser.steps}, fmax={fmax:.4f} (fexit activated)."
                )
            else:
                logger.debug(
                    f"Molecule ID {conf.id}, Conformer {i} failed: "
                    f"Steps={optimiser.steps}, fmax={fmax:.4f} (max steps reached)."
                )

    if no_converged > 0:
        logger.warning(f"{no_converged} molecules have 0 converged conformers after minimisation.")
