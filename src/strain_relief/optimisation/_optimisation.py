from neural_optimiser.conformers import ConformerBatch
from neural_optimiser.datasets.base import ConformerDataLoader, ConformerDataset
from neural_optimiser.optimisers.base import Optimiser


def run_optimisation(
    conformers: ConformerBatch, optimiser: Optimiser, batch_size: int, num_workers: int
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

    Returns
    -------
    ConformerBatch
        A single ConformerBatch containing all the optimised results.
    """
    dataset = ConformerDataset([conformers.conformer(i) for i in range(conformers.n_conformers)])
    dataloader = ConformerDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    minimised = []
    for batch in dataloader:
        optimiser.run(batch)
        minimised.append(batch)

    return ConformerBatch.cat(minimised)
