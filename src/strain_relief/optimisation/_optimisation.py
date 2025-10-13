from neural_optimiser.conformers import ConformerBatch
from neural_optimiser.datasets.base import ConformerDataLoader, ConformerDataset
from neural_optimiser.optimisers.base import Optimiser


def run_optimisation(
    conformers: ConformerBatch, optimiser: Optimiser, batch_size: int, num_workers: int
) -> ConformerBatch:
    """
    Helper method to run batch optimisation using a DataLoader.

    Args:
        conformers: A ConformerBatch of all conformers to be optimized.
        optimiser: The optimiser instance to use (local or global).
        batch_size:

    Returns:
        A single ConformerBatch containing all the optimised results.
    """
    dataset = ConformerDataset([conformers.conformer(i) for i in range(conformers.n_conformers)])

    # Use config values for dataloader parameters
    dataloader = ConformerDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Process batches and collect the results
    minimised = []
    for batch in dataloader:
        optimiser.run(batch)
        minimised.append(batch)

    return ConformerBatch.cat(minimised)
