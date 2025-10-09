from neural_optimiser.conformers import ConformerBatch
from neural_optimiser.datasets import ConformerDataLoader, ConformerDataset
from neural_optimiser.optimiser.base import Optimiser


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
    result_batches = [optimiser.run(batch) for batch in dataloader]

    # Combine the list of result batches into a single ConformerBatch
    # Note: This assumes you have a way to concatenate batches. If not,
    # you'll need to implement that logic. A placeholder is shown below.
    final_results = ConformerBatch.cat(result_batches)

    return _remove_non_converged(final_results)


def _remove_non_converged(conformers: ConformerBatch) -> ConformerBatch:
    # TODO: extensive logging on how many conformers converged for each conformer and how/why
    pass
