from neural_optimiser.conformers import ConformerBatch
from neural_optimiser.optimisers import BFGS
from strain_relief.optimisation._optimisation import run_optimisation


def test_run_optimisation_no_batching(opt: BFGS, batch: ConformerBatch) -> ConformerBatch:
    """Test running optimisation without batching."""
    out = run_optimisation(conformers=batch, optimiser=opt, batch_size=1, num_workers=0)

    assert isinstance(out, ConformerBatch)
    assert out.n_conformers == batch.n_conformers
    assert out.n_atoms == batch.n_atoms


def test_run_optimisation_batching(opt: BFGS, batch: ConformerBatch) -> ConformerBatch:
    """Test running optimisation with batching."""
    out = run_optimisation(conformers=batch, optimiser=opt, batch_size=3, num_workers=0)

    assert isinstance(out, ConformerBatch)
    assert out.n_conformers == batch.n_conformers
    assert out.n_atoms == batch.n_atoms
