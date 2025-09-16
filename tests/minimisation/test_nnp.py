import pytest
from strain_relief.constants import MOL_KEY
from strain_relief.minimisation._nnp import NNP_min


@pytest.mark.gpu
@pytest.mark.parametrize(
    "method, model_path",
    [("MACE", "mace_model_path"), ("FAIRChem", "esen_model_path")],
)
def test_NNP_min(mols: dict[str, dict], method: str, model_path: str, request):
    """Test minimisation with NNPs."""
    model_path = request.getfixturevalue(model_path)

    energies, mols = NNP_min(
        mols,
        method,
        calculator_kwargs={
            "model_paths": str(model_path),
            "device": "cuda",
            "default_dtype": "float32",
        },
        model_paths=str(model_path),
        maxIters=1,
        fmax=0.05,
        fexit=250,
    )
    # Conformers will not have been minimised in 1 iteration and so will be removed.
    assert all([energy == {} for energy in energies.values()])
    assert all(
        [mol_properties[MOL_KEY].GetNumConformers() == 0 for mol_properties in mols.values()]
    )
