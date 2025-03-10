import pytest
from rdkit import Chem
from strain_relief.minimisation._mace import MACE_min


@pytest.mark.gpu
def test_MACE_min(mols: dict[str, Chem.Mol], model_path: str):
    energies, mols = MACE_min(
        mols,
        str(model_path),
        maxIters=1,
        default_dtype="float32",
        device="cuda",
        fmax=0.05,
        fexit=250,
    )
    # Conformers will not have been minimised in 1 iteration and so will be removed.
    assert all([energy == {} for energy in energies.values()])
    assert all([mol.GetNumConformers() == 0 for mol in mols.values()])
