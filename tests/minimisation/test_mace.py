import pytest
from mace.calculators import MACECalculator
from rdkit import Chem

from strain_relief.minimisation._mace import MACE_min, _MACE_min, run_minimisation
from strain_relief.utils import rdkit_to_ase


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


@pytest.mark.gpu
def test__MACE_min(mol_w_confs: Chem.Mol, model_path: str):
    calculator = MACECalculator(model_paths=str(model_path), device="cuda", default_dtype="float32")
    energies, mol = _MACE_min(
        mol_w_confs,
        id="0",
        calculator=calculator,
        maxIters=1,
        fmax=0.05,
        fexit=250,
        conversion_factor=1,
    )
    energies, mol = _MACE_min(
        mol_w_confs,
        id="0",
        calculator=calculator,
        maxIters=1,
        fmax=0.05,
        fexit=250,
        conversion_factor=1,
    )
    # Conformers will not have been minimised in 1 iteration and so will be removed.
    assert energies == {}
    assert mol.GetNumConformers() == 0


@pytest.mark.gpu
@pytest.mark.parametrize(
    "maxIters, fmax, fexit, expected",
    [
        (100, 1.0, 250, 0),  # should converge
        (1, 0.05, 250, 1),  # not converge (steps > maxIters)
        (100, 0.05, 0.05, 1),  # not converge (forces > fexit)
    ],
)
def test_run_minimisation(
    mol: Chem.Mol, model_path: str, maxIters: int, fmax: float, fexit: float, expected: int
):
    calculator = MACECalculator(
        model_paths=str(model_path), device="cuda", default_dtype="float32", fmax=fmax
    )
    [(_, conf)] = rdkit_to_ase(mol)
    _, converged, _ = run_minimisation(conf, calculator, maxIters, fmax, fexit)
    assert converged == expected
