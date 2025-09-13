import numpy as np
import pytest
from rdkit import Chem
from strain_relief.calculators import RDKitMMFFCalculator, mace_calculator
from strain_relief.io import rdkit_to_ase
from strain_relief.minimisation.utils_minimisation import (
    _method_min,
    method_min,
    remove_non_converged,
    run_minimisation,
)


@pytest.mark.gpu
@pytest.mark.parametrize("calculator_fixture", ["mace_calculator", "esen_calculator"])
def test_method_min_nnp(mols: dict[str : Chem.Mol], calculator_fixture: str, request):
    calculator = request.getfixturevalue(calculator_fixture)
    energies, mols = method_min(
        mols,
        calculator,
        maxIters=1,
        fmax=0.05,
        fexit=250,
        conversion_factor=1,
    )
    # Conformers will not have been minimised in 1 iteration and so will be removed.
    assert all([energy == {} for energy in energies.values()])
    assert all([mol.GetNumConformers() == 0 for mol in mols.values()])


@pytest.mark.parametrize("fixture", ["mols", "mols_wo_bonds"])
@pytest.mark.parametrize("force_field", ["MMFF94", "MMFF94s"])
def test_method_min_mmff(request, fixture: dict[str : Chem.Mol], force_field: str):
    mols = request.getfixturevalue(fixture)
    calculator = RDKitMMFFCalculator(
        MMFFGetMoleculeProperties={"mmffVariant": force_field}, MMFFGetMoleculeForceField={}
    )
    energies, mols = method_min(
        mols,
        calculator,
        maxIters=1,
        fmax=0.05,
        fexit=250,
        conversion_factor=1,
    )
    # Conformers will not have been minimised in 1 iteration and so will be removed.
    assert all([energy == {} for energy in energies.values()])
    assert all([mol.GetNumConformers() == 0 for mol in mols.values()])


@pytest.mark.gpu
@pytest.mark.parametrize("calculator_fixture", ["mace_calculator", "esen_calculator"])
def test__method_min_nnp(mol_w_confs: Chem.Mol, calculator_fixture: str, request):
    calculator = request.getfixturevalue(calculator_fixture)
    energies, mol = _method_min(
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


@pytest.mark.parametrize("maxIts", [1, 1000])
@pytest.mark.parametrize("force_field", ["MMFF94", "MMFF94s"])
def test__method_min_mmff(mol_w_confs: Chem.Mol, maxIts: int, force_field: str):
    calculator = RDKitMMFFCalculator(
        MMFFGetMoleculeProperties={"mmffVariant": force_field}, MMFFGetMoleculeForceField={}
    )
    energies, mol = _method_min(
        mol_w_confs,
        id="0",
        calculator=calculator,
        maxIters=maxIts,
        fmax=0.05,
        fexit=250,
        conversion_factor=1,
    )
    assert mol.GetNumConformers() == len(energies)


@pytest.mark.parametrize(
    "results, expected",
    [([(0, -10), (1, -5)], np.array([[0, -10]])), ([(1, -5), (1, -5)], np.empty((0, 2)))],
)
def test_remove_non_converged(
    mol_w_confs: Chem.Mol, results: list[tuple[int, float]], expected: list[tuple[int, float]]
):
    mol = mol_w_confs
    results = remove_non_converged(mol, "id", results)

    assert mol.GetNumConformers() == len(results)
    assert all([not_converged == 0 for not_converged, E in results])
    assert np.array_equal(results, expected)


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
    mol: Chem.Mol, mace_model_path: str, maxIters: int, fmax: float, fexit: float, expected: int
):
    calculator = mace_calculator(
        model_paths=str(mace_model_path), device="cuda", default_dtype="float32", fmax=fmax
    )
    [(_, conf)] = rdkit_to_ase(mol)
    _, converged, _ = run_minimisation(conf, calculator, maxIters, fmax, fexit)
    assert converged == expected
