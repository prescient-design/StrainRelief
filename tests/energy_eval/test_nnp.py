import ase
import numpy as np
import pytest
from rdkit import Chem
from strain_relief.constants import EV_TO_KCAL_PER_MOL
from strain_relief.energy_eval._nnp import NNP_energy, _NNP_energy


@pytest.mark.gpu
@pytest.mark.parametrize(
    "method, model_path, energies",
    [
        ("FAIRChem", "esen_model_path", "esen_energies"),
        ("MACE", "mace_model_path", "mace_energies"),
    ],
)
def test_nnp_energy(
    mols_wo_bonds: dict[str, Chem.Mol], method: str, model_path: str, energies: list[float], request
):
    model_path = request.getfixturevalue(model_path)
    energies = request.getfixturevalue(energies)
    mols = mols_wo_bonds

    result = NNP_energy(
        mols,
        method,
        model_paths=str(model_path),
        calculator_kwargs={
            "model_paths": str(model_path),
            "device": "cuda",
            "default_dtype": "float32",
        },
        energy_units="eV",
    )
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == len(mols)

    for id, mol in result.items():
        assert isinstance(mol, dict)
        assert len(mol) == mols[id].GetNumConformers()

        for conf_id, energy in mol.items():
            assert isinstance(conf_id, int)
            assert isinstance(energy, float)

    expected = {
        "00G_3FUJ_A_710_unminimized": {0: energies["0"]},
        "02Z_3RZB_A_458_unminimized": {0: energies["1"]},
    }

    assert result.keys() == expected.keys()
    for mol_id in result.keys():
        for conf_id in result[mol_id].keys():
            assert np.isclose(result[mol_id][conf_id], expected[mol_id][conf_id], atol=1e-6), (
                f"{result[mol_id][conf_id]} != {expected[mol_id][conf_id]} "
                f"(diff = {result[mol_id][conf_id] - expected[mol_id][conf_id]})"
            )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "calculator, energies",
    [("esen_calculator", "esen_energies"), ("mace_calculator", "mace_energies")],
)
def test__NNP_energy(
    mol_wo_bonds_w_confs: Chem.Mol, calculator: ase.calculators, energies: list[float], request
):
    calculator = request.getfixturevalue(calculator)
    energies = request.getfixturevalue(energies)
    mol = mol_wo_bonds_w_confs

    result = _NNP_energy(mol, "id", calculator, EV_TO_KCAL_PER_MOL)
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == mol.GetNumConformers()

    for conf_id, energy in result.items():
        assert isinstance(conf_id, int)
        assert isinstance(energy, float)

    expected = {0: energies["0"], 1: energies["0"]}

    assert result.keys() == expected.keys()
    for conf_if in result.keys():
        assert np.isclose(result[conf_id], expected[conf_id], atol=1e-6), (
            f"{result[conf_id]} != {expected[conf_id]} "
            f"(diff = {result[conf_id] - energies[conf_id]})"
        )
