import ase
import numpy as np
import pytest
from rdkit import Chem
from strain_relief.constants import EV_TO_KCAL_PER_MOL
from strain_relief.energy_eval._mace import MACE_energy, _MACE_energy


@pytest.mark.gpu
def test_MACE_energy(
    mols_wo_bonds: dict[str, Chem.Mol], mace_model_path: str, mace_energies: list[float]
):
    mols = mols_wo_bonds
    result = MACE_energy(mols, str(mace_model_path), device="cuda", energy_units="eV")
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
        "00G_3FUJ_A_710_unminimized": {0: mace_energies["0"]},
        "02Z_3RZB_A_458_unminimized": {0: mace_energies["1"]},
    }

    assert result.keys() == expected.keys()
    for mol_id in result.keys():
        for conf_id in result[mol_id].keys():
            assert np.isclose(result[mol_id][conf_id], expected[mol_id][conf_id], atol=1e-6), (
                f"{result[mol_id][conf_id]} != {expected[mol_id][conf_id]} "
                f"(diff = {result[mol_id][conf_id] - expected[mol_id][conf_id]})"
            )


@pytest.mark.gpu
def test__MACE_energy(
    mol_wo_bonds_w_confs: Chem.Mol, calculator: ase.calculators, mace_energies: list[float]
):
    mol = mol_wo_bonds_w_confs
    result = _MACE_energy(mol, "id", calculator, EV_TO_KCAL_PER_MOL)
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == mol.GetNumConformers()

    for conf_id, energy in result.items():
        assert isinstance(conf_id, int)
        assert isinstance(energy, float)

    expected = {0: mace_energies["0"], 1: mace_energies["0"]}

    assert result.keys() == expected.keys()
    for conf_if in result.keys():
        assert np.isclose(result[conf_id], expected[conf_id], atol=1e-6), (
            f"{result[conf_id]} != {expected[conf_id]} "
            f"(diff = {result[conf_id] - mace_energies[conf_id]})"
        )
