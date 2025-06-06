import pytest
from rdkit import Chem
from strain_relief.constants import ENERGY_PROPERTY_NAME
from strain_relief.energy_eval import predict_energy


@pytest.mark.parametrize(
    "method, expected_exception, kwargs",
    [
        (
            "MMFF94",
            None,
            {
                "MMFFGetMoleculeProperties": {"mmffVariant": "MMFF94"},
                "MMFFGetMoleculeForceField": {},
            },
        ),
        (
            "MMFF94s",
            None,
            {
                "MMFFGetMoleculeProperties": {"mmffVariant": "MMFF94s"},
                "MMFFGetMoleculeForceField": {},
            },
        ),
        ("XXX", ValueError, {}),
    ],
)
def test_predict_energy(mols: dict[str, Chem.Mol], method: str, expected_exception, kwargs: dict):
    mols = mols
    if expected_exception:
        with pytest.raises(expected_exception):
            predict_energy(mols, method, **kwargs)
    else:
        result = predict_energy(mols, method, **kwargs)
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == len(mols)

        for mol in result.values():
            assert isinstance(mol, Chem.Mol)
            for conf in mol.GetConformers():
                assert conf.HasProp(ENERGY_PROPERTY_NAME)


@pytest.mark.gpu
def test_predict_energy_MACE(mols: dict[str, Chem.Mol], mace_model_path: str):
    mols = mols
    kwargs = {
        "device": "cuda",
        "model_paths": str(mace_model_path),
        "energy_units": "eV",
    }
    result = predict_energy(mols, "MACE", **kwargs)
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == len(mols)

    for mol in result.values():
        assert isinstance(mol, Chem.Mol)
        for conf in mol.GetConformers():
            assert conf.HasProp(ENERGY_PROPERTY_NAME)
