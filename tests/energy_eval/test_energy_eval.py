import pytest
from rdkit import Chem
from strain_relief.constants import ENERGY_PROPERTY_NAME, MOL_KEY
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
def test_predict_energy(mols: dict[str, dict], method: str, expected_exception, kwargs: dict):
    mols = mols
    if expected_exception:
        with pytest.raises(expected_exception):
            predict_energy(mols, method, **kwargs)
    else:
        result = predict_energy(mols, method, **kwargs)
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == len(mols)

        for mol_properties in result.values():
            mol = mol_properties[MOL_KEY]
            assert isinstance(mol, Chem.Mol)
            for conf in mol.GetConformers():
                assert conf.HasProp(ENERGY_PROPERTY_NAME)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_path_fixture,architecture",
    [("mace_model_path", "MACE"), ("esen_model_path", "FAIRChem")],
)
def test_predict_energy_nnp(
    mols: dict[str, dict], model_path_fixture: str, architecture: str, request
):
    model_path = request.getfixturevalue(model_path_fixture)
    kwargs = {
        "model_paths": str(model_path),
        "energy_units": "eV",
        "calculator_kwargs": {
            "device": "cuda",
            "default_dtype": "float32",
            "model_paths": str(model_path),
        },
    }
    result = predict_energy(mols, architecture, **kwargs)
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == len(mols)

    for mol_properties in result.values():
        mol = mol_properties[MOL_KEY]
        assert isinstance(mol, Chem.Mol)
        for conf in mol.GetConformers():
            assert conf.HasProp(ENERGY_PROPERTY_NAME)
