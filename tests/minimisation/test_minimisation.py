import pytest
from rdkit import Chem
from strain_relief.constants import ENERGY_PROPERTY_NAME, MOL_KEY
from strain_relief.minimisation import minimise_conformers


@pytest.mark.parametrize(
    "method, expected_exception, kwargs",
    [
        (
            "MMFF94",
            None,
            {
                "MMFFGetMoleculeProperties": {"mmffVariant": "MMFF94"},
                "MMFFGetMoleculeForceField": {},
                "maxIters": 1,
                "fmax": 0.05,
                "fexit": 250,
            },
        ),
        (
            "MMFF94s",
            None,
            {
                "MMFFGetMoleculeProperties": {"mmffVariant": "MMFF94s"},
                "MMFFGetMoleculeForceField": {},
                "maxIters": 1,
                "fmax": 0.05,
                "fexit": 250,
            },
        ),
        ("XXX", ValueError, {}),
    ],
)
def test_minimise_conformers(
    method: str, expected_exception, kwargs: dict, mols: dict[str, Chem.Mol]
):
    mols = mols
    smile = "CN(c1n[nH]c2nc(OC3CC3)ccc12)S(=O)(=O)c1cccc(Cl)c1F"
    if expected_exception:
        with pytest.raises(expected_exception):
            minimise_conformers(mols, method, **kwargs)
    else:
        result = minimise_conformers(mols, method, **kwargs)
        assert result is not None
        assert isinstance(result, dict)
        assert isinstance(result[smile], dict)
        assert isinstance(result[smile][MOL_KEY], Chem.Mol)

        for conf in result[smile][MOL_KEY].GetConformers():
            assert conf.HasProp(ENERGY_PROPERTY_NAME)
