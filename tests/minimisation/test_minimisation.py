import pytest
from rdkit import Chem

from strain_relief.constants import ENERGY_PROPERTY_NAME
from strain_relief.minimisation import minimise_conformers


@pytest.mark.parametrize(
    "method, expected_exception, kwargs",
    [
        (
            "UFF",
            None,
            {"UFFOptimizeMoleculeConfs": {"maxIters": 1, "numThreads": 0}, "maxIters": 1},
        ),
        (
            "MMFF94",
            None,
            {
                "MMFFGetMoleculeProperties": {"mmffVariant": "MMFF94"},
                "MMFFGetMoleculeForceField": {},
                "Minimize": {"maxIts": 1},
                "maxIters": 1,
            },
        ),
        (
            "MMFF94s",
            None,
            {
                "MMFFGetMoleculeProperties": {"mmffVariant": "MMFF94s"},
                "MMFFGetMoleculeForceField": {},
                "Minimize": {"maxIts": 1},
                "maxIters": 1,
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
        assert isinstance(result[smile], Chem.Mol)

        for conf in result[smile].GetConformers():
            assert conf.HasProp(ENERGY_PROPERTY_NAME)
