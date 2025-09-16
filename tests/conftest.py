import os

import numpy as np
import pytest
from strain_relief import test_dir
from strain_relief.calculators import fairchem_calculator as FAIRChem_calculator
from strain_relief.calculators import mace_calculator as MACE_calculator
from strain_relief.constants import EV_TO_KCAL_PER_MOL, MOL_KEY
from strain_relief.io import load_parquet, to_mols_dict


@pytest.fixture(scope="function")
def mols() -> dict[str, dict]:
    """Two posed molecules from an internal target."""
    df = load_parquet(parquet_path=test_dir / "data" / "target.parquet", id_col_name="SMILES")
    return to_mols_dict(df, "", "mol", "SMILES", True)


@pytest.fixture(scope="function")
def mol(mols) -> dict:
    k = list(mols.keys())[0]
    return mols[k]


@pytest.fixture(scope="function")
def mols_w_confs(mols) -> dict[str, dict]:
    """Two posed molecules from an internal target.

    Each molecule has two conformers."""
    for mol_properties in mols.values():
        m = mol_properties["mol"]
        m.AddConformer(m.GetConformer(0), assignId=True)
    return mols


@pytest.fixture(scope="function")
def mol_w_confs(mol) -> dict:
    """Two posed molecules from an internal target.

    Each molecule has two conformers."""
    mol[MOL_KEY].AddConformer(mol[MOL_KEY].GetConformer(0), assignId=True)
    return mol


## LIGBOUNDCONF TEST MOLECULES
@pytest.fixture(scope="function")
def mols_wo_bonds() -> dict[str, dict]:
    """This is two bound conformers taken from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds."""
    df = load_parquet(parquet_path=test_dir / "data" / "ligboundconf.parquet")
    return to_mols_dict(df, "", "mol", "id", True)


@pytest.fixture(scope="function")
def mol_wo_bonds(mols_wo_bonds) -> dict:
    """Bound conformer from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds."""
    k = list(mols_wo_bonds.keys())[0]
    return mols_wo_bonds[k]


@pytest.fixture(scope="function")
def mols_wo_bonds_w_confs(mols_wo_bonds) -> dict[str, dict]:
    """Two bound conformers from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds.
    Each molecule has two conformers."""
    for mol_properties in mols_wo_bonds.values():
        m = mol_properties[MOL_KEY]
        m.AddConformer(m.GetConformer(0), assignId=True)
    return mols_wo_bonds


@pytest.fixture(scope="function")
def mol_wo_bonds_w_confs(mol_wo_bonds) -> dict:
    """Bound conformer from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds.
    Has two conformers."""
    mol_wo_bonds[MOL_KEY].AddConformer(mol_wo_bonds[MOL_KEY].GetConformer(0), assignId=True)
    return mol_wo_bonds


@pytest.fixture(scope="session")
def mace_energies() -> list[float]:
    """The MACE energies as calculated using the mace repo (in eV)."""
    return {
        idx: E
        for idx, E in zip(
            ["0", "1"], np.array([-19786.040533272728, -29390.87077464851]) * EV_TO_KCAL_PER_MOL
        )
    }


@pytest.fixture(scope="session")
def esen_energies() -> list[float]:
    """The MACE energies as calculated using the mace repo (in eV)."""
    return {
        idx: E
        for idx, E in zip(
            ["0", "1"], np.array([-19772.31732206841, -29376.818942909442]) * EV_TO_KCAL_PER_MOL
        )
    }


@pytest.fixture(scope="session")
def mace_model_path() -> str:
    """This is the MACE_SPICE2_NEUTRAL.model"""
    return str(test_dir / "models" / "MACE.model")


@pytest.fixture(scope="session")
def esen_model_path() -> str:
    """This is the OMol25 eSEN small conserving model."""
    if os.path.exists(test_dir / "models" / "eSEN.pt"):
        return str(test_dir / "models" / "eSEN.pt")
    return pytest.skip(f"eSEN model not found at {test_dir / 'models' / 'eSEN.pt'}")


@pytest.fixture(scope="session")
def mace_calculator(mace_model_path):
    """The MACE ASE calculator."""
    return MACE_calculator(model_paths=mace_model_path, device="cuda", default_dtype="float32")


@pytest.fixture(scope="session")
def fairchem_calculator(esen_model_path):
    """The eSEN ASE calculator."""
    return FAIRChem_calculator(model_paths=esen_model_path, device="cuda", default_dtype="float32")
