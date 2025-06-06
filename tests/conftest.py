import numpy as np
import pytest
from mace.calculators import MACECalculator
from rdkit import Chem
from strain_relief import test_dir
from strain_relief.constants import EV_TO_KCAL_PER_MOL
from strain_relief.io import load_parquet, to_mols_dict


## LITLMOL TEST MOLECULES
@pytest.fixture(scope="function")
def mols() -> dict[str, Chem.Mol]:
    """Two posed molecules from an internal target."""
    df = load_parquet(parquet_path=test_dir / "data" / "target.parquet", id_col_name="SMILES")
    return to_mols_dict(df, "mol", "SMILES")


@pytest.fixture(scope="function")
def mol(mols) -> Chem.Mol:
    k = list(mols.keys())[0]
    return mols[k]


@pytest.fixture(scope="function")
def mols_w_confs(mols) -> dict[str, Chem.Mol]:
    """Two posed molecules from an internal target.

    Each molecule has two conformers."""
    for m in mols.values():
        m.AddConformer(m.GetConformer(0), assignId=True)
    return mols


@pytest.fixture(scope="function")
def mol_w_confs(mol) -> Chem.Mol:
    """Two posed molecules from an internal target.

    Each molecule has two conformers."""
    mol.AddConformer(mol.GetConformer(0), assignId=True)
    return mol


## LIGBOUNDCONF TEST MOLECULES
@pytest.fixture(scope="function")
def mols_wo_bonds() -> dict[str, Chem.Mol]:
    """This is two bound conformers taken from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds."""
    df = load_parquet(parquet_path=test_dir / "data" / "ligboundconf.parquet")
    return to_mols_dict(df, "mol", "id")


@pytest.fixture(scope="function")
def mol_wo_bonds(mols_wo_bonds) -> Chem.Mol:
    """Bound conformer from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds."""
    k = list(mols_wo_bonds.keys())[0]
    return mols_wo_bonds[k]


@pytest.fixture(scope="function")
def mols_wo_bonds_w_confs(mols_wo_bonds) -> dict[str, Chem.Mol]:
    """Two bound conformers from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds.
    Each molecule has two conformers."""
    for m in mols_wo_bonds.values():
        m.AddConformer(m.GetConformer(0), assignId=True)
    return mols_wo_bonds


@pytest.fixture(scope="function")
def mol_wo_bonds_w_confs(mol_wo_bonds) -> Chem.Mol:
    """Bound conformer from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds.
    Has two conformers."""
    mol_wo_bonds.AddConformer(mol_wo_bonds.GetConformer(0), assignId=True)
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
def mace_model_path() -> str:
    """This is the MACE_SPICE2_NEUTRAL.model"""
    return str(test_dir / "models" / "MACE.model")


@pytest.fixture(scope="session")
def calculator(model_path):
    """The MACE ASE calculator."""
    return MACECalculator(model_paths=model_path, device="cuda", default_dtype="float32")
