import pytest
import torch
from strain_relief import test_dir
from strain_relief.constants import MOL_KEY
from strain_relief.data_types import MolPropertiesDict, MolsDict
from strain_relief.io import load_parquet, to_mols_dict


@pytest.fixture(scope="session")
def device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="function")
def mols() -> MolsDict:
    """Two posed molecules from an internal target."""
    df = load_parquet(
        parquet_path=test_dir / "data" / "target.parquet",
        id_col_name="SMILES",
        include_charged=True,
    )
    return to_mols_dict(df=df, mol_col_name="mol", id_col_name="SMILES", include_charged=True)


@pytest.fixture(scope="function")
def mol(mols) -> MolPropertiesDict:
    k = list(mols.keys())[0]
    return mols[k]


@pytest.fixture(scope="function")
def mol_w_confs(mol) -> MolPropertiesDict:
    """Two posed molecules from an internal target.

    Each molecule has two conformers."""
    mol[MOL_KEY].AddConformer(mol[MOL_KEY].GetConformer(0), assignId=True)
    return mol


# LIGBOUNDCONF TEST MOLECULES
@pytest.fixture(scope="function")
def mols_wo_bonds() -> MolsDict:
    """This is two bound conformers taken from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds."""
    df = load_parquet(parquet_path=test_dir / "data" / "ligboundconf.parquet", include_charged=True)
    return to_mols_dict(df=df, mol_col_name="mol", id_col_name="id", include_charged=True)


@pytest.fixture(scope="function")
def mol_wo_bonds(mols_wo_bonds) -> MolPropertiesDict:
    """Bound conformer from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds."""
    k = list(mols_wo_bonds.keys())[0]
    return mols_wo_bonds[k]


@pytest.fixture(scope="function")
def mol_wo_bonds_w_confs(mol_wo_bonds) -> MolPropertiesDict:
    """Bound conformer from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds.
    Has two conformers."""
    mol_wo_bonds[MOL_KEY].AddConformer(mol_wo_bonds[MOL_KEY].GetConformer(0), assignId=True)
    return mol_wo_bonds


# TODO: only mol_w_confs and mol_wo_bonds_w_confs, mol, mol_wo_bonds are used in conf generation
# tests. Remove unused fixtures?
