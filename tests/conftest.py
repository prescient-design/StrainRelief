import pytest
import torch
from ase.build import molecule
from neural_optimiser.calculators import MACECalculator
from neural_optimiser.conformers import ConformerBatch
from neural_optimiser.optimisers import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem
from strain_relief import test_dir
from strain_relief.constants import MOL_KEY
from strain_relief.data_types import MolPropertiesDict
from strain_relief.io import load_parquet, to_mols_dict

# --------- GENERAL FIXTURES ---------


@pytest.fixture(scope="session")
def device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------- OPTIMISER TESTS FIXTURES ----------


@pytest.fixture(scope="session")
def mace_model_path() -> str:
    """Path to the MACE model used in tests."""
    return str(test_dir / "models" / "MACE.model")


@pytest.fixture(scope="module")
def opt(device: str, mace_model_path: str) -> BFGS:
    """BFGS Optimiser"""
    calculator = MACECalculator(model_paths=mace_model_path, device=device)
    optimiser = BFGS(steps=10, fmax=0.50, fexit=25)
    optimiser.calculator = calculator
    return optimiser


@pytest.fixture(scope="function")
def batch(device: str):
    """ConformerBatch with three small molecules."""
    batch = ConformerBatch.from_ase([molecule("H2O"), molecule("H2O"), molecule("NH3")])
    batch.id = [0, 1, 2]
    batch.to(device)
    return batch


# --------- CONFORMER GENERATION FIXTURES ---------


@pytest.fixture(scope="function")
def mol() -> MolPropertiesDict:
    df = load_parquet(
        parquet_path=test_dir / "data" / "target.parquet",
        id_col_name="SMILES",
        include_charged=True,
    )
    mols_dict = to_mols_dict(df=df, mol_col_name="mol", id_col_name="SMILES", include_charged=True)
    k = list(mols_dict.keys())[0]
    return mols_dict[k]


@pytest.fixture(scope="function")
def mol_w_confs(mol) -> MolPropertiesDict:
    """Two posed molecules from an internal target.

    Each molecule has two conformers."""
    mol[MOL_KEY].AddConformer(mol[MOL_KEY].GetConformer(0), assignId=True)
    return mol


@pytest.fixture(scope="function")
def mol_wo_bonds() -> MolPropertiesDict:
    """Bound conformer from LigBoundConf 2.0.

    Bond information is determined using RDKit's DetermineBonds."""
    df = load_parquet(parquet_path=test_dir / "data" / "ligboundconf.parquet", include_charged=True)
    mols_dict = to_mols_dict(df=df, mol_col_name="mol", id_col_name="id", include_charged=True)
    k = list(mols_dict.keys())[0]
    return mols_dict[k]


# --------- INPUT FIXTURES ---------


@pytest.fixture(scope="function")
def sample_mol_bytes() -> tuple[bytes, bytes]:
    """Two small molecules in bytes format."""
    mol_c = Chem.MolFromSmiles("C")
    mol_o = Chem.MolFromSmiles("O")
    return mol_c.ToBinary(), mol_o.ToBinary()


# --------- OUTPUT FIXTURES ---------


@pytest.fixture(scope="function")
def minimised_batch(batch: ConformerBatch, device: str) -> ConformerBatch:
    """ConformerBatch with three small molecules."""
    batch = ConformerBatch.cat([batch, batch])
    batch.id = [0, 1, 2, 3, 4, 5]
    batch.name = ["H2O", "H2O", "NH3", "H2O", "H2O", "NH3"]
    batch.converged = torch.tensor([False, True, True, True, True, True])
    batch.energies = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
    batch.to(device)
    return batch


# --------- INTEGRATION TEST FIXTURES ---------


@pytest.fixture(scope="function")
def mols_input2() -> list[Chem.Mol]:
    """Two small molecules for integration tests."""
    mols = []
    for smiles in ["C", "CC"]:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mols.append(mol)
    return mols


@pytest.fixture(scope="function")
def mols_input3() -> list[Chem.Mol]:
    """Three small molecules for integration tests."""
    mols = []
    for smiles in ["C", "CC", "CCO"]:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mols.append(mol)
    return mols
