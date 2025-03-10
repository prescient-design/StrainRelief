from ase import Atoms
from rdkit import Chem


def rdkit_to_ase(mol: Chem.Mol) -> list[tuple[int, Atoms]]:
    """Convert an RDKit molecule to an ASE Atoms object.

    Parameters
    ----------
    mol : RDKit molecule
        The RDKit molecule to convert (with multiple conformers).

    Returns
    -------
    list[tuple[int, Atoms]]
        A list of tuples containing the conformer ID and the ASE Atoms object.
    """
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    conf_id_and_conf = [
        (conf.GetId(), Atoms(numbers=atomic_numbers, positions=conf.GetPositions()))
        for conf in mol.GetConformers()
    ]
    return conf_id_and_conf


def ase_to_rdkit(conf_id_and_conf: list[tuple[int, Atoms]]) -> Chem.Mol:
    """Convert a list of ASE Atoms objects to an RDKit molecule.

    Parameters
    ----------
    confs : list[tuple[int, Atoms]]
        A list of tuples containing the conformer ID and the ASE Atoms object.

    Returns
    -------
    Chem.Mol
        The RDKit molecule (with multiple conformers).
    """
    atomic_numbers = conf_id_and_conf[0][1].get_atomic_numbers()
    mol = Chem.RWMol()
    for atomic_num in atomic_numbers:
        atom = Chem.Atom(int(atomic_num))
        mol.AddAtom(atom)

    for conf_id, ase_atoms in conf_id_and_conf:
        conf = Chem.Conformer(len(atomic_numbers))
        for i, pos in enumerate(ase_atoms.get_positions()):
            conf.SetAtomPosition(i, pos)
        conf.SetId(conf_id)
        mol.AddConformer(conf, assignId=True)

    return mol
