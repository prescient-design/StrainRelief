from ase import Atoms
from rdkit import Chem

from strain_relief.types import ConformerASEList


def rdkit_to_ase(mol: Chem.Mol) -> ConformerASEList:
    """Convert an RDKit molecule (with conformers) to ASE Atoms objects.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule containing one or more conformers.

    Returns
    -------
    ConformerASEList
        List of (conformer_id, Atoms) tuples.
    """
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    conf_id_and_conf: ConformerASEList = [
        (conf.GetId(), Atoms(numbers=atomic_numbers, positions=conf.GetPositions()))
        for conf in mol.GetConformers()
    ]
    if mol.GetNumBonds() > 0:
        bond_info = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in mol.GetBonds()
        ]
        for _, conf in conf_id_and_conf:
            conf.info["bond_info"] = bond_info
    return conf_id_and_conf


def ase_to_rdkit(conf_id_and_conf: ConformerASEList) -> Chem.Mol:
    """Convert a list of ASE Atoms conformers back into an RDKit molecule.

    Parameters
    ----------
    conf_id_and_conf : ConformerASEList
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

    if "bond_info" in ase_atoms.info:
        for BeginAtomIdx, EndAtomIdx, BondType in ase_atoms.info["bond_info"]:
            mol.AddBond(BeginAtomIdx, EndAtomIdx, BondType)

    return mol
