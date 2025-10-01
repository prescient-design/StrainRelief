from ase.units import Ang, Bohr, Hartree, eV, kcal, mol

# Energy

# 1 Ha = 627.503 kcal/mol
HARTREE_TO_KCAL_PER_MOL: float = Hartree / (kcal / mol)
KCAL_PER_MOL_TO_HARTREE: float = (kcal / mol) / Hartree

# 1 Ha = 27.2107 eV
HARTREE_TO_EV: float = Hartree / eV
EV_TO_HARTREE: float = eV / Hartree

# 1 eV = 23.0609 kcal/mol
EV_TO_KCAL_PER_MOL: float = eV / (kcal / mol)
KCAL_PER_MOL_TO_EV: float = (kcal / mol) / eV

# 1 Bohr = 0.529177 Angstrom
BOHR_TO_ANGSTROM: float = Bohr / Ang
ANGSTROM_TO_BOHR: float = Ang / Bohr
