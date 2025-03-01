from ase.units import Ang, Bohr, Hartree, eV, kcal, mol

# Energy

# 1 Ha = 627.503 kcal/mol
HARTREE_TO_KCAL_PER_MOL = Hartree / (kcal / mol)
KCAL_PER_MOL_TO_HARTREE = (kcal / mol) / Hartree

# 1 Ha = 27.2107 eV
HARTREE_TO_EV = Hartree / eV
EV_TO_HARTREE = eV / Hartree

# 1 eV = 23.0609 kcal/mol
EV_TO_KCAL_PER_MOL = eV / (kcal / mol)
KCAL_PER_MOL_TO_EV = (kcal / mol) / eV

# 1 Bohr = 0.529177 Angstrom
BOHR_TO_ANGSTROM = Bohr / Ang
ANGSTROM_TO_BOHR = Ang / Bohr
