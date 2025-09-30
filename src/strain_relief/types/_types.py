from typing import NewType

from ase import Atoms

MolsDict = NewType("MolsDict", dict[str, dict])
"""
mols = {
    "mol_id": {
        "charge": int,
        "spin": int,
        "mol": RDKit.Mol,
    }
}
"""

MolPropertiesDict = NewType("MolPropertiesDict", dict)
"""
mol_properties = {
    "charge": int,
    "spin": int,
    "mol": RDKit.Mol,
}
"""


EnergiesDict = NewType("EnergiesDict", dict[str, dict[str, float]])
"""
energies = {
    "mol_id": {
        "conf_id": float
    }
}
"""

ConfEnergiesDict = NewType("ConfEnergiesDict", dict[int, float])
"""
conf_energies = {
    "conf_id": float
}
"""

ConformerASEList = list[tuple[int, Atoms]]
"""
conformer_ase_list = [
    (conf_id, Atoms),
]
"""
