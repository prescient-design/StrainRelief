from typing import NewType

MolsDict = NewType("MolsDict", dict[str, dict])
"""
mols = {
    "mol_id": {
        "charge": charge_val,
        "spin": spin_val,
        "mol": Chem.Mol,
    }
}
"""

MolPropertiesDict = NewType("MolPropertiesDict", dict)
"""
mol_properties = {
    "charge": charge_val,
    "spin": spin_val,
    "mol": Chem.Mol,
}
"""


EnergiesDict = NewType("EnergiesDict", dict[str, dict[str, float]])
"""
energies = {
    "mol_id": {
        "conf_id": energy
    }
}
"""

ConfEnergiesDict = NewType("ConfEnergiesDict", dict[int, float])
"""
conf_energies = {
    "conf_id": energy
}
"""
