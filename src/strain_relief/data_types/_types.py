from typing import NewType

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
